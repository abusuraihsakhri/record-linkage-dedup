"""
Record Linkage Deduplicator
============================

Identifies likely duplicate patient records across one or more datasets using
the Fellegi-Sunter (1969) probabilistic record linkage model, with a
Winkler-style EM step to estimate agreement/disagreement weights and
Jaro-Winkler string similarity for fuzzy comparison of name fields.

Usage (CLI):
    python record_linkage.py --input a.csv b.csv --output report.csv --em

Run `python record_linkage.py --help` for all options.
"""

import argparse
import itertools
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# String similarity: Jaro and Jaro-Winkler
# ---------------------------------------------------------------------------

def jaro_similarity(s1: str, s2: str) -> float:
    """Classic Jaro string similarity in [0, 1]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(0, max(len1, len2) // 2 - 1)

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0


def jaro_winkler_similarity(
    s1: str,
    s2: str,
    prefix_scale: float = 0.1,
    max_prefix: int = 4,
    boost_threshold: float = 0.7,
) -> float:
    """Jaro-Winkler similarity: boosts the Jaro score for strings that share
    a common prefix, as in Winkler (1990). The boost is applied only when the
    base Jaro score already exceeds `boost_threshold`, per the original method.
    """
    if s1 == s2:
        return 1.0
    jaro = jaro_similarity(s1, s2)
    if jaro <= boost_threshold:
        return jaro

    prefix = 0
    for c1, c2 in zip(s1, s2):
        if c1 != c2:
            break
        prefix += 1
        if prefix == max_prefix:
            break

    return jaro + prefix * prefix_scale * (1 - jaro)


# ---------------------------------------------------------------------------
# Field comparison functions
# ---------------------------------------------------------------------------

def compare_name(a, b) -> float:
    """Fuzzy name comparison via Jaro-Winkler similarity."""
    if pd.isna(a) or pd.isna(b):
        return 0.0
    a_s, b_s = str(a).strip().upper(), str(b).strip().upper()
    if a_s == "" or b_s == "":
        return 0.0
    return jaro_winkler_similarity(a_s, b_s)


def compare_dob(a, b) -> float:
    """Date-of-birth comparison with partial credit for near matches
    (e.g. transposed day/month, a common data-entry error), so full
    agreement requires an exact date match while close dates still score
    above zero.
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    if a == b:
        return 1.0
    score = 0.0
    if a.year == b.year:
        score += 1.0 / 3.0
    if a.month == b.month:
        score += 1.0 / 3.0
    if a.day == b.day:
        score += 1.0 / 3.0
    if a.year == b.year and a.month == b.day and a.day == b.month and a.month != a.day:
        score = max(score, 2.0 / 3.0)
    return score


def compare_identifier(a, b) -> float:
    """Exact/near comparison for identifier fields (e.g. SSN, MRN):
    exact string match scores 1.0, otherwise falls back to Jaro-Winkler
    similarity so single-digit typos still score close to (but below) 1.0.
    """
    if pd.isna(a) or pd.isna(b):
        return 0.0
    a_s, b_s = str(a).strip().upper(), str(b).strip().upper()
    if a_s == "" or b_s == "":
        return 0.0
    if a_s == b_s:
        return 1.0
    return jaro_winkler_similarity(a_s, b_s)


# ---------------------------------------------------------------------------
# Field specification
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    key: str
    column: str
    comparator: Callable[[object, object], float]
    threshold: float   # similarity >= threshold counts as "agree" (gamma=1)
    default_m: float   # P(agree | true match)
    default_u: float   # P(agree | true non-match)


def build_field_specs(args, df_a: pd.DataFrame, df_b: Optional[pd.DataFrame]) -> List[FieldSpec]:
    frames = [df_a] + ([df_b] if df_b is not None else [])

    def present(col: str) -> bool:
        return all(col in f.columns for f in frames)

    specs: List[FieldSpec] = []
    if present(args.first_name_col):
        specs.append(FieldSpec("first_name", args.first_name_col, compare_name, args.name_threshold, 0.9, 0.10))
    if present(args.last_name_col):
        specs.append(FieldSpec("last_name", args.last_name_col, compare_name, args.name_threshold, 0.9, 0.10))
    if present(args.dob_col):
        specs.append(FieldSpec("dob", args.dob_col, compare_dob, args.dob_threshold, 0.9, 0.02))
    if present(args.identifier_col):
        specs.append(FieldSpec("identifier", args.identifier_col, compare_identifier, args.id_threshold, 0.95, 0.005))
    return specs


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------

def make_block_key(row: pd.Series, first_name_col: str, dob_col: str) -> str:
    """Blocking key: birth year + first-name initial. Limits comparisons to
    record pairs that plausibly could match, avoiding an O(n^2) full
    cross-product for large datasets.
    """
    year = "UNK"
    if dob_col in row.index and pd.notna(row[dob_col]):
        year = str(row[dob_col].year)
    initial = "?"
    if first_name_col in row.index and pd.notna(row[first_name_col]):
        name = str(row[first_name_col]).strip()
        if name:
            initial = name[0].upper()
    return f"{year}_{initial}"


def build_blocks(df: pd.DataFrame, first_name_col: str, dob_col: str) -> dict:
    blocks: dict = {}
    for idx, row in df.iterrows():
        key = make_block_key(row, first_name_col, dob_col)
        blocks.setdefault(key, []).append(idx)
    return blocks


def generate_candidate_pairs(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
    first_name_col: str,
    dob_col: str,
) -> List[Tuple]:
    blocks_a = build_blocks(df_a, first_name_col, dob_col)

    if df_b is None:
        pairs = []
        for idxs in blocks_a.values():
            for i, j in itertools.combinations(idxs, 2):
                pairs.append((i, j))
        return pairs

    blocks_b = build_blocks(df_b, first_name_col, dob_col)
    pairs = []
    for key, idxs_a in blocks_a.items():
        idxs_b = blocks_b.get(key)
        if not idxs_b:
            continue
        for i in idxs_a:
            for j in idxs_b:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Pair comparison vectors
# ---------------------------------------------------------------------------

def compute_pair_features(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
    pairs: List[Tuple],
    fields: List[FieldSpec],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (similarity_matrix, gamma_matrix), each shape (n_pairs, n_fields).
    gamma is the binary agreement pattern used by the Fellegi-Sunter model.
    """
    frame_b = df_b if df_b is not None else df_a
    n, k = len(pairs), len(fields)
    sim_matrix = np.zeros((n, k))
    gamma_matrix = np.zeros((n, k))

    for row_idx, (i, j) in enumerate(pairs):
        ra = df_a.loc[i]
        rb = frame_b.loc[j]
        for col_idx, f in enumerate(fields):
            s = f.comparator(ra[f.column], rb[f.column])
            sim_matrix[row_idx, col_idx] = s
            gamma_matrix[row_idx, col_idx] = 1.0 if s >= f.threshold else 0.0

    return sim_matrix, gamma_matrix


# ---------------------------------------------------------------------------
# Fellegi-Sunter weights
# ---------------------------------------------------------------------------

def compute_weights(m: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fellegi-Sunter (1969) field weights:
        agreement weight    = log2(m_k / u_k)
        disagreement weight = log2((1-m_k) / (1-u_k))
    where m_k = P(field k agrees | true match), u_k = P(field k agrees | true non-match).
    """
    m = np.clip(m, 1e-4, 1 - 1e-4)
    u = np.clip(u, 1e-4, 1 - 1e-4)
    agree_w = np.log2(m / u)
    disagree_w = np.log2((1 - m) / (1 - u))
    return agree_w, disagree_w


def compute_scores(gamma_matrix: np.ndarray, agree_w: np.ndarray, disagree_w: np.ndarray) -> np.ndarray:
    """Composite Fellegi-Sunter match score per pair: sum of per-field weights
    given the pair's observed binary agreement pattern.
    """
    if gamma_matrix.shape[0] == 0:
        return np.array([])
    return gamma_matrix @ agree_w + (1 - gamma_matrix) @ disagree_w


def classify(scores: np.ndarray, upper: float, lower: float) -> np.ndarray:
    return np.where(scores >= upper, "match", np.where(scores >= lower, "possible_match", "non_match"))


# ---------------------------------------------------------------------------
# EM estimation of m/u (Winkler-style expectation-maximization)
# ---------------------------------------------------------------------------

def run_em(
    gamma_matrix: np.ndarray,
    m_init: np.ndarray,
    u_init: np.ndarray,
    p_init: float = 0.1,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate m (P(agree|match)) and u (P(agree|non-match)) directly from
    the observed agreement patterns, via the EM algorithm for the
    Fellegi-Sunter model (Winkler, 1988/1993). The latent variable is
    whether each candidate pair is a true match; E-step computes the
    posterior match probability for every pair, M-step re-estimates m, u,
    and the match prior p from those posteriors.
    """
    m = np.clip(np.array(m_init, dtype=float), 1e-4, 1 - 1e-4)
    u = np.clip(np.array(u_init, dtype=float), 1e-4, 1 - 1e-4)
    p = float(np.clip(p_init, 1e-6, 1 - 1e-6))

    n = gamma_matrix.shape[0]
    if n == 0:
        return m, u, p

    prev_ll = None
    for _ in range(max_iter):
        log_m1, log_m0 = np.log(m), np.log(1 - m)
        log_u1, log_u0 = np.log(u), np.log(1 - u)

        log_match = gamma_matrix @ log_m1 + (1 - gamma_matrix) @ log_m0 + np.log(p)
        log_nonmatch = gamma_matrix @ log_u1 + (1 - gamma_matrix) @ log_u0 + np.log(1 - p)

        max_log = np.maximum(log_match, log_nonmatch)
        denom = max_log + np.log(np.exp(log_match - max_log) + np.exp(log_nonmatch - max_log))
        g = np.exp(log_match - denom)  # posterior P(match | gamma) per pair

        ll = float(np.sum(denom))

        sum_g = max(float(np.sum(g)), 1e-10)
        sum_1mg = max(float(np.sum(1 - g)), 1e-10)

        m_new = np.clip((gamma_matrix * g[:, None]).sum(axis=0) / sum_g, 1e-4, 1 - 1e-4)
        u_new = np.clip((gamma_matrix * (1 - g)[:, None]).sum(axis=0) / sum_1mg, 1e-4, 1 - 1e-4)
        p_new = float(np.clip(sum_g / n, 1e-6, 1 - 1e-6))

        m, u, p = m_new, u_new, p_new

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return m, u, p


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def build_report(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
    pairs: List[Tuple],
    fields: List[FieldSpec],
    sim_matrix: np.ndarray,
    gamma_matrix: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    id_col: str,
) -> pd.DataFrame:
    frame_b = df_b if df_b is not None else df_a
    rows = []
    for row_idx, (i, j) in enumerate(pairs):
        ra = df_a.loc[i]
        rb = frame_b.loc[j]
        rec = {
            "record_id_a": ra[id_col] if id_col in df_a.columns else i,
            "record_id_b": rb[id_col] if id_col in frame_b.columns else j,
            "score": round(float(scores[row_idx]), 4),
            "classification": labels[row_idx],
        }
        for col_idx, f in enumerate(fields):
            rec[f"{f.key}_similarity"] = round(float(sim_matrix[row_idx, col_idx]), 4)
            rec[f"{f.key}_agree"] = bool(gamma_matrix[row_idx, col_idx])
        rows.append(rec)

    columns = ["record_id_a", "record_id_b", "score", "classification"]
    for f in fields:
        columns += [f"{f.key}_similarity", f"{f.key}_agree"]

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# End-to-end pipeline (importable for tests, used by the CLI below)
# ---------------------------------------------------------------------------

def run_linkage(
    df_a: pd.DataFrame,
    df_b: Optional[pd.DataFrame],
    fields: List[FieldSpec],
    first_name_col: str,
    dob_col: str,
    id_col: str,
    use_em: bool = False,
    em_max_iter: int = 50,
    upper_threshold: Optional[float] = None,
    lower_threshold: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Runs blocking, comparison, (optional) EM estimation, scoring, and
    classification. Returns (report_dataframe, info_dict) where info_dict
    carries the m/u/weights/thresholds actually used, for diagnostics.
    """
    pairs = generate_candidate_pairs(df_a, df_b, first_name_col, dob_col)
    sim_matrix, gamma_matrix = compute_pair_features(df_a, df_b, pairs, fields)

    default_m = np.array([f.default_m for f in fields])
    default_u = np.array([f.default_u for f in fields])

    if use_em and len(pairs) > 0:
        m, u, p = run_em(gamma_matrix, default_m, default_u, max_iter=em_max_iter)
    else:
        m, u, p = default_m, default_u, None

    agree_w, disagree_w = compute_weights(m, u)
    scores = compute_scores(gamma_matrix, agree_w, disagree_w)

    max_score = float(np.sum(np.maximum(agree_w, disagree_w)))
    min_score = float(np.sum(np.minimum(agree_w, disagree_w)))
    upper = upper_threshold if upper_threshold is not None else min_score + 0.7 * (max_score - min_score)
    lower = lower_threshold if lower_threshold is not None else min_score + 0.3 * (max_score - min_score)

    labels = classify(scores, upper, lower) if len(pairs) > 0 else np.array([])

    report = build_report(df_a, df_b, pairs, fields, sim_matrix, gamma_matrix, scores, labels, id_col)

    info = {
        "n_pairs": len(pairs),
        "m": m, "u": u, "p": p,
        "agree_weights": agree_w, "disagree_weights": disagree_w,
        "upper_threshold": upper, "lower_threshold": lower,
    }
    return report, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Record Linkage Deduplicator: Fellegi-Sunter probabilistic "
                     "record linkage with Jaro-Winkler fuzzy field comparison."
    )
    parser.add_argument("--input", nargs="+", required=True, metavar="CSV",
                         help="One CSV file to deduplicate, or two CSV files to link against each other.")
    parser.add_argument("--output", default="linkage_report.csv", help="Path to write the linkage report CSV.")
    parser.add_argument("--id-col", default="record_id", help="Column holding each record's unique id.")
    parser.add_argument("--first-name-col", default="first_name")
    parser.add_argument("--last-name-col", default="last_name")
    parser.add_argument("--dob-col", default="dob")
    parser.add_argument("--identifier-col", default="ssn",
                         help="Column holding an identifier field to compare (e.g. SSN, MRN).")
    parser.add_argument("--name-threshold", type=float, default=0.85,
                         help="Jaro-Winkler similarity >= this counts as agreement for name fields.")
    parser.add_argument("--dob-threshold", type=float, default=0.99,
                         help="DOB comparison score >= this counts as agreement (effectively exact match).")
    parser.add_argument("--id-threshold", type=float, default=0.90,
                         help="Identifier comparison score >= this counts as agreement.")
    parser.add_argument("--em", action="store_true",
                         help="Estimate m/u probabilities from the data via EM instead of using fixed defaults.")
    parser.add_argument("--em-max-iter", type=int, default=50)
    parser.add_argument("--upper-threshold", type=float, default=None,
                         help="Composite score >= this is classified 'match'. Auto-calibrated if omitted.")
    parser.add_argument("--lower-threshold", type=float, default=None,
                         help="Composite score >= this (and < upper) is 'possible_match'. Auto-calibrated if omitted.")

    args = parser.parse_args(argv)
    if len(args.input) not in (1, 2):
        parser.error("--input accepts either 1 CSV file (dedup) or 2 CSV files (link A vs B)")
    return args


def _load(path: str, dob_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if dob_col in df.columns:
        df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
    return df


def main(argv=None):
    args = parse_args(argv)

    df_a = _load(args.input[0], args.dob_col)
    df_b = _load(args.input[1], args.dob_col) if len(args.input) == 2 else None

    fields = build_field_specs(args, df_a, df_b)
    if len(fields) < 2:
        print("Error: at least 2 comparison fields must be present in the input data "
              "(first_name, last_name, dob, identifier).", file=sys.stderr)
        sys.exit(1)

    report, info = run_linkage(
        df_a, df_b, fields,
        first_name_col=args.first_name_col,
        dob_col=args.dob_col,
        id_col=args.id_col,
        use_em=args.em,
        em_max_iter=args.em_max_iter,
        upper_threshold=args.upper_threshold,
        lower_threshold=args.lower_threshold,
    )

    report.to_csv(args.output, index=False)

    counts = report["classification"].value_counts().to_dict() if len(report) else {}
    print(f"Candidate pairs after blocking: {info['n_pairs']}")
    if args.em:
        print("EM-estimated m/u probabilities:")
        for f, m_k, u_k in zip(fields, info["m"], info["u"]):
            print(f"  {f.key:<12} m={m_k:.4f}  u={u_k:.4f}")
    print(f"Thresholds: upper={info['upper_threshold']:.3f}  lower={info['lower_threshold']:.3f}")
    for label in ("match", "possible_match", "non_match"):
        print(f"  {label:<15} {counts.get(label, 0)}")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
