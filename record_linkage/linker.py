"""Pipeline orchestration: normalize fields, block, compare candidate pairs,
score them with a Fellegi-Sunter model, and classify the result.
"""

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .blocking import build_blocking_keys, parse_blocking_spec
from .model import FieldSpec, FellegiSunterModel
from .similarity import jaro_winkler

DEFAULT_FIELD_CANDIDATES = {
    "first_name": "jw",
    "given_name": "jw",
    "last_name": "jw",
    "surname": "jw",
    "family_name": "jw",
    "dob": "near",
    "date_of_birth": "near",
    "birth_date": "near",
}

DEFAULT_BLOCKING_SPEC = "dob:year,first_name:initial"


def normalize_string(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_dob(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.strftime("%Y-%m-%d")


def _column_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for col in df.columns:
        lookup[col.strip().lower()] = col
    return lookup


def parse_field_spec(spec: str, column_lookup: dict, jw_threshold: float, near_threshold: float) -> List[FieldSpec]:
    fields = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid field '{part}', expected 'column:method'")
        col, method = part.split(":", 1)
        col = col.strip().lower()
        method = method.strip().lower()
        if col not in column_lookup:
            raise ValueError(
                f"Field column '{col}' not found in input data. "
                f"Available columns: {sorted(column_lookup.keys())}"
            )
        if method not in ("jw", "near", "exact"):
            raise ValueError(f"Unknown comparison method '{method}' for field '{col}'")
        threshold = jw_threshold if method == "jw" else (near_threshold if method == "near" else 1.0)
        fields.append(FieldSpec(name=col, column=column_lookup[col], method=method, threshold=threshold))
    if not fields:
        raise ValueError("Field spec produced no usable fields")
    return fields


def infer_field_specs(column_lookup: dict, jw_threshold: float, near_threshold: float) -> List[FieldSpec]:
    found = []
    for candidate, method in DEFAULT_FIELD_CANDIDATES.items():
        if candidate in column_lookup:
            threshold = jw_threshold if method == "jw" else near_threshold
            found.append(FieldSpec(name=candidate, column=column_lookup[candidate], method=method, threshold=threshold))
    if len(found) < 2:
        raise ValueError(
            "Could not infer at least two comparison fields from column names "
            f"{sorted(column_lookup.keys())}. Pass --fields explicitly, "
            "e.g. --fields first_name:jw,last_name:jw,dob:near"
        )
    return found


def load_and_normalize(paths: List[str], field_specs_raw: Optional[str],
                        jw_threshold: float, near_threshold: float) -> Tuple[pd.DataFrame, List[FieldSpec], dict]:
    """Read all input CSVs, tag each row with its source file, and return the
    combined dataframe alongside the resolved field specs and column lookup.
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path, dtype=str)
        df["_source"] = path
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["_row_id"] = combined.index

    lookup = _column_lookup(combined)

    if field_specs_raw:
        field_specs = parse_field_spec(field_specs_raw, lookup, jw_threshold, near_threshold)
    else:
        field_specs = infer_field_specs(lookup, jw_threshold, near_threshold)

    # Normalize each compared column in place, using dob-aware parsing for
    # anything that looks like a date column.
    for fs in field_specs:
        if "dob" in fs.name or "date" in fs.name or "birth" in fs.name:
            combined[fs.column] = combined[fs.column].apply(normalize_dob)
        else:
            combined[fs.column] = combined[fs.column].apply(normalize_string)

    return combined, field_specs, lookup


def generate_candidate_pairs(df: pd.DataFrame, blocking_fields: List[Tuple[str, str]]) -> List[Tuple[int, int]]:
    """Return (row_a, row_b) index pairs restricted to records sharing a
    blocking key, avoiding the full O(n^2) comparison space.
    """
    keys = build_blocking_keys(df, blocking_fields)
    pairs = []
    for key, group in df.groupby(keys).groups.items():
        # Missing blocking components should not form a giant all-missing block.
        if any(part == "" for part in key):
            continue
        indices = list(group)
        if len(indices) < 2:
            continue
        pairs.extend(combinations(indices, 2))
    return pairs


def build_agreement_matrix(df: pd.DataFrame, pairs: List[Tuple[int, int]],
                            field_specs: List[FieldSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the binary agreement matrix and the parallel similarity-score
    matrix for every candidate pair, one column per field.
    """
    n_pairs = len(pairs)
    n_fields = len(field_specs)
    agreement = np.zeros((n_pairs, n_fields), dtype=float)
    similarity = np.zeros((n_pairs, n_fields), dtype=float)

    columns = {fs.column: df[fs.column].to_numpy() for fs in field_specs}

    for row, (a, b) in enumerate(pairs):
        for col_idx, fs in enumerate(field_specs):
            va = columns[fs.column][a]
            vb = columns[fs.column][b]
            if va == "" or vb == "":
                sim = 0.0
                agree = False
            elif fs.method == "exact":
                sim = 1.0 if va == vb else 0.0
                agree = va == vb
            else:  # 'jw' or 'near' both use Jaro-Winkler, different thresholds
                sim = 1.0 if va == vb else jaro_winkler(va, vb)
                agree = sim >= fs.threshold
            similarity[row, col_idx] = sim
            agreement[row, col_idx] = 1.0 if agree else 0.0

    return agreement, similarity


def run_linkage(paths: List[str], field_specs_raw: Optional[str] = None,
                 blocking_spec_raw: str = DEFAULT_BLOCKING_SPEC,
                 use_em: bool = False, em_iterations: int = 20, prior: float = 0.05,
                 m_default: float = 0.9, u_default: float = 0.1,
                 jw_threshold: float = 0.85, near_threshold: float = 0.90,
                 upper_threshold: Optional[float] = None,
                 lower_threshold: Optional[float] = None) -> dict:
    """Run the full pipeline and return a dict with the report dataframe and
    metadata (resolved thresholds, fitted m/u, EM diagnostics).
    """
    df, field_specs, lookup = load_and_normalize(paths, field_specs_raw, jw_threshold, near_threshold)
    blocking_fields = parse_blocking_spec(blocking_spec_raw, lookup)
    # A blocking column need not also be a comparison field.
    compared_columns = {fs.column for fs in field_specs}
    for column, transform in blocking_fields:
        if column not in compared_columns:
            normalizer = normalize_dob if transform == "year" else normalize_string
            df[column] = df[column].apply(normalizer)

    pairs = generate_candidate_pairs(df, blocking_fields)

    field_names = [fs.name for fs in field_specs]
    model = FellegiSunterModel.with_default_probabilities(field_names, m_default, u_default, prior)

    em_info = None
    if pairs:
        agreement, similarity = build_agreement_matrix(df, pairs, field_specs)
        if use_em:
            em_info = model.fit_em(agreement, max_iter=em_iterations)
        scores = model.score_pairs(agreement)
    else:
        agreement = np.zeros((0, len(field_specs)))
        similarity = np.zeros((0, len(field_specs)))
        scores = np.zeros(0)

    max_score = model.max_possible_score()
    min_score = model.min_possible_score()
    resolved_upper = upper_threshold if upper_threshold is not None else 0.75 * max_score
    resolved_lower = lower_threshold if lower_threshold is not None else 0.25 * max_score
    if resolved_lower < min_score:
        resolved_lower = min_score

    if not np.isfinite(resolved_upper) or not np.isfinite(resolved_lower) or resolved_lower > resolved_upper:
        raise ValueError("Score thresholds must be finite and lower <= upper")

    labels = FellegiSunterModel.classify(scores, resolved_upper, resolved_lower) if len(pairs) else []

    report_rows = []
    for i, (a, b) in enumerate(pairs):
        row = {
            "row_a": int(a),
            "row_b": int(b),
            "source_a": df.at[a, "_source"],
            "source_b": df.at[b, "_source"],
            "score": float(scores[i]),
            "classification": labels[i],
        }
        for col_idx, fs in enumerate(field_specs):
            row[f"{fs.name}_a"] = df.at[a, fs.column]
            row[f"{fs.name}_b"] = df.at[b, fs.column]
            row[f"{fs.name}_similarity"] = round(float(similarity[i, col_idx]), 4)
            row[f"{fs.name}_agree"] = bool(agreement[i, col_idx])
        report_rows.append(row)

    report_columns = ["row_a", "row_b", "source_a", "source_b", "score", "classification"]
    for fs in field_specs:
        report_columns.extend([f"{fs.name}_a", f"{fs.name}_b", f"{fs.name}_similarity", f"{fs.name}_agree"])
    report_df = pd.DataFrame(report_rows, columns=report_columns)
    if not report_df.empty:
        report_df = report_df.sort_values("score", ascending=False).reset_index(drop=True)

    return {
        "report": report_df,
        "field_specs": field_specs,
        "model": model,
        "em_info": em_info,
        "upper_threshold": resolved_upper,
        "lower_threshold": resolved_lower,
        "max_possible_score": max_score,
        "min_possible_score": min_score,
        "n_records": len(df),
        "n_candidate_pairs": len(pairs),
    }
