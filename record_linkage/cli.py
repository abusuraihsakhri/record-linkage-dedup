"""Command-line interface for the Record Linkage Deduplicator."""

import argparse
import sys

from .linker import run_linkage, DEFAULT_BLOCKING_SPEC


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="record-linkage-dedup",
        description=(
            "Identify likely duplicate patient records across one or more "
            "CSV datasets using the Fellegi-Sunter probabilistic record "
            "linkage model with Jaro-Winkler fuzzy field comparison."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="One or more input CSV files. A single file deduplicates "
             "within itself; multiple files are linked against each other.",
    )
    parser.add_argument(
        "--output", default="linkage_report.csv",
        help="Path to write the linkage report CSV (default: linkage_report.csv)",
    )
    parser.add_argument(
        "--fields", default=None,
        help="Comma-separated 'column:method' list, methods are jw|near|exact, "
             "e.g. 'first_name:jw,last_name:jw,dob:near,patient_id:near'. "
             "If omitted, common column names are auto-detected.",
    )
    parser.add_argument(
        "--blocking", default=DEFAULT_BLOCKING_SPEC,
        help="Comma-separated 'column:transform' list, transforms are "
             "year|initial|exact (default: 'dob:year,first_name:initial')",
    )
    parser.add_argument(
        "--em", action="store_true",
        help="Estimate m/u agreement probabilities from the data with EM "
             "instead of using the fixed defaults.",
    )
    parser.add_argument("--em-iterations", type=int, default=20)
    parser.add_argument("--prior", type=float, default=0.05,
                         help="Initial/assumed prior probability that a candidate pair is a true match.")
    parser.add_argument("--m-default", type=float, default=0.9,
                         help="Default m-probability (agreement | true match) before EM.")
    parser.add_argument("--u-default", type=float, default=0.1,
                         help="Default u-probability (agreement | true non-match) before EM.")
    parser.add_argument("--jw-threshold", type=float, default=0.85,
                         help="Jaro-Winkler similarity cutoff for 'jw' fields to count as agreeing.")
    parser.add_argument("--near-threshold", type=float, default=0.90,
                         help="Jaro-Winkler similarity cutoff for 'near' fields (DOB/IDs) to count as agreeing.")
    parser.add_argument("--upper-threshold", type=float, default=None,
                         help="Composite score at/above which a pair is classified 'match'. "
                              "Default: 75% of the maximum possible score.")
    parser.add_argument("--lower-threshold", type=float, default=None,
                         help="Composite score at/above which a pair is classified 'possible_match'. "
                              "Default: 25% of the maximum possible score.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_linkage(
            paths=args.inputs,
            field_specs_raw=args.fields,
            blocking_spec_raw=args.blocking,
            use_em=args.em,
            em_iterations=args.em_iterations,
            prior=args.prior,
            m_default=args.m_default,
            u_default=args.u_default,
            jw_threshold=args.jw_threshold,
            near_threshold=args.near_threshold,
            upper_threshold=args.upper_threshold,
            lower_threshold=args.lower_threshold,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = result["report"]
    report.to_csv(args.output, index=False)

    n_match = int((report["classification"] == "match").sum()) if not report.empty else 0
    n_possible = int((report["classification"] == "possible_match").sum()) if not report.empty else 0
    n_non = int((report["classification"] == "non_match").sum()) if not report.empty else 0

    print(f"Records loaded:        {result['n_records']}")
    print(f"Candidate pairs:       {result['n_candidate_pairs']} (after blocking)")
    print(f"Fields compared:       {', '.join(fs.name for fs in result['field_specs'])}")
    if result["em_info"] is not None:
        print(f"EM iterations run:     {result['em_info']['iterations']}")
        print(f"Fitted prior P(match): {result['model'].prior:.4f}")
        for name, m, u in zip((fs.name for fs in result["field_specs"]),
                               result["model"].m_prob, result["model"].u_prob):
            print(f"  {name}: m={m:.4f} u={u:.4f}")
    print(f"Score thresholds:      match >= {result['upper_threshold']:.3f} "
          f"> possible_match >= {result['lower_threshold']:.3f} > non_match")
    print(f"Classification counts: match={n_match} possible_match={n_possible} non_match={n_non}")
    print(f"Report written to:     {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
