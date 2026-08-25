"""Blocking key construction.

Comparing every record against every other record is O(n^2) and does not
scale. Blocking groups records by a cheap-to-compute key (e.g. birth year
plus first-name initial) and only compares records that share a key,
which is the standard way to make Fellegi-Sunter linkage tractable on
real datasets.
"""

from typing import List, Tuple
import pandas as pd


def _year_of(value: str) -> str:
    if not value:
        return ""
    # normalized dob strings are "YYYY-MM-DD"
    return value[:4]


def _initial_of(value: str) -> str:
    if not value:
        return ""
    return value[0]


def _exact_of(value: str) -> str:
    return value


TRANSFORMS = {
    "year": _year_of,
    "initial": _initial_of,
    "exact": _exact_of,
}


def parse_blocking_spec(spec: str, column_lookup: dict) -> List[Tuple[str, str]]:
    """Parse "col:transform,col2:transform2" into [(actual_col, transform), ...].

    ``column_lookup`` maps lowercased column name -> actual column name in
    the working dataframe, so the spec can be written case-insensitively.
    """
    fields = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid blocking field '{part}', expected 'column:transform'")
        col, transform = part.split(":", 1)
        col = col.strip().lower()
        transform = transform.strip().lower()
        if col not in column_lookup:
            raise ValueError(
                f"Blocking column '{col}' not found in input data. "
                f"Available columns: {sorted(column_lookup.keys())}"
            )
        if transform not in TRANSFORMS:
            raise ValueError(
                f"Unknown blocking transform '{transform}'. Choices: {sorted(TRANSFORMS.keys())}"
            )
        fields.append((column_lookup[col], transform))
    if not fields:
        raise ValueError("Blocking spec produced no usable fields")
    return fields


def build_blocking_keys(df: pd.DataFrame, blocking_fields: List[Tuple[str, str]]) -> pd.Series:
    """Compute one blocking-key tuple per row of ``df``.

    ``blocking_fields`` is a list of (column, transform_name) pairs, where
    column already holds normalized string values (see linker.normalize_*).
    """
    keys = []
    for _, row in df.iterrows():
        parts = []
        for col, transform in blocking_fields:
            fn = TRANSFORMS[transform]
            parts.append(fn(row[col]))
        keys.append(tuple(parts))
    return pd.Series(keys, index=df.index)
