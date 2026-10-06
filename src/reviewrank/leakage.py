"""Leakage tripwires. These catch obvious mistakes; they do not prove absence of leakage."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


class LeakageError(ValueError):
    """Raised when a feature set or feature computation could leak the label or the future."""


def assert_no_forbidden_columns(
    columns: Iterable[str], forbidden_substrings: Sequence[str]
) -> None:
    """Fail if any feature column name contains a forbidden substring (case-insensitive)."""
    bad = [
        c
        for c in columns
        if any(s.lower() in str(c).lower() for s in forbidden_substrings)
    ]
    if bad:
        raise LeakageError(f"forbidden feature columns (label-derived?): {sorted(bad)}")


def assert_as_of(source_timestamp: pd.Series, row_timestamp: pd.Series) -> None:
    """Fail unless every source used for a feature was posted strictly BEFORE the row.

    Example: each review's product-mean rating must be built only from reviews
    with an earlier timestamp. Pass the latest source timestamp used per row.
    """
    if len(source_timestamp) != len(row_timestamp):
        raise ValueError("series must be aligned")
    bad = (source_timestamp.reset_index(drop=True) >= row_timestamp.reset_index(drop=True)).sum()
    if bad:
        raise LeakageError(f"{int(bad)} feature value(s) use data at or after the row's own posting time")
