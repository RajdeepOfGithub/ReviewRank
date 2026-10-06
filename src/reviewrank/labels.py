"""Relevance labels and query groups (ADR-011..013, configs/eval.yaml).

- Grades from absolute vote bins: 0 votes -> 0, 1 -> 1, 2-4 -> 2, 5+ -> 3.
- Query group = (parent_asin, calendar window). Windows are half-year (primary) or quarter
  (sensitivity). Split boundaries align with half-year starts, so a group never crosses splits.
- A group is eligible for NDCG when it has at least `min_size` reviews and at least
  `min_voted` reviews with >= 1 vote (otherwise NDCG is undefined or uninformative).
- Reviews flagged `text_seen_in_train` (ADR-009) and reviews younger than the maturity limit
  (ADR-013) are removed before groups are formed.

`helpful_vote` is read here ONLY to build labels. It must never reach a feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SNAPSHOT_DATE = "2023-09-11"
MATURITY_MONTHS = 18
WINDOWS = ("half_year", "quarter")


def vote_grade(votes: pd.Series, mode: str = "bins") -> pd.Series:
    v = votes.to_numpy()
    if mode == "bins":
        grade = np.select([v >= 5, v >= 2, v >= 1], [3, 2, 1], default=0)
    elif mode == "binary":
        grade = (v >= 1).astype(int)
    else:
        raise ValueError(f"unknown grade mode: {mode}")
    return pd.Series(grade, index=votes.index)


def window_label(timestamp_ms: pd.Series, window: str = "half_year") -> pd.Series:
    ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
    year = ts.dt.year.astype(str)
    if window == "half_year":
        return year + "H" + (((ts.dt.month - 1) // 6) + 1).astype(str)
    if window == "quarter":
        return year + "Q" + (((ts.dt.month - 1) // 3) + 1).astype(str)
    raise ValueError(f"unknown window: {window}")


def maturity_cutoff_ms(snapshot: str = SNAPSHOT_DATE, months: int = MATURITY_MONTHS) -> int:
    cutoff = pd.Timestamp(snapshot, tz="UTC") - pd.DateOffset(months=months)
    return int(cutoff.timestamp() * 1000)


def build_groups(
    df: pd.DataFrame,
    *,
    window: str = "half_year",
    min_size: int = 5,
    min_voted: int = 1,
    grade_mode: str = "bins",
    snapshot: str = SNAPSHOT_DATE,
    maturity_months: int = MATURITY_MONTHS,
) -> tuple[pd.DataFrame, dict]:
    """Return (rows in eligible groups with group_id/grade/group_size/group_voted, coverage stats)."""
    stats: dict = {"rows_in": int(len(df))}

    d = df
    if "text_seen_in_train" in d.columns:
        d = d.loc[~d["text_seen_in_train"]]
    stats["rows_after_text_overlap_exclusion"] = int(len(d))

    d = d.loc[d["timestamp_ms"] <= maturity_cutoff_ms(snapshot, maturity_months)]
    stats["rows_after_maturity_rule"] = int(len(d))

    d = d.assign(
        group_id=d["parent_asin"] + "|" + window_label(d["timestamp_ms"], window),
        grade=vote_grade(d["helpful_vote"], grade_mode),
        _voted=(d["helpful_vote"] >= 1).astype(int),
    )
    grouped = d.groupby("group_id")
    d["group_size"] = grouped["grade"].transform("size")
    d["group_voted"] = grouped["_voted"].transform("sum")
    d = d.drop(columns="_voted")

    size_ok = d["group_size"] >= min_size
    eligible = size_ok & (d["group_voted"] >= min_voted)
    stats["groups_size_ok"] = int(d.loc[size_ok, "group_id"].nunique())
    stats["groups_eligible"] = int(d.loc[eligible, "group_id"].nunique())
    stats["reviews_in_size_ok_groups"] = int(size_ok.sum())
    stats["reviews_in_eligible_groups"] = int(eligible.sum())
    stats["group_coverage"] = (
        stats["groups_eligible"] / stats["groups_size_ok"] if stats["groups_size_ok"] else float("nan")
    )
    return d.loc[eligible].copy(), stats
