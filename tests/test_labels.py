from pathlib import Path

import pandas as pd
import pytest

from reviewrank.labels import (
    build_groups,
    maturity_cutoff_ms,
    vote_grade,
    window_label,
)
from reviewrank.split import boundaries, load_split_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"


def ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def review(i, parent="P1", date="2020-08-01", votes=0, seen=False):
    return {
        "asin": f"A{i}",
        "parent_asin": parent,
        "timestamp_ms": ms(date),
        "helpful_vote": votes,
        "text_seen_in_train": seen,
    }


def frame(rows):
    return pd.DataFrame(rows)


def test_vote_grade_bins_boundaries():
    s = pd.Series([0, 1, 2, 4, 5, 100])
    assert vote_grade(s).tolist() == [0, 1, 2, 2, 3, 3]


def test_vote_grade_binary():
    assert vote_grade(pd.Series([0, 1, 7]), mode="binary").tolist() == [0, 1, 1]


def test_vote_grade_unknown_mode():
    with pytest.raises(ValueError):
        vote_grade(pd.Series([1]), mode="nope")


def test_window_labels_at_boundaries():
    s = pd.Series([ms("2020-06-30"), ms("2020-07-01"), ms("2020-12-31"), ms("2021-01-01")])
    assert window_label(s, "half_year").tolist() == ["2020H1", "2020H2", "2020H2", "2021H1"]
    q = pd.Series([ms("2020-03-31"), ms("2020-04-01"), ms("2020-09-30"), ms("2020-10-01")])
    assert window_label(q, "quarter").tolist() == ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]


def test_split_boundaries_align_with_half_year_starts():
    cfg = load_split_config(CONFIG)
    for b in boundaries(cfg):
        assert b.day == 1 and b.month in (1, 7)


def test_maturity_cutoff_is_18_months_before_snapshot():
    assert maturity_cutoff_ms() == ms("2022-03-11")


def test_group_with_five_reviews_and_a_vote_is_eligible():
    rows = [review(i, votes=(2 if i == 0 else 0)) for i in range(5)]
    out, stats = build_groups(frame(rows))
    assert len(out) == 5
    assert stats["groups_eligible"] == 1
    assert set(out["group_id"]) == {"P1|2020H2"}
    assert out.set_index("asin").loc["A0", "grade"] == 2


def test_group_below_minimum_size_is_excluded():
    rows = [review(i, votes=3) for i in range(4)]
    out, stats = build_groups(frame(rows))
    assert out.empty and stats["groups_size_ok"] == 0


def test_group_without_any_voted_review_is_excluded_but_counted_as_size_ok():
    rows = [review(i, votes=0) for i in range(5)]
    out, stats = build_groups(frame(rows))
    assert out.empty
    assert stats["groups_size_ok"] == 1 and stats["groups_eligible"] == 0
    assert stats["group_coverage"] == 0.0


def test_text_overlap_rows_are_removed_before_grouping():
    rows = [review(i, votes=1) for i in range(5)]
    rows[0]["text_seen_in_train"] = True  # leaves only 4 reviews
    out, stats = build_groups(frame(rows))
    assert out.empty
    assert stats["rows_after_text_overlap_exclusion"] == 4


def test_immature_reviews_are_not_labeled():
    rows = [review(i, date="2022-03-12", votes=1) for i in range(5)]
    out, stats = build_groups(frame(rows))
    assert out.empty and stats["rows_after_maturity_rule"] == 0


def test_different_products_and_windows_form_separate_groups():
    rows = (
        [review(i, parent="P1", date="2020-08-01", votes=1) for i in range(5)]
        + [review(10 + i, parent="P2", date="2020-08-01", votes=1) for i in range(5)]
        + [review(20 + i, parent="P1", date="2020-12-31", votes=1) for i in range(2)]
    )
    out, stats = build_groups(frame(rows))
    assert set(out["group_id"]) == {"P1|2020H2", "P2|2020H2"}
    assert len(out) == 12  # P1 H2 has 7 reviews (5 + 2 in the same half-year)


def test_require_two_voted_reviews_sensitivity():
    rows = [review(i, votes=(1 if i == 0 else 0)) for i in range(5)]
    assert len(build_groups(frame(rows), min_voted=1)[0]) == 5
    assert build_groups(frame(rows), min_voted=2)[0].empty
