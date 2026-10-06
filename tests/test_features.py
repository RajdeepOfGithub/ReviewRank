import numpy as np
import pandas as pd
import pytest

from reviewrank.features import FEATURE_COLUMNS, FEATURE_GROUPS, build_features
from reviewrank.leakage import LeakageError, assert_no_forbidden_columns


def ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def row(i, date, rating=5.0, text="A fine review. It works!", parent="P1", title="Nice", images="[]", votes=0):
    return {
        "asin": f"A{i}",
        "parent_asin": parent,
        "user_id": f"U{i}",
        "rating": rating,
        "title": title,
        "text": text,
        "images_json": images,
        "timestamp_ms": ms(date),
        "verified_purchase": True,
        "helpful_vote": votes,
        "words": len(text.split()),
    }


def frame(rows):
    return pd.DataFrame(rows)


def test_feature_columns_have_no_label_derived_names():
    assert_no_forbidden_columns(FEATURE_COLUMNS, ("helpful", "vote"))
    assert sorted(FEATURE_COLUMNS) == sorted(c for g in FEATURE_GROUPS.values() for c in g)


def test_input_without_votes_works_and_votes_do_not_change_features():
    df = frame([row(1, "2018-01-01", votes=0), row(2, "2018-02-01", votes=0), row(3, "2018-03-01", votes=0)])
    base = build_features(df)
    assert "helpful_vote" not in base.columns
    altered = df.copy()
    altered["helpful_vote"] = [50, 7, 99]
    pd.testing.assert_frame_equal(base, build_features(altered))
    pd.testing.assert_frame_equal(base, build_features(df.drop(columns="helpful_vote")))


def test_prior_aggregates_use_only_earlier_reviews():
    df = frame([
        row(1, "2018-01-01", rating=5.0, text="one two"),
        row(2, "2018-01-02", rating=3.0, text="one two three four"),
        row(3, "2018-01-03", rating=1.0, text="x"),
    ])
    f = build_features(df)
    assert f["prior_reviews"].tolist() == [0, 1, 2]
    assert np.isnan(f["prior_mean_rating"].iloc[0])
    assert f["prior_mean_rating"].iloc[1] == pytest.approx(5.0)
    assert f["prior_mean_rating"].iloc[2] == pytest.approx(4.0)
    assert f["rating_minus_prior_mean"].iloc[2] == pytest.approx(1.0 - 4.0)
    assert f["prior_mean_words"].iloc[2] == pytest.approx((2 + 4) / 2)


def test_changing_a_later_review_never_changes_earlier_features():
    rows = [row(i, f"2018-01-0{i}", rating=float(i)) for i in range(1, 6)]
    base = build_features(frame(rows))
    rows[4] = row(5, "2018-01-05", rating=1.0, text="completely different later review text " * 5)
    changed = build_features(frame(rows))
    pd.testing.assert_frame_equal(base.iloc[:4], changed.iloc[:4])


def test_same_timestamp_reviews_do_not_see_each_other():
    df = frame([row(1, "2018-01-01"), row(2, "2018-01-01"), row(3, "2018-01-02")])
    f = build_features(df)
    assert f["prior_reviews"].tolist() == [0, 0, 2]


def test_products_are_independent_and_row_order_does_not_matter():
    df = frame([
        row(1, "2018-01-01", parent="P1", rating=5.0),
        row(2, "2018-01-02", parent="P2", rating=1.0),
        row(3, "2018-01-03", parent="P1", rating=3.0),
        row(4, "2018-01-04", parent="P2", rating=2.0),
    ])
    f = build_features(df)
    assert f["prior_reviews"].tolist() == [0, 0, 1, 1]
    assert f["prior_mean_rating"].iloc[2] == pytest.approx(5.0)
    assert f["prior_mean_rating"].iloc[3] == pytest.approx(1.0)
    shuffled = df.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(shuffled).sort_index(), f)


def test_text_and_media_features():
    images = '[{"large_image_url": "a"}, {"large_image_url": "b"}]'
    df = frame([row(1, "2018-06-01", text="Great!! Really? 5 stars. Wow", images=images, title="Top pick here")])
    f = build_features(df).iloc[0]
    assert f["exclaims"] == 2 and f["questions"] == 1 and f["digits"] == 1
    assert f["sentences"] == 3  # runs of [.!?]: "!!", "?", "."
    assert f["image_count"] == 2 and f["has_image"] == 1
    assert f["title_words"] == 3
    assert f["verified"] == 1.0
    assert 0 < f["unique_word_ratio"] <= 1


def test_calendar_uses_utc_month_and_hour_only():
    df = frame([row(1, "2019-07-15")])
    f = build_features(df)
    assert f["month"].iloc[0] == 7 and f["hour"].iloc[0] == 0
    assert "year" not in f.columns and "timestamp_ms" not in f.columns


def test_empty_text_row_is_handled():
    f = build_features(frame([row(1, "2018-01-01", text="")])).iloc[0]
    assert f["is_empty_text"] == 1.0 and f["words"] == 0
    assert np.isnan(f["avg_word_len"])


def test_missing_required_column_is_reported():
    with pytest.raises(KeyError):
        build_features(frame([row(1, "2018-01-01")]).drop(columns="rating"))


def test_forbidden_name_tripwire_still_fires():
    with pytest.raises(LeakageError):
        assert_no_forbidden_columns(FEATURE_COLUMNS + ["helpful_ratio"], ("helpful", "vote"))
