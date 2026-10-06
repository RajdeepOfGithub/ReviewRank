from pathlib import Path

import pandas as pd
import pytest
import yaml

from reviewrank.leakage import LeakageError, assert_as_of, assert_no_forbidden_columns

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def forbidden():
    cfg = yaml.safe_load((ROOT / "configs" / "project.yaml").read_text(encoding="utf-8"))
    return cfg["forbidden_feature_substrings"]


def test_clean_feature_set_passes(forbidden):
    assert_no_forbidden_columns(["word_count", "rating", "verified_purchase", "image_count"], forbidden)


@pytest.mark.parametrize("bad", ["helpful_vote", "log_helpful_vote", "votes_rank", "HelpfulRatio"])
def test_label_derived_column_fails(bad, forbidden):
    with pytest.raises(LeakageError):
        assert_no_forbidden_columns(["word_count", bad], forbidden)


def test_as_of_passes_when_sources_are_strictly_earlier():
    row = pd.Series(pd.to_datetime(["2020-01-10", "2020-02-10"]))
    src = pd.Series(pd.to_datetime(["2020-01-09", "2020-02-09"]))
    assert_as_of(src, row)


def test_as_of_fails_when_source_is_same_time_or_later():
    row = pd.Series(pd.to_datetime(["2020-01-10", "2020-02-10"]))
    src = pd.Series(pd.to_datetime(["2020-01-10", "2020-03-01"]))
    with pytest.raises(LeakageError):
        assert_as_of(src, row)
