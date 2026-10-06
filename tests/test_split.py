from pathlib import Path

import pandas as pd
import pytest

from reviewrank.split import assign_split, load_split_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "splits.yaml"


def ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


@pytest.fixture
def cfg():
    return load_split_config(CONFIG)


def test_config_loads_and_is_marked_provisional(cfg):
    assert cfg["status"] == "provisional"


def test_boundary_belongs_to_later_split(cfg):
    s = pd.Series([ms("2020-06-30"), ms("2020-07-01"), ms("2020-12-31"), ms("2021-01-01"), ms("2021-12-31"), ms("2022-01-01")])
    out = assign_split(s, cfg).tolist()
    assert out == ["train", "validation", "validation", "test", "test", "stream"]


def test_split_is_deterministic(cfg):
    s = pd.Series([ms("2015-03-05"), ms("2021-06-01"), ms("2023-02-02")])
    assert assign_split(s, cfg).tolist() == assign_split(s, cfg).tolist()


def test_splits_are_disjoint_in_time(cfg):
    dates = pd.date_range("2016-01-01", "2023-06-30", freq="7D", tz="UTC")
    s = pd.Series([int(d.timestamp() * 1000) for d in dates])
    labels = assign_split(s, cfg)
    ts = pd.to_datetime(s, unit="ms", utc=True)
    latest = {name: ts[labels == name].max() for name in ["train", "validation", "test"]}
    earliest = {name: ts[labels == name].min() for name in ["validation", "test", "stream"]}
    assert latest["train"] < earliest["validation"]
    assert latest["validation"] < earliest["test"]
    assert latest["test"] < earliest["stream"]


def test_non_increasing_boundaries_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text('train_end: "2021-01-01"\nvalidation_end: "2020-01-01"\ntest_end: "2022-01-01"\n')
    with pytest.raises(ValueError):
        load_split_config(bad)
