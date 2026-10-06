"""Temporal split by review posting time.

Boundaries come from configs/splits.yaml and are UTC, half-open [start, end).
A review posted exactly at a boundary belongs to the LATER split.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

SPLITS = ("train", "validation", "test", "stream")


def load_split_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    bounds = boundaries(cfg)
    if not (bounds[0] < bounds[1] < bounds[2]):
        raise ValueError("split boundaries must be strictly increasing")
    return cfg


def boundaries(cfg: dict) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(cfg["train_end"], tz="UTC"),
        pd.Timestamp(cfg["validation_end"], tz="UTC"),
        pd.Timestamp(cfg["test_end"], tz="UTC"),
    ]


def assign_split(timestamp_ms: pd.Series, cfg: dict) -> pd.Series:
    """Label each review train/validation/test/stream from its posting time (ms since epoch).

    Reviews before an explicit train_start (if set) get the label "excluded".
    """
    ts = pd.to_datetime(timestamp_ms, unit="ms", utc=True)
    train_end, validation_end, test_end = boundaries(cfg)

    out = pd.Series("stream", index=timestamp_ms.index, dtype="object")
    out[ts < test_end] = "test"
    out[ts < validation_end] = "validation"
    out[ts < train_end] = "train"
    if cfg.get("train_start"):
        out[ts < pd.Timestamp(cfg["train_start"], tz="UTC")] = "excluded"
    return out
