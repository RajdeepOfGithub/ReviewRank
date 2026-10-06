"""Stage 2 cleaning rules. Every exclusion is counted by reason.

Rules (docs/planning/03_METRICS_AND_DECISIONS.md, ADR-008..ADR-010):
- ADR-008  Exact duplicates on (user_id, parent_asin, timestamp_ms) are removed (keep one).
           Measured: in development rows all such duplicates had identical text and votes.
- ADR-009  A validation/test review whose normalized text (>= 20 chars) also appears in the
           training period is FLAGGED `text_seen_in_train`; evaluation excludes flagged rows.
           Only text and time are used, never votes.
- ADR-010  Reviews before `train_start` get split == "excluded" (not used for training).

Run:  python -m reviewrank.clean
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .acquire import sha256_file, write_manifest
from .split import assign_split, load_split_config

DEDUP_KEY = ["user_id", "parent_asin", "timestamp_ms"]
MIN_CHARS_FOR_TEXT_OVERLAP = 20


def normalize_text(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean_reviews(df: pd.DataFrame, split_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned frame, counts). Does not read or use `helpful_vote`."""
    counts: dict = {"rows_in": int(len(df))}

    df = df.sort_values(DEDUP_KEY + ["asin"], kind="mergesort")
    dup = df.duplicated(subset=DEDUP_KEY, keep="first")
    counts["exact_duplicates_removed"] = int(dup.sum())
    df = df.loc[~dup].copy()

    df["split"] = assign_split(df["timestamp_ms"], split_cfg)
    df["words"] = df["text"].str.split().str.len().fillna(0).astype(int)

    nt = normalize_text(df["text"])
    long_enough = nt.str.len() >= MIN_CHARS_FOR_TEXT_OVERLAP
    train_texts = set(nt[(df["split"] == "train") & long_enough])
    df["text_seen_in_train"] = (
        df["split"].isin(["validation", "test"]) & long_enough & nt.isin(train_texts)
    )

    counts["rows_out"] = int(len(df))
    counts["rows_per_split"] = {k: int(v) for k, v in df["split"].value_counts().sort_index().items()}
    counts["text_seen_in_train_per_split"] = {
        k: int(v)
        for k, v in df.loc[df["text_seen_in_train"], "split"].value_counts().sort_index().items()
    }
    return df.reset_index(drop=True), counts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_split_config(root / "configs" / "splits.yaml")
    src = root / "data" / "processed" / "reviews.parquet"
    out = root / "data" / "processed" / "reviews_clean.parquet"

    df = pd.read_parquet(src)
    cleaned, counts = clean_reviews(df, cfg)
    cleaned.to_parquet(out, index=False, compression="zstd")

    manifest = {
        "input": "data/processed/reviews.parquet",
        "input_sha256": sha256_file(src),
        "output": "data/processed/reviews_clean.parquet",
        "output_sha256": sha256_file(out),
        "split_config": {k: cfg[k] for k in ("status", "train_start", "train_end", "validation_end", "test_end")},
        "counts": counts,
    }
    write_manifest(manifest, root / "artifacts" / "manifests" / "clean_manifest.json")
    print(manifest["counts"])


if __name__ == "__main__":
    main()
