"""Stage 4: build as-of features for excluded + train + validation rows (sealed test NOT loaded).

Run from the repo root:  python scripts/run_stage4_features.py
Outputs: data/processed/features_dev.parquet, reports/features/feature_report.md and .json
The excluded (pre-2015) rows are loaded only as HISTORY for the as-of product aggregates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file  # noqa: E402
from reviewrank.features import FEATURE_COLUMNS, FEATURE_GROUPS, build_features  # noqa: E402

CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
OUT_DATA = ROOT / "data" / "processed" / "features_dev.parquet"
OUT_REPORT = ROOT / "reports" / "features"
RUN_ID = "STAGE4-20260930"

COLUMNS = [
    "asin", "parent_asin", "user_id", "rating", "title", "text", "images_json", "timestamp_ms",
    "helpful_vote", "verified_purchase", "split", "words",
]
df = pd.read_parquet(CLEAN, columns=COLUMNS, filters=[("split", "in", ["excluded", "train", "validation"])])
assert not (df["split"] == "test").any() and not (df["split"] == "stream").any(), "test/stream rows must not be loaded"
print("rows loaded:", df["split"].value_counts().to_dict())

feats = build_features(df)  # never reads helpful_vote
out = pd.concat([df[["user_id", "parent_asin", "timestamp_ms", "split"]], feats], axis=1)
out.to_parquet(OUT_DATA, index=False, compression="zstd")

train_mask = df["split"] == "train"
tf = feats.loc[train_mask]
votes = df.loc[train_mask, "helpful_vote"]
missing = (tf.isna().mean() * 100).round(2).to_dict()
# Univariate Spearman with votes on the TRAINING split (analysis only; votes are not a feature).
spearman = {c: float(tf[c].corr(votes, method="spearman")) for c in FEATURE_COLUMNS}

report = {
    "run_id": RUN_ID,
    "rows_by_split": df["split"].value_counts().to_dict(),
    "feature_groups": FEATURE_GROUPS,
    "n_features": len(FEATURE_COLUMNS),
    "missing_percent_train": missing,
    "spearman_with_votes_train": spearman,
    "features_parquet": "data/processed/features_dev.parquet",
    "features_parquet_sha256": sha256_file(OUT_DATA),
    "clean_parquet_sha256": sha256_file(CLEAN),
}
OUT_REPORT.mkdir(parents=True, exist_ok=True)
(OUT_REPORT / "feature_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")

lines = [
    "# ReviewRank — Feature report (Stage 4)",
    "",
    f"Run `{RUN_ID}`. Features for excluded + train + validation rows; **the sealed test year and the stream were not loaded**. "
    f"{len(FEATURE_COLUMNS)} features in 5 groups. Source: `reports/features/feature_report.json`.",
    "",
    "Safety checks (unit tests, `tests/test_features.py`): changing votes never changes any feature; changing a later review never changes an "
    "earlier review's features; reviews with the same timestamp do not see each other; the output has no column named like a vote.",
    "",
    "| Group | Feature | Missing % (train) | Spearman with votes (train) |",
    "|---|---|---:|---:|",
]
for group, cols in FEATURE_GROUPS.items():
    for c in cols:
        lines.append(f"| {group} | {c} | {missing[c]} | {spearman[c]:+.3f} |")
lines += [
    "",
    "Notes: Spearman values are descriptive, on the training split only, and are not evidence a feature will help a ranker. "
    "`prior_*` features are missing for a product's first review by design (no earlier reviews). "
    "`prior_reviews` and `calendar` are a proxy for age/exposure and are ablated in Stage 5.",
]
(OUT_REPORT / "feature_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written:", OUT_REPORT / "feature_report.md")
