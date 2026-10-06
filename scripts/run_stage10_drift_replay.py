"""Stage 10: offline drift replay over the unlabeled stream (2022-01-01 onward).

Run from the repo root:  python scripts/run_stage10_drift_replay.py
Reads NON-LABEL columns only (the script never loads `helpful_vote`). Earlier rows, including the
sealed test year, are loaded only as HISTORY for the as-of features of stream rows; no labels,
no evaluation. Reference = validation period (2020-H2); the training period is a second reference.
Outputs: reports/monitoring/drift.md, drift.json, drift.png
This replay shows INPUT and SCORE drift only. It does not show accuracy changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file  # noqa: E402
from reviewrank.features import build_features  # noqa: E402
from reviewrank.monitor import psi, psi_level  # noqa: E402

CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
OUT = ROOT / "reports" / "monitoring"
COLUMNS = ["parent_asin", "timestamp_ms", "rating", "title", "text", "images_json", "verified_purchase", "words", "split"]
assert "helpful_vote" not in COLUMNS
manifest = json.loads((ROOT / "artifacts" / "manifests" / "primary_model.json").read_text(encoding="utf-8"))
CONTENT = manifest["feature_columns"]
MONITORED = ["words", "chars", "rating", "image_count", "verified", "prior_mean_rating", "prior_mean_words", "words_vs_prior_mean", "score"]

df = pd.read_parquet(CLEAN, columns=COLUMNS)  # all splits, non-label columns only
assert "helpful_vote" not in df.columns
feats = build_features(df)  # as-of features; earlier rows (any split) serve as history
booster = lgb.Booster(model_file=str(ROOT / "artifacts" / "models" / "lgbm_primary.txt"))
feats["score"] = booster.predict(feats[CONTENT])
feats["split"] = df["split"].to_numpy()
feats["month"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m").to_numpy()
feats["is_empty"] = (df["text"].str.strip().str.len() == 0).to_numpy()

ref = feats[feats["split"] == "validation"]
train_ref = feats[feats["split"] == "train"]
stream = feats[feats["split"] == "stream"]
months = sorted(stream["month"].unique())

rows = []
for m in months:
    cur = stream[stream["month"] == m]
    rec = {"month": m, "reviews": int(len(cur)), "empty_text_share": float(cur["is_empty"].mean()), "mean_words": float(cur["words"].mean()), "mean_rating": float(cur["rating"].mean())}
    for f in MONITORED:
        rec[f"psi_{f}"] = psi(ref[f], cur[f])
    rows.append(rec)
table = pd.DataFrame(rows)
psi_cols = [c for c in table.columns if c.startswith("psi_")]
stream_vs_train = {f: psi(train_ref[f], stream[f]) for f in MONITORED}
summary = {
    "max_psi_by_monitored_column": {c.replace("psi_", ""): float(table[c].max()) for c in psi_cols},
    "months_flagged_psi_above_0_25": {c.replace("psi_", ""): table.loc[table[c] > 0.25, "month"].tolist() for c in psi_cols},
    "months_with_fewer_than_1000_reviews": {r["month"]: r["reviews"] for r in rows if r["reviews"] < 1000},
    "whole_stream_vs_train_psi": stream_vs_train,
    "whole_stream_vs_validation_psi": {f: psi(ref[f], stream[f]) for f in MONITORED},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "drift.json").write_text(json.dumps({"rows_by_split": feats["split"].value_counts().to_dict(), "months": rows, "summary": summary, "clean_sha256": sha256_file(CLEAN), "model": manifest.get("run_id"), "git_commit": "none (repository has no commits yet at time of run)"}, indent=2, default=float) + "\n", encoding="utf-8")

fig, ax = plt.subplots(figsize=(8, 4))
for f in ["score", "words", "rating", "words_vs_prior_mean"]:
    ax.plot(table["month"], table[f"psi_{f}"], marker="o", markersize=3, label=f)
ax.axhline(0.25, color="red", linestyle="--", linewidth=1, label="0.25 (conventional 'large')")
ax.set_ylabel("PSI vs validation period")
ax.set_title("Monthly drift, unlabeled stream (input and score only)")
ax.tick_params(axis="x", labelrotation=60)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "drift.png", dpi=130)

lines = ["# ReviewRank — Offline drift replay (Stage 10)", "",
         f"Stream: {len(stream)} unlabeled reviews, {months[0]} to {months[-1]} (the last month may be partial). Reference: validation period (2020-07 to 2020-12), {len(ref)} reviews. "
         "**Input and score drift only; no accuracy claim.** The script reads no vote column. Source: `reports/monitoring/drift.json`, plot `drift.png`.", "",
         "| Monitored column | Max monthly PSI | Level of max | Whole stream vs validation | Whole stream vs train | Months above 0.25 |", "|---|---:|---|---:|---:|---|"]
for f in MONITORED:
    mx = summary["max_psi_by_monitored_column"][f]
    flagged = ", ".join(summary["months_flagged_psi_above_0_25"][f]) or "none"
    lines.append(f"| {f} | {mx:.3f} | {psi_level(mx)} | {summary['whole_stream_vs_validation_psi'][f]:.3f} | {stream_vs_train[f]:.3f} | {flagged} |")
small = summary["months_with_fewer_than_1000_reviews"]
lines += ["", f"Months with fewer than 1,000 reviews (PSI is noisy at small n): {small if small else 'none'}.",
          "", "PSI thresholds (0.1 / 0.25) are conventions. A flagged month means the inputs or scores look different from the reference; it does not say the ranking got worse. "
          "Checking accuracy would need votes, which arrive late (maturity rule, ADR-013)."]
(OUT / "drift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
