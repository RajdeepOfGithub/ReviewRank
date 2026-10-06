"""Stage 6b: do zero-shot sentiment probabilities add ranking signal? Validation only; test never loaded.

Run (main environment) after scripts/run_stage6a_sentiment_features.py has produced
data/processed/sent_finbert.parquet and sent_sst2.parquet:
    python scripts/run_stage6b_text_features.py
Rule (ADR-021): a variant is kept only if its paired NDCG@3 gain over the primary content-only model
has a CI lower bound above 0. Also reports polarity agreement with star ratings (domain-mismatch evidence).
Outputs: reports/text_models/stage6b.md and stage6b.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file  # noqa: E402
from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff  # noqa: E402
from reviewrank.evaluate import per_group_metrics  # noqa: E402
from reviewrank.labels import build_groups  # noqa: E402
from reviewrank.ranker import group_sizes, predict, sort_by_group, train_lambdarank  # noqa: E402

RUN_ID = "EXP-20260930-T01"
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
FEATS = ROOT / "data" / "processed" / "features_dev.parquet"
OUT = ROOT / "reports" / "text_models"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
SEED, REPS, CI = eval_cfg["ties"]["seed"], eval_cfg["uncertainty"]["replicates"], eval_cfg["uncertainty"]["ci_level"]
GRID = [{"num_leaves": nl, "min_data_in_leaf": md} for nl in (15, 31, 63) for md in (50, 200)]
CONTENT = json.loads((ROOT / "artifacts" / "manifests" / "primary_model.json").read_text(encoding="utf-8"))["feature_columns"]
FIN = ["finbert_positive", "finbert_negative", "finbert_neutral"]
SST = ["sst2_negative", "sst2_positive"]

clean = pd.read_parquet(CLEAN, columns=KEY + ["helpful_vote", "split", "text_seen_in_train"], filters=[("split", "in", ["train", "validation"])])
feats = pd.read_parquet(FEATS, filters=[("split", "in", ["train", "validation"])]).drop(columns="split")
df = clean.merge(feats, on=KEY, how="left", validate="one_to_one")
for tag in ("finbert", "sst2"):
    sent = pd.read_parquet(ROOT / "data" / "processed" / f"sent_{tag}.parquet")
    df = df.merge(sent, on=KEY, how="left", validate="one_to_one")
assert set(df["split"].unique()) <= {"train", "validation"}, "test rows must never be loaded here"
assert df[FIN + SST].notna().all().all(), "sentiment scores must cover every train/validation row"
train, val = df[df["split"] == "train"], df[df["split"] == "validation"]

tr = sort_by_group(build_groups(train, min_size=2, min_voted=1)[0])
va = sort_by_group(build_groups(val)[0])
tr_sizes, va_sizes = group_sizes(tr), group_sizes(va)
print(f"train groups {len(tr_sizes)}, validation groups {len(va_sizes)}", flush=True)

SETS = {
    "content_only (primary)": CONTENT,
    "plus_finbert": CONTENT + FIN,
    "plus_sst2": CONTENT + SST,
    "plus_both": CONTENT + FIN + SST,
}
best, per_group, grids = {}, {}, {}
for name, cols in SETS.items():
    rows = []
    for params in GRID:
        booster = train_lambdarank(tr[cols], tr["grade"], tr_sizes, va[cols], va["grade"], va_sizes, params=params)
        pg = per_group_metrics(va, predict(booster, va[cols]), seed=SEED)
        rows.append({**params, "best_iteration": int(booster.best_iteration), "val_ndcg_at_3": float(pg["ndcg_at_3"].mean()), "_pg": pg})
    top = max(rows, key=lambda r: r["val_ndcg_at_3"])
    best[name], per_group[name] = top, top["_pg"]
    grids[name] = [{k: v for k, v in r.items() if k != "_pg"} for r in rows]
    print(name, "best", {k: v for k, v in top.items() if k != "_pg"}, flush=True)

ref_name = "content_only (primary)"
ref_pg = per_group[ref_name]
ids = ref_pg["group_id"].to_numpy()
for pg in per_group.values():
    assert np.array_equal(pg["group_id"].to_numpy(), ids)
clusters = ref_pg["parent_asin"].to_numpy()
summary = {n: cluster_bootstrap_mean(pg["ndcg_at_3"].to_numpy(), clusters, replicates=REPS, ci_level=CI, seed=SEED) for n, pg in per_group.items()}
paired = {
    n: paired_cluster_bootstrap_diff(pg["ndcg_at_3"].to_numpy(), ref_pg["ndcg_at_3"].to_numpy(), clusters, replicates=REPS, ci_level=CI, seed=SEED)
    for n, pg in per_group.items() if n != ref_name
}
verdict = {n: ("keep" if p["ci_lower"] > 0 else "reject (CI lower bound not above 0)") for n, p in paired.items()}

# ---- polarity agreement with star ratings on ALL validation rows (rating is a noisy sentiment label)
v = val[val["rating"] != 3].copy()
truth_pos = (v["rating"] >= 4).to_numpy()
pol = {}
for tag, pos, neg in [("finbert", "finbert_positive", "finbert_negative"), ("sst2", "sst2_positive", "sst2_negative")]:
    pred_pos = (v[pos] > v[neg]).to_numpy()
    net = (v[pos] - v[neg])
    pol[tag] = {
        "rows": int(len(v)),
        "polarity_accuracy_vs_rating": float((pred_pos == truth_pos).mean()),
        "spearman_net_score_vs_rating": float(net.corr(v["rating"], method="spearman")),
    }
pol["finbert_share_neutral_argmax"] = float((val[FIN].to_numpy().argmax(1) == 2).mean())
pol["finbert_vs_sst2_net_spearman"] = float((val["finbert_positive"] - val["finbert_negative"]).corr(val["sst2_positive"] - val["sst2_negative"], method="spearman"))
vote_corr = {c: float(train[c].corr(train["helpful_vote"], method="spearman")) for c in FIN + SST}

result = {
    "run_id": RUN_ID,
    "split_evaluated": "validation; sealed test NOT loaded",
    "groups_evaluated": int(len(ids)),
    "summary_ndcg_at_3": summary,
    "paired_vs_primary_ndcg_at_3": paired,
    "verdict_by_rule": verdict,
    "grids": grids,
    "polarity_vs_rating_validation": pol,
    "spearman_with_votes_train": vote_corr,
    "inputs": {
        "clean_sha256": sha256_file(CLEAN), "features_sha256": sha256_file(FEATS),
        "sent_finbert_sha256": sha256_file(ROOT / "data" / "processed" / "sent_finbert.parquet"),
        "sent_sst2_sha256": sha256_file(ROOT / "data" / "processed" / "sent_sst2.parquet"),
    },
    "git_commit": "none (repository has no commits yet at time of run)",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "stage6b.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")


def fmt(r):
    return f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"


lines = [
    "# ReviewRank — Zero-shot sentiment features (Stage 6b)", "",
    f"Run `{RUN_ID}`. **Validation only; the sealed test year was not loaded.** {len(ids)} groups; product-clustered bootstrap ({REPS} replicates, {CI:.0%}). Source: `reports/text_models/stage6b.json`.", "",
    "| Model | NDCG@3 [95% CI] | Paired gain vs primary [95% CI] | Verdict by rule |", "|---|---|---|---|",
]
for n in SETS:
    if n == ref_name:
        lines.append(f"| {n} | {fmt(summary[n])} | — | reference |")
    else:
        p = paired[n]
        lines.append(f"| {n} | {fmt(summary[n])} | {p['mean_diff']:+.4f} [{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}] | {verdict[n]} |")
lines += ["", "## Polarity agreement with star ratings (validation rows with rating != 3; rating used as a noisy sentiment label)", "",
          "| Model | Rows | Polarity accuracy (positive vs negative) | Spearman(net score, rating) |", "|---|---:|---:|---:|"]
for tag in ("finbert", "sst2"):
    p = pol[tag]
    lines.append(f"| {tag} | {p['rows']} | {p['polarity_accuracy_vs_rating']:.3f} | {p['spearman_net_score_vs_rating']:+.3f} |")
lines += ["", f"FinBERT predicts *neutral* as its top class for {pol['finbert_share_neutral_argmax']:.1%} of validation reviews. "
          f"Spearman between FinBERT and SST-2 net scores: {pol['finbert_vs_sst2_net_spearman']:+.3f}.", "",
          "## Spearman of each score with votes (train; descriptive only)", "", "| Score | Spearman |", "|---|---:|"]
for c, v_ in vote_corr.items():
    lines.append(f"| {c} | {v_:+.3f} |")
(OUT / "stage6b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written:", OUT / "stage6b.md", flush=True)
