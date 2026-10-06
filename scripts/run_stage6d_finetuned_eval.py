"""Stage 6d: evaluate fine-tuned transformers standalone and as a blend with the primary ranker.

Validation only; the sealed test is never loaded. Run (main environment) after Stage 6c:
    python scripts/run_stage6d_finetuned_eval.py
Blend: score = (1 - w) * z(primary) + w * z(fine-tuned), z = standardized over validation rows (label-free).
The weight w is chosen by 2-fold CROSS-FITTING over validation products (choose on one half, evaluate on
the other half, then swap), which limits the optimism of tuning and testing on the same groups.
Rule (ADR-021): keep a blend only if its paired NDCG@3 gain over the primary has a CI lower bound > 0.
Outputs: reports/text_models/stage6d.md and stage6d.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff  # noqa: E402
from reviewrank.evaluate import per_group_metrics  # noqa: E402
from reviewrank.labels import build_groups  # noqa: E402
from reviewrank.ranker import product_folds  # noqa: E402

RUN_ID = "EXP-20260930-T02"
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
FEATS = ROOT / "data" / "processed" / "features_dev.parquet"
OUT = ROOT / "reports" / "text_models"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
SEED, REPS, CI = eval_cfg["ties"]["seed"], eval_cfg["uncertainty"]["replicates"], eval_cfg["uncertainty"]["ci_level"]
WEIGHTS = [round(w, 1) for w in np.arange(0.1, 1.0, 0.1)]
primary_cfg = json.loads((ROOT / "artifacts" / "manifests" / "primary_model.json").read_text(encoding="utf-8"))
CONTENT = primary_cfg["feature_columns"]
assert not primary_cfg["uses_tfidf_score"], "this script assumes the content-only primary"

clean = pd.read_parquet(CLEAN, columns=KEY + ["helpful_vote", "split", "text_seen_in_train"], filters=[("split", "in", ["validation"])])
feats = pd.read_parquet(FEATS, filters=[("split", "in", ["validation"])]).drop(columns="split")
val = clean.merge(feats, on=KEY, how="left", validate="one_to_one")
models = {}
for tag in ("distilbert", "finbert"):
    path = ROOT / "data" / "processed" / f"ft_{tag}_val.parquet"
    if path.exists():
        val = val.merge(pd.read_parquet(path)[KEY + [f"ft_{tag}"]], on=KEY, how="left", validate="one_to_one")
        models[tag] = f"ft_{tag}"
assert set(val["split"].unique()) == {"validation"}
assert models, "no fine-tuned score files found"
primary = lgb.Booster(model_file=str(ROOT / "artifacts" / "models" / "lgbm_primary.txt"))
val["primary_score"] = primary.predict(val[CONTENT])

va = build_groups(val)[0].reset_index(drop=True)
z = lambda s: (s - val[s.name].mean()) / val[s.name].std()  # noqa: E731  standardize over all validation rows (label-free)
va["z_primary"] = z(va["primary_score"])
pg = {"longest_first": per_group_metrics(va, va["words"].to_numpy(float), seed=SEED), "primary_content_only": per_group_metrics(va, va["primary_score"].to_numpy(), seed=SEED)}
for tag, col in models.items():
    assert va[col].notna().all()
    va[f"z_{tag}"] = (va[col] - val[col].mean()) / val[col].std()
    pg[f"{tag}_standalone"] = per_group_metrics(va, va[col].to_numpy(), seed=SEED)

ids = pg["longest_first"]["group_id"].to_numpy()
clusters = pg["longest_first"]["parent_asin"].to_numpy()
fold = product_folds(pd.Series(clusters), n_splits=2, seed=SEED)  # one fold id per group, split by product

blend_info = {}
for tag in models:
    by_w = {}
    for w in WEIGHTS:
        score = (1 - w) * va["z_primary"] + w * va[f"z_{tag}"]
        by_w[w] = per_group_metrics(va, score.to_numpy(), seed=SEED)["ndcg_at_3"].to_numpy()
    chosen, crossfit = {}, np.zeros(len(ids))
    for k in (0, 1):
        other = fold != k
        w_k = max(WEIGHTS, key=lambda w: by_w[w][other].mean())  # choose on the other half
        chosen[k] = w_k
        crossfit[fold == k] = by_w[w_k][fold == k]                  # evaluate on this half
    blend_info[tag] = {"weights_chosen_per_fold": chosen, "ndcg_by_weight_full_validation": {str(w): float(v.mean()) for w, v in by_w.items()}}
    pgb = pg[f"{tag}_standalone"].copy()
    pgb["ndcg_at_3"] = crossfit
    pg[f"blend_{tag}_crossfit"] = pgb

summary = {n: cluster_bootstrap_mean(p["ndcg_at_3"].to_numpy(), clusters, replicates=REPS, ci_level=CI, seed=SEED) for n, p in pg.items()}
ref = pg["primary_content_only"]["ndcg_at_3"].to_numpy()
paired = {n: paired_cluster_bootstrap_diff(p["ndcg_at_3"].to_numpy(), ref, clusters, replicates=REPS, ci_level=CI, seed=SEED)
          for n, p in pg.items() if n not in ("primary_content_only",)}
verdict = {n: ("keep" if paired[n]["ci_lower"] > 0 else "reject (CI lower bound not above 0)") for n in paired if n.startswith("blend_")}

result = {"run_id": RUN_ID, "split_evaluated": "validation; sealed test NOT loaded", "groups_evaluated": int(len(ids)),
          "summary_ndcg_at_3": summary, "paired_vs_primary_ndcg_at_3": paired, "verdict_for_blends": verdict, "blend_info": blend_info,
          "blend_weight_selection": "2-fold cross-fitting over validation products", "git_commit": "none (repository has no commits yet at time of run)"}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "stage6d.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")


def fmt(r):
    return f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"


lines = ["# ReviewRank — Fine-tuned transformers (Stage 6d)", "",
         f"Run `{RUN_ID}`. **Validation only; the sealed test year was not loaded.** {len(ids)} groups; product-clustered bootstrap ({REPS} replicates, {CI:.0%}).",
         "Blend weight chosen by 2-fold cross-fitting over validation products.", "",
         "| Model | NDCG@3 [95% CI] | Paired vs primary [95% CI] | Verdict |", "|---|---|---|---|"]
for n in pg:
    if n == "primary_content_only":
        lines.append(f"| {n} | {fmt(summary[n])} | — | reference |")
    else:
        p = paired[n]
        lines.append(f"| {n} | {fmt(summary[n])} | {p['mean_diff']:+.4f} [{p['ci_lower']:+.4f}, {p['ci_upper']:+.4f}] | {verdict.get(n, '')} |")
lines += ["", "## Blend weights", ""]
for tag, info in blend_info.items():
    lines.append(f"- {tag}: weights chosen per fold {info['weights_chosen_per_fold']}; full-validation NDCG@3 by weight: " + ", ".join(f"{w}:{v:.4f}" for w, v in info["ndcg_by_weight_full_validation"].items()))
(OUT / "stage6d.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
