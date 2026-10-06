"""Stage 5 (run R02): LightGBM LambdaRank + ablations on VALIDATION groups. Sealed test never loaded.

Run from the repo root:  python scripts/run_stage5_ranker.py
Outputs: reports/ranker/ranker.md and ranker.json; models in artifacts/models/ (git-ignored);
artifacts/manifests/primary_model.json (columns and parameters of the primary candidate).

Design (ADR-019): the exposure/age proxies (prior_reviews, month, hour) are EXCLUDED from the
primary candidates, because they let a ranker learn the first-mover/position advantage instead of
review quality. The model with all features is reported for context. TF-IDF stacking is used only
if its paired interval against the content-only model has a lower bound above zero.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file, write_manifest  # noqa: E402
from reviewrank.baselines import TfidfRidge  # noqa: E402
from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff  # noqa: E402
from reviewrank.evaluate import per_group_metrics  # noqa: E402
from reviewrank.features import FEATURE_COLUMNS, FEATURE_GROUPS  # noqa: E402
from reviewrank.labels import build_groups  # noqa: E402
from reviewrank.ranker import (  # noqa: E402
    group_sizes,
    oof_tfidf_scores,
    predict,
    sort_by_group,
    train_lambdarank,
)

RUN_ID = "EXP-20260930-R02"
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
FEATS = ROOT / "data" / "processed" / "features_dev.parquet"
OUT = ROOT / "reports" / "ranker"
MODELS = ROOT / "artifacts" / "models"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
TFIDF_ALPHA = 30.0  # chosen in EXP-20260930-B02

eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
SEED = eval_cfg["ties"]["seed"]
REPS = eval_cfg["uncertainty"]["replicates"]
CI = eval_cfg["uncertainty"]["ci_level"]
METRICS = ["ndcg_at_3", "ndcg_at_5", "precision_at_3"]
GRID = [{"num_leaves": nl, "min_data_in_leaf": md} for nl in (15, 31, 63) for md in (50, 200)]

EXPOSURE_PROXIES = ["prior_reviews", "month", "hour"]
ALL = list(FEATURE_COLUMNS)
CONTENT = [c for c in ALL if c not in EXPOSURE_PROXIES]
CONTENT_TFIDF = CONTENT + ["tfidf_score"]
SETS = {"features_all": ALL, "content_only": CONTENT, "content_plus_tfidf": CONTENT_TFIDF}

# ---------------------------------------------------------------- load (train + validation only)
clean = pd.read_parquet(
    CLEAN,
    columns=KEY + ["title", "text", "helpful_vote", "split", "text_seen_in_train"],
    filters=[("split", "in", ["train", "validation"])],
)
feats = pd.read_parquet(FEATS, filters=[("split", "in", ["train", "validation"])]).drop(columns="split")
df = clean.merge(feats, on=KEY, how="left", validate="one_to_one")
assert len(df) == len(clean)
assert set(df["split"].unique()) <= {"train", "validation"}, "test rows must never be loaded here"
train, val = df[df["split"] == "train"].copy().reset_index(drop=True), df[df["split"] == "validation"].copy()

# ---------------------------------------------------------------- stacked TF-IDF feature (out-of-fold on train)
t0 = time.time()
train["tfidf_score"] = oof_tfidf_scores(train, alpha=TFIDF_ALPHA, n_splits=5, seed=SEED)
tfidf_full = TfidfRidge(alpha=TFIDF_ALPHA).fit(train)
val["tfidf_score"] = tfidf_full.score(val)
tfidf_seconds = round(time.time() - t0, 1)
print(f"tfidf oof + full fit: {tfidf_seconds}s", flush=True)

# ---------------------------------------------------------------- groups
train_groups, train_stats = build_groups(train, min_size=2, min_voted=1)  # looser definition for training
tr = sort_by_group(train_groups)
tr_sizes = group_sizes(tr)
val_elig, val_stats = build_groups(val)  # primary evaluation definition
va = sort_by_group(val_elig)
va_sizes = group_sizes(va)
print(f"train groups {len(tr_sizes)} (rows {len(tr)}); validation groups {len(va_sizes)} (rows {len(va)})", flush=True)


def fit_eval(cols, params):
    t = time.time()
    booster = train_lambdarank(tr[cols], tr["grade"], tr_sizes, va[cols], va["grade"], va_sizes, params=params)
    scores = predict(booster, va[cols])
    pg = per_group_metrics(va, scores, seed=SEED)
    return booster, scores, pg, round(time.time() - t, 1)


# ---------------------------------------------------------------- grids and final fits
grids, best = {}, {}
for set_name, cols in SETS.items():
    rows = []
    for params in GRID:
        booster, _, pg, secs = fit_eval(cols, params)
        rows.append({**params, "best_iteration": int(booster.best_iteration), "val_ndcg_at_3": float(pg["ndcg_at_3"].mean()), "seconds": secs})
    grids[set_name] = rows
    best[set_name] = max(rows, key=lambda r: r["val_ndcg_at_3"])
    print(set_name, "best", best[set_name], flush=True)
best_params = {k: {p: best[k][p] for p in ("num_leaves", "min_data_in_leaf")} for k in best}

fits = {name: fit_eval(cols, best_params[name]) for name, cols in SETS.items()}  # booster, scores, pg, secs

per_group = {
    "oldest_first": per_group_metrics(va, -va["timestamp_ms"].to_numpy(float), seed=SEED),
    "longest_first": per_group_metrics(va, va["words"].to_numpy(float), seed=SEED),
    "tfidf_ridge": per_group_metrics(va, va["tfidf_score"].to_numpy(), seed=SEED),
    **{f"lgbm_{name}": fits[name][2] for name in SETS},
}
ref_ids = per_group["longest_first"]["group_id"].to_numpy()
for name, pg in per_group.items():
    assert np.array_equal(pg["group_id"].to_numpy(), ref_ids), "identical groups required"
clusters = per_group["longest_first"]["parent_asin"].to_numpy()


def paired(a, b):
    return paired_cluster_bootstrap_diff(a, b, clusters, replicates=REPS, ci_level=CI, seed=SEED)


summary = {
    n: {m: cluster_bootstrap_mean(pg[m].to_numpy(), clusters, replicates=REPS, ci_level=CI, seed=SEED) for m in METRICS}
    for n, pg in per_group.items()
}

# ---------------------------------------------------------------- primary selection (pre-stated rule)
tfidf_gain = paired(per_group["lgbm_content_plus_tfidf"]["ndcg_at_3"], per_group["lgbm_content_only"]["ndcg_at_3"])
primary = "content_plus_tfidf" if tfidf_gain["ci_lower"] > 0 else "content_only"
pg_p = per_group[f"lgbm_{primary}"]
booster_p = fits[primary][0]
print("primary candidate:", primary, flush=True)

paired_vs = {
    f"primary_{primary}_minus_longest_first": paired(pg_p["ndcg_at_3"], per_group["longest_first"]["ndcg_at_3"]),
    f"primary_{primary}_minus_oldest_first": paired(pg_p["ndcg_at_3"], per_group["oldest_first"]["ndcg_at_3"]),
    f"primary_{primary}_minus_tfidf_ridge": paired(pg_p["ndcg_at_3"], per_group["tfidf_ridge"]["ndcg_at_3"]),
    "content_plus_tfidf_minus_content_only": tfidf_gain,
    f"features_all_minus_primary_{primary}": paired(per_group["lgbm_features_all"]["ndcg_at_3"], pg_p["ndcg_at_3"]),
    "oldest_first_minus_longest_first": paired(per_group["oldest_first"]["ndcg_at_3"], per_group["longest_first"]["ndcg_at_3"]),
}

# ---------------------------------------------------------------- ablations from the primary candidate
prim_cols, prim_params = SETS[primary], best_params[primary]
ablate_cols = {}
for group, cols in FEATURE_GROUPS.items():
    removed = [c for c in cols if c in prim_cols]
    if removed:
        ablate_cols[f"minus_{group}"] = [c for c in prim_cols if c not in removed]
ablate_cols["minus_words_vs_prior_mean"] = [c for c in prim_cols if c != "words_vs_prior_mean"]
if "tfidf_score" in prim_cols:
    ablate_cols["minus_tfidf_score"] = [c for c in prim_cols if c != "tfidf_score"]
ablate_cols["plus_exposure_proxies"] = prim_cols + EXPOSURE_PROXIES
ablation = {}
for name, cols in ablate_cols.items():
    _, _, pg, secs = fit_eval(cols, prim_params)
    ablation[name] = {"n_features": len(cols), "ndcg_at_3": float(pg["ndcg_at_3"].mean()), "vs_primary": paired(pg["ndcg_at_3"], pg_p["ndcg_at_3"]), "seconds": secs}
    print("ablation", name, round(ablation[name]["ndcg_at_3"], 4), flush=True)

# ---------------------------------------------------------------- importance and sensitivity
imp = pd.Series(booster_p.feature_importance("gain"), index=prim_cols)
importance = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()

SENS = {
    "min_size_3": {"min_size": 3}, "min_size_10": {"min_size": 10}, "require_two_voted": {"min_voted": 2},
    "quarter_window": {"window": "quarter"}, "binary_grade": {"grade_mode": "binary"},
}
sensitivity = {}
for label, kwargs in SENS.items():
    rows_s, st = build_groups(val, **kwargs)
    rows_s = rows_s.reset_index(drop=True)
    sc = {
        "oldest_first": -rows_s["timestamp_ms"].to_numpy(float),
        "longest_first": rows_s["words"].to_numpy(float),
        "tfidf_ridge": rows_s["tfidf_score"].to_numpy(),
        "lgbm_features_all": predict(fits["features_all"][0], rows_s[ALL]),
        f"lgbm_primary_{primary}": predict(booster_p, rows_s[prim_cols]),
    }
    sensitivity[label] = {"groups": st["groups_eligible"], **{n: float(per_group_metrics(rows_s, s, seed=SEED)["ndcg_at_3"].mean()) for n, s in sc.items()}}

# ---------------------------------------------------------------- save artifacts and report
MODELS.mkdir(parents=True, exist_ok=True)
booster_p.save_model(str(MODELS / "lgbm_primary.txt"), num_iteration=booster_p.best_iteration)
fits["features_all"][0].save_model(str(MODELS / "lgbm_features_all.txt"), num_iteration=fits["features_all"][0].best_iteration)
joblib.dump(tfidf_full, MODELS / "tfidf_ridge_alpha30.joblib")
write_manifest(
    {
        "run_id": RUN_ID,
        "primary": primary,
        "feature_columns": prim_cols,
        "uses_tfidf_score": "tfidf_score" in prim_cols,
        "tfidf_alpha": TFIDF_ALPHA,
        "lightgbm_params": {**prim_params, "best_iteration": int(booster_p.best_iteration)},
        "status": "candidate on validation only; not frozen (Stage 7)",
        "model_file": "artifacts/models/lgbm_primary.txt (git-ignored)",
    },
    ROOT / "artifacts" / "manifests" / "primary_model.json",
)

run = {
    "run_id": RUN_ID,
    "split_evaluated": "validation (2020-07-01..2020-12-31); sealed test NOT loaded",
    "groups_evaluated": int(len(ref_ids)),
    "validation_coverage": val_stats,
    "train_groups_min_size_2": {"groups": int(len(tr_sizes)), "rows": int(len(tr)), "stats": train_stats},
    "exposure_proxies_excluded_from_primary": EXPOSURE_PROXIES,
    "primary_rule": "content_plus_tfidf only if its paired NDCG@3 difference vs content_only has a CI lower bound > 0; otherwise content_only",
    "primary": primary,
    "grid": grids,
    "selected_params": best_params,
    "selection_rule": "highest mean validation NDCG@3 (early stopping also uses validation; not final until the Stage 7 freeze)",
    "seconds": {"tfidf_oof_and_full_fit": tfidf_seconds, **{name: fits[name][3] for name in SETS}},
    "tfidf_score_feature": f"5-fold out-of-fold by product on train (alpha {TFIDF_ALPHA}); full-train model for validation",
    "artifacts_sha256": {p.name: sha256_file(p) for p in sorted(MODELS.glob("*")) if p.is_file()},
    "inputs": {
        "clean_parquet_sha256": sha256_file(CLEAN), "features_parquet_sha256": sha256_file(FEATS),
        "eval_yaml_sha256": sha256_file(ROOT / "configs" / "eval.yaml"), "splits_yaml_sha256": sha256_file(ROOT / "configs" / "splits.yaml"),
    },
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
    "git_commit": "none (repository has no commits yet at time of run)",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ranker.json").write_text(
    json.dumps({"run": run, "summary": summary, "paired_ndcg_at_3": paired_vs, "ablation": ablation, "importance_gain": importance, "sensitivity_ndcg_at_3_means": sensitivity}, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
    encoding="utf-8",
)


def fmt(r):
    return f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"


def fdiff(r):
    return f"{r['mean_diff']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"


names = ["oldest_first", "longest_first", "tfidf_ridge", "lgbm_features_all", "lgbm_content_only", "lgbm_content_plus_tfidf"]
lines = [
    "# ReviewRank — LightGBM LambdaRank on validation (Stage 5, run R02)",
    "",
    f"Run `{RUN_ID}`. **Validation only (2020-07-01 to 2020-12-31); the sealed test year was not loaded.** Source: `reports/ranker/ranker.json`. The first run (`R01`) is kept in `ranker_R01.*`.",
    f"Groups evaluated: **{len(ref_ids)}**; intervals are product-clustered bootstrap ({REPS} replicates, {CI:.0%}). "
    f"Training groups: (product, half-year) with >=2 reviews and >=1 voted review ({len(tr_sizes)} groups, {len(tr)} rows).",
    "Selection is on validation, so these numbers are **candidates, not a final result**.",
    "",
    f"**Primary candidate: `{primary}`.** Exposure/age proxies ({', '.join(EXPOSURE_PROXIES)}) are excluded from the primary candidates (ADR-019); `features_all` includes them for context.",
    "",
    "| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |", "|---|---|---|---|",
]
for n in names:
    s = summary[n]
    lines.append(f"| {n} | {fmt(s['ndcg_at_3'])} | {fmt(s['ndcg_at_5'])} | {fmt(s['precision_at_3'])} |")
lines += ["", "## Paired differences in NDCG@3 (same groups, same resampled products)", "", "| Comparison | Difference [95% CI] |", "|---|---|"]
for k, v in paired_vs.items():
    lines.append(f"| {k} | {fdiff(v)} |")
lines += ["", "## Hyperparameter grid (validation NDCG@3)", "", "| Feature set | num_leaves | min_data_in_leaf | best iteration | NDCG@3 | seconds |", "|---|---:|---:|---:|---:|---:|"]
for set_name, rows in grids.items():
    for r in rows:
        mark = " **(selected)**" if r == best[set_name] else ""
        lines.append(f"| {set_name} | {r['num_leaves']} | {r['min_data_in_leaf']} | {r['best_iteration']} | {r['val_ndcg_at_3']:.4f}{mark} | {r['seconds']} |")
lines += ["", f"## Ablations from the primary candidate (`{primary}`); difference in NDCG@3 vs the primary", "", "| Variant | Features | NDCG@3 | Difference vs primary [95% CI] |", "|---|---:|---:|---|"]
for k, v in ablation.items():
    lines.append(f"| {k} | {v['n_features']} | {v['ndcg_at_3']:.4f} | {fdiff(v['vs_primary'])} |")
lines += ["", "## Top features by gain (primary)", "", "| Feature | Share of gain |", "|---|---:|"]
for k, v in importance.items():
    lines.append(f"| {k} | {v:.3f} |")
lines += ["", "## Sensitivity (mean NDCG@3, different group definitions)", "", f"| Definition | Groups | oldest_first | longest_first | tfidf_ridge | lgbm_features_all | lgbm_primary_{primary} |", "|---|---:|---:|---:|---:|---:|---:|"]
for label, v in sensitivity.items():
    lines.append(f"| {label} | {v['groups']} | {v['oldest_first']:.4f} | {v['longest_first']:.4f} | {v['tfidf_ridge']:.4f} | {v['lgbm_features_all']:.4f} | {v[f'lgbm_primary_{primary}']:.4f} |")
(OUT / "ranker.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written:", OUT / "ranker.md", flush=True)
