"""Stage 3: baseline ladder on VALIDATION groups. The sealed test split is never loaded.

Run from the repo root:  python scripts/run_stage3_baselines.py
Outputs: reports/baselines/baselines.json and baselines.md
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file  # noqa: E402
from reviewrank.baselines import (  # noqa: E402
    TfidfRidge,
    score_longest_first,
    score_lowest_rating_first,
    score_newest_first,
    score_oldest_first,
    score_random,
)
from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff  # noqa: E402
from reviewrank.evaluate import per_group_metrics  # noqa: E402
from reviewrank.labels import build_groups  # noqa: E402

CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
OUT = ROOT / "reports" / "baselines"
RUN_ID = "EXP-20260930-B03"  # B01 alphas [1,3,10] (best at grid edge); B02 extended grid; B03 adds the oldest_first baseline

eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
SEED = eval_cfg["ties"]["seed"]
REPS = eval_cfg["uncertainty"]["replicates"]
CI = eval_cfg["uncertainty"]["ci_level"]
ALPHAS = [3.0, 10.0, 30.0, 100.0]
METRICS = ["ndcg_at_3", "ndcg_at_5", "precision_at_3"]

COLUMNS = [
    "asin", "parent_asin", "user_id", "rating", "title", "text", "timestamp_ms",
    "helpful_vote", "verified_purchase", "split", "words", "text_seen_in_train",
]
# The Parquet filter means test-period rows are never read into memory.
df = pd.read_parquet(CLEAN, columns=COLUMNS, filters=[("split", "in", ["train", "validation"])])
assert set(df["split"].unique()) <= {"train", "validation"}, "test rows must never be loaded here"
train, val = df[df["split"] == "train"], df[df["split"] == "validation"]

elig, stats = build_groups(val)
elig = elig.reset_index(drop=True)
print(f"validation groups eligible: {stats['groups_eligible']} / size-ok {stats['groups_size_ok']}; rows {len(elig)}")

models: dict[str, np.ndarray] = {
    "random_seeded": score_random(elig, SEED),
    "newest_first": score_newest_first(elig),
    "oldest_first": score_oldest_first(elig),
    "longest_first": score_longest_first(elig),
    "lowest_rating_first": score_lowest_rating_first(elig),
}

# TF-IDF + Ridge: alpha chosen on validation mean NDCG@3 (this slightly favors the baseline; disclosed).
tfidf_fits, tfidf_seconds = {}, {}
for alpha in ALPHAS:
    t0 = time.time()
    tfidf_fits[alpha] = TfidfRidge(alpha=alpha).fit(train)
    tfidf_seconds[alpha] = round(time.time() - t0, 1)
    pg = per_group_metrics(elig, tfidf_fits[alpha].score(elig), seed=SEED)
    print(f"tfidf alpha={alpha}: ndcg@3={pg['ndcg_at_3'].mean():.4f} ({tfidf_seconds[alpha]}s)")
tfidf_valid_ndcg3 = {
    a: float(per_group_metrics(elig, tfidf_fits[a].score(elig), seed=SEED)["ndcg_at_3"].mean()) for a in ALPHAS
}
best_alpha = max(tfidf_valid_ndcg3, key=tfidf_valid_ndcg3.get)
best_tfidf = tfidf_fits[best_alpha]
models["tfidf_ridge"] = best_tfidf.score(elig)

per_group = {name: per_group_metrics(elig, s, seed=SEED) for name, s in models.items()}
ref = per_group["longest_first"]["group_id"].to_numpy()
for name, pg in per_group.items():
    assert np.array_equal(pg["group_id"].to_numpy(), ref), "all models must be scored on identical groups"
clusters = per_group["longest_first"]["parent_asin"].to_numpy()

summary = {}
for name, pg in per_group.items():
    summary[name] = {
        m: cluster_bootstrap_mean(pg[m].to_numpy(), clusters, replicates=REPS, ci_level=CI, seed=SEED)
        for m in METRICS
    }

paired = {}
for name in per_group:
    if name == "longest_first":
        continue
    paired[f"{name}_minus_longest_first"] = paired_cluster_bootstrap_diff(
        per_group[name]["ndcg_at_3"].to_numpy(), per_group["longest_first"]["ndcg_at_3"].to_numpy(),
        clusters, replicates=REPS, ci_level=CI, seed=SEED,
    )
paired["longest_first_minus_random_seeded"] = paired_cluster_bootstrap_diff(
    per_group["longest_first"]["ndcg_at_3"].to_numpy(), per_group["random_seeded"]["ndcg_at_3"].to_numpy(),
    clusters, replicates=REPS, ci_level=CI, seed=SEED,
)

# Sensitivity analyses (means only): different group definitions on the same validation rows.
SENS = {
    "min_size_3": {"min_size": 3},
    "min_size_10": {"min_size": 10},
    "require_two_voted": {"min_voted": 2},
    "quarter_window": {"window": "quarter"},
    "binary_grade": {"grade_mode": "binary"},
}
sensitivity = {}
for label, kwargs in SENS.items():
    rows, st = build_groups(val, **kwargs)
    rows = rows.reset_index(drop=True)
    scores = {
        "random_seeded": score_random(rows, SEED),
        "newest_first": score_newest_first(rows),
        "oldest_first": score_oldest_first(rows),
        "longest_first": score_longest_first(rows),
        "lowest_rating_first": score_lowest_rating_first(rows),
        "tfidf_ridge": best_tfidf.score(rows),
    }
    sensitivity[label] = {
        "groups_eligible": st["groups_eligible"],
        "reviews": st["reviews_in_eligible_groups"],
        **{n: float(per_group_metrics(rows, s, seed=SEED)["ndcg_at_3"].mean()) for n, s in scores.items()},
    }

run = {
    "run_id": RUN_ID,
    "split_evaluated": "validation (2020-07-01..2020-12-31); sealed test NOT loaded",
    "primary_group_definition": "parent_asin x calendar half-year, size>=5, >=1 voted review, grades 0/1/2-4/5+",
    "coverage": stats,
    "groups_evaluated": int(len(ref)),
    "products_evaluated": int(len(set(clusters))),
    "tie_seed": SEED,
    "bootstrap": {"replicates": REPS, "ci_level": CI, "cluster": "parent_asin", "seed": SEED},
    "tfidf": {
        "alphas_tried": ALPHAS,
        "validation_ndcg_at_3_by_alpha": tfidf_valid_ndcg3,
        "chosen_alpha": best_alpha,
        "selection_rule": "highest mean NDCG@3 on validation (slightly favors the baseline)",
        "fit_seconds_by_alpha": tfidf_seconds,
        "vectorizer": {"ngram_range": [1, 2], "min_df": 5, "max_features": 300000, "sublinear_tf": True},
        "target": "log1p(helpful_vote), training split only",
    },
    "inputs": {
        "clean_parquet_sha256": sha256_file(CLEAN),
        "eval_yaml_sha256": sha256_file(ROOT / "configs" / "eval.yaml"),
        "splits_yaml_sha256": sha256_file(ROOT / "configs" / "splits.yaml"),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
    },
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    },
    "git_commit": "none (repository has no commits yet at time of run)",
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "baselines.json").write_text(
    json.dumps(
        {"run": run, "summary": summary, "paired_ndcg_at_3": paired, "sensitivity_ndcg_at_3_means": sensitivity},
        indent=2,
        default=lambda o: o.item() if hasattr(o, "item") else str(o),
    )
    + "\n",
    encoding="utf-8",
)


def fmt(r):
    return f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"


order = ["random_seeded", "newest_first", "oldest_first", "lowest_rating_first", "longest_first", "tfidf_ridge"]
lines = [
    "# ReviewRank — Baseline ladder on validation (Stage 3)",
    "",
    f"Run `{RUN_ID}`. **Validation only (2020-07-01 to 2020-12-31); the sealed test year was not loaded.** Source: `reports/baselines/baselines.json`.",
    "",
    f"Groups evaluated: **{len(ref)}** (products: {len(set(clusters))}); eligible reviews: {len(elig)}. "
    f"Size-ok groups {stats['groups_size_ok']}, with a voted review {stats['groups_eligible']} "
    f"(coverage {stats['group_coverage']:.1%}). Intervals: product-clustered bootstrap, {REPS} replicates, {CI:.0%}.",
    "",
    "| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |",
    "|---|---|---|---|",
]
for n in order:
    s = summary[n]
    lines.append(f"| {n} | {fmt(s['ndcg_at_3'])} | {fmt(s['ndcg_at_5'])} | {fmt(s['precision_at_3'])} |")
lines += ["", "## Paired differences in NDCG@3 (same groups, same resampled products)", "",
          "| Comparison | Mean difference [95% CI] |", "|---|---|"]
for k, v in paired.items():
    lines.append(f"| {k} | {v['mean_diff']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}] |")
lines += ["", "## Sensitivity (mean NDCG@3 on the same validation rows, different group definitions)", "",
          "| Definition | Groups | " + " | ".join(order) + " |", "|---|---:|" + "---|" * len(order)]
for label, v in sensitivity.items():
    lines.append(f"| {label} | {v['groups_eligible']} | " + " | ".join(f"{v[n]:.4f}" for n in order) + " |")
lines += [
    "",
    f"TF-IDF Ridge: alpha chosen on validation from {ALPHAS} (best {best_alpha}); this slightly favors the baseline. "
    "Trained on the training split only (2015-01-01 to 2020-06-30).",
    "",
    "Caveat: a single half-year (2020H2) of validation; these are baselines, not the final claim. "
    "The primary comparison for the project is against the best baseline here.",
]
(OUT / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written:", OUT / "baselines.md")
