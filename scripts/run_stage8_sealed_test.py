"""Stage 8: evaluate the FROZEN model.

    python scripts/run_stage8_sealed_test.py --split validation   # dry run: proves the pipeline reproduces the known validation numbers
    python scripts/run_stage8_sealed_test.py --split test         # THE ONE-TIME sealed evaluation (default)

Test mode protections: freeze.json must exist and every recorded hash must still match; the script refuses
to run if artifacts/manifests/test_opened.json exists; it writes that file when it finishes. Test labels are
read only in test mode, and only for the test split. Features for target rows are built as-of from earlier
rows' NON-LABEL columns (the whole file is loaded without votes first).
Outputs: reports/test/ (test mode) or reports/test_dryrun/ (validation mode): test.md and test.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file  # noqa: E402
from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff  # noqa: E402
from reviewrank.evaluate import per_group_metrics  # noqa: E402
from reviewrank.features import build_features  # noqa: E402
from reviewrank.labels import build_groups, window_label  # noqa: E402
from reviewrank.metrics import order_by_score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["test", "validation"], default="test")
args = ap.parse_args()
MAN = ROOT / "artifacts" / "manifests"
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
eval_cfg = yaml.safe_load((ROOT / "configs" / "eval.yaml").read_text(encoding="utf-8"))
SEED, REPS, CI = eval_cfg["ties"]["seed"], eval_cfg["uncertainty"]["replicates"], eval_cfg["uncertainty"]["ci_level"]
OUT = ROOT / "reports" / ("test" if args.split == "test" else "test_dryrun")

# ---------------------------------------------------------------- guards
freeze = None
if args.split == "test":
    if (MAN / "test_opened.json").exists():
        sys.exit("the sealed test was already opened; refusing to run again")
    if not (MAN / "freeze.json").exists():
        sys.exit("no freeze record; run scripts/run_stage7_freeze.py first")
    freeze = json.loads((MAN / "freeze.json").read_text(encoding="utf-8"))
    checks = {
        "model_sha256": (ROOT / freeze["chosen_model"]["model_file"], freeze["chosen_model"]["model_sha256"]),
        "primary_manifest_sha256": (MAN / "primary_model.json", freeze["chosen_model"]["primary_manifest_sha256"]),
        "tfidf_model_sha256": (ROOT / "artifacts" / "models" / "tfidf_ridge_alpha30.joblib", freeze["comparators"]["tfidf_model_sha256"]),
        "eval_yaml_sha256": (ROOT / "configs" / "eval.yaml", freeze["configs"]["eval_yaml_sha256"]),
        "splits_yaml_sha256": (ROOT / "configs" / "splits.yaml", freeze["configs"]["splits_yaml_sha256"]),
    }
    for name, (path, expected) in checks.items():
        if sha256_file(path) != expected:
            sys.exit(f"{name} changed since the freeze ({path.name}); refusing to open the test")
    for name, expected in freeze["code"]["src_files_sha256"].items():
        if sha256_file(ROOT / "src" / "reviewrank" / name) != expected:
            sys.exit(f"source file {name} changed since the freeze; refusing to open the test")
    if sha256_file(Path(__file__).resolve()) != freeze["evaluation"]["script_sha256"]:
        sys.exit("this evaluation script changed since the freeze; refusing to open the test")
    clean_manifest = json.loads((MAN / "clean_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(MAN / "clean_manifest.json") != freeze["data"]["clean_manifest_sha256"]:
        sys.exit("clean_manifest.json changed since the freeze; refusing to open the test")
    if sha256_file(CLEAN) != clean_manifest["output_sha256"]:
        sys.exit("reviews_clean.parquet does not match its manifest; refusing to open the test")

primary = json.loads((MAN / "primary_model.json").read_text(encoding="utf-8"))
CONTENT = primary["feature_columns"]
booster = lgb.Booster(model_file=str(ROOT / "artifacts" / "models" / "lgbm_primary.txt"))
tfidf = joblib.load(ROOT / "artifacts" / "models" / "tfidf_ridge_alpha30.joblib")

# ---------------------------------------------------------------- load: non-label columns for ALL rows, votes only for the target split
NOLABEL = KEY + ["parent_asin", "rating", "title", "text", "images_json", "verified_purchase", "words", "split", "text_seen_in_train"]
NOLABEL = list(dict.fromkeys(NOLABEL))
df = pd.read_parquet(CLEAN, columns=NOLABEL)
assert "helpful_vote" not in df.columns
feats = build_features(df)  # as-of features from earlier rows' non-label columns
votes = pd.read_parquet(CLEAN, columns=KEY + ["helpful_vote"], filters=[("split", "==", args.split)])
mask = (df["split"] == args.split).to_numpy()
tgt = pd.concat([df.loc[mask, NOLABEL], feats.loc[mask].drop(columns=["rating", "words"])], axis=1)
tgt = tgt.merge(votes, on=KEY, how="left", validate="one_to_one").reset_index(drop=True)
assert tgt["helpful_vote"].notna().all() and set(tgt["split"].unique()) == {args.split}
tgt["image_count"] = tgt["image_count"].astype(float)
print(f"split={args.split}: {len(tgt)} rows", flush=True)

tgt["tfidf_score"] = tfidf.score(tgt)
tgt["lgbm_score"] = booster.predict(tgt[CONTENT])
tgt["_t"] = tgt["timestamp_ms"]


def score_models(rows: pd.DataFrame) -> dict:
    return {
        "random_seeded": np.random.default_rng(SEED).random(len(rows)),
        "oldest_first": -rows["timestamp_ms"].to_numpy(float),
        "longest_first": rows["words"].to_numpy(float),
        "tfidf_ridge": rows["tfidf_score"].to_numpy(),
        "frozen_lgbm_content_only": rows["lgbm_score"].to_numpy(),
    }


# ---------------------------------------------------------------- primary evaluation
elig, stats = build_groups(tgt)
elig = elig.reset_index(drop=True)
scores = score_models(elig)
pg = {n: per_group_metrics(elig, s, seed=SEED) for n, s in scores.items()}
ids = pg["longest_first"]["group_id"].to_numpy()
for p in pg.values():
    assert np.array_equal(p["group_id"].to_numpy(), ids)
clusters = pg["longest_first"]["parent_asin"].to_numpy()


def boot(a):
    return cluster_bootstrap_mean(a, clusters, replicates=REPS, ci_level=CI, seed=SEED)


def pair(a, b, idx=None):
    a, b, c = np.asarray(a), np.asarray(b), clusters
    if idx is not None:
        a, b, c = a[idx], b[idx], c[idx]
    return paired_cluster_bootstrap_diff(a, b, c, replicates=REPS, ci_level=CI, seed=SEED)


METRICS = ["ndcg_at_3", "ndcg_at_5", "precision_at_3"]
summary = {n: {m: boot(p[m].to_numpy()) for m in METRICS} for n, p in pg.items()}
frozen = pg["frozen_lgbm_content_only"]["ndcg_at_3"].to_numpy()
paired = {
    "frozen_minus_longest_first": pair(frozen, pg["longest_first"]["ndcg_at_3"]),
    "frozen_minus_tfidf_ridge": pair(frozen, pg["tfidf_ridge"]["ndcg_at_3"]),
    "frozen_minus_oldest_first": pair(frozen, pg["oldest_first"]["ndcg_at_3"]),
    "frozen_minus_random_seeded": pair(frozen, pg["random_seeded"]["ndcg_at_3"]),
    "longest_first_minus_tfidf_ridge": pair(pg["longest_first"]["ndcg_at_3"], pg["tfidf_ridge"]["ndcg_at_3"]),
}

# ---------------------------------------------------------------- segments (group-level)
size = pg["longest_first"]["size"].to_numpy()
half = elig.groupby("group_id", sort=True)["timestamp_ms"].first()
window = window_label(pd.Series(half.to_numpy()), "half_year").to_numpy()
gmean_rating = elig.groupby("group_id", sort=True)["rating"].mean().to_numpy()
segments = {
    "group_size_5_6": size <= 6, "group_size_7_9": (size >= 7) & (size <= 9), "group_size_10_plus": size >= 10,
    "group_mean_rating_below_4": gmean_rating < 4.0, "group_mean_rating_4_plus": gmean_rating >= 4.0,
}
for w in sorted(set(window)):
    segments[f"window_{w}"] = window == w
seg_out = {}
for name, idx in segments.items():
    if idx.sum() < 30:
        seg_out[name] = {"groups": int(idx.sum()), "note": "fewer than 30 groups; not reported"}
        continue
    seg_out[name] = {
        "groups": int(idx.sum()),
        "frozen_ndcg_at_3": float(frozen[idx].mean()),
        "longest_first_ndcg_at_3": float(pg["longest_first"]["ndcg_at_3"].to_numpy()[idx].mean()),
        "paired_frozen_minus_longest_first": pair(frozen, pg["longest_first"]["ndcg_at_3"], idx),
    }

# ---------------------------------------------------------------- what do the top-3 look like? (bias analysis)
def top3_profile(score_vec):
    rows = []
    for gid, idx in elig.groupby("group_id").indices.items():
        order = order_by_score(score_vec[idx], SEED)[:3]
        sel = idx[order]
        rows.append(elig.iloc[sel][["words", "rating", "image_count", "verified", "helpful_vote"]].mean())
    return pd.DataFrame(rows).mean().round(3).to_dict()


oracle = elig["helpful_vote"].to_numpy(float)
profile = {n: top3_profile(s) for n, s in scores.items()}
profile["oracle_by_votes"] = top3_profile(oracle)
profile["all_reviews_in_eligible_groups"] = elig[["words", "rating", "image_count", "verified", "helpful_vote"]].mean().round(3).to_dict()

# ---------------------------------------------------------------- sensitivity (same frozen model, other group definitions)
SENS = {"min_size_3": {"min_size": 3}, "min_size_10": {"min_size": 10}, "require_two_voted": {"min_voted": 2},
        "quarter_window": {"window": "quarter"}, "binary_grade": {"grade_mode": "binary"}}
sens = {}
for label, kw in SENS.items():
    rows_s, st = build_groups(tgt, **kw)
    rows_s = rows_s.reset_index(drop=True)
    sc = score_models(rows_s)
    sens[label] = {"groups": st["groups_eligible"], **{n: float(per_group_metrics(rows_s, s, seed=SEED)["ndcg_at_3"].mean()) for n, s in sc.items()}}

result = {
    "mode": args.split, "freeze_id": freeze["freeze_id"] if freeze else None,
    "evaluated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "coverage": stats, "groups_evaluated": int(len(ids)), "products_evaluated": int(len(set(clusters))),
    "summary": summary, "paired_ndcg_at_3": paired, "segments": seg_out, "top3_profile": profile, "sensitivity_ndcg_at_3_means": sens,
    "bootstrap": {"replicates": REPS, "ci_level": CI, "cluster": "parent_asin", "seed": SEED},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "test.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n", encoding="utf-8")


def fmt(r):
    return f"{r['mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"


def fd(r):
    return f"{r['mean_diff']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"


title = "SEALED TEST (one-time evaluation)" if args.split == "test" else "DRY RUN ON VALIDATION (pipeline check, not a result)"
lines = [f"# ReviewRank — {title}", "",
         f"Freeze `{result['freeze_id']}`. Split: **{args.split}**. Groups: **{len(ids)}** (products {len(set(clusters))}); size-ok groups {stats['groups_size_ok']}, with a voted review {stats['groups_eligible']} (coverage {stats['group_coverage']:.1%}). Intervals: product-clustered bootstrap ({REPS} replicates, {CI:.0%}). Source: `{OUT.relative_to(ROOT)}/test.json`.", "",
         "| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |", "|---|---|---|---|"]
for n in ["random_seeded", "oldest_first", "longest_first", "tfidf_ridge", "frozen_lgbm_content_only"]:
    s = summary[n]
    lines.append(f"| {n} | {fmt(s['ndcg_at_3'])} | {fmt(s['ndcg_at_5'])} | {fmt(s['precision_at_3'])} |")
lines += ["", "## Paired differences in NDCG@3", "", "| Comparison | Difference [95% CI] |", "|---|---|"]
for k, v in paired.items():
    lines.append(f"| {k} | {fd(v)} |")
lines += ["", "## Segments (group level)", "", "| Segment | Groups | Frozen NDCG@3 | Longest-first NDCG@3 | Paired difference [95% CI] |", "|---|---:|---:|---:|---|"]
for k, v in seg_out.items():
    if "note" in v:
        lines.append(f"| {k} | {v['groups']} | — | — | {v['note']} |")
    else:
        lines.append(f"| {k} | {v['groups']} | {v['frozen_ndcg_at_3']:.4f} | {v['longest_first_ndcg_at_3']:.4f} | {fd(v['paired_frozen_minus_longest_first'])} |")
lines += ["", "## What the top-3 reviews look like (mean over groups)", "", "| Selection | words | rating | images | verified | votes |", "|---|---:|---:|---:|---:|---:|"]
for k, v in profile.items():
    lines.append(f"| {k} | {v['words']} | {v['rating']} | {v['image_count']} | {v['verified']} | {v['helpful_vote']} |")
lines += ["", "## Sensitivity (mean NDCG@3, other group definitions, same frozen model)", "", "| Definition | Groups | random | oldest_first | longest_first | tfidf_ridge | frozen_lgbm |", "|---|---:|---:|---:|---:|---:|---:|"]
for k, v in sens.items():
    lines.append(f"| {k} | {v['groups']} | {v['random_seeded']:.4f} | {v['oldest_first']:.4f} | {v['longest_first']:.4f} | {v['tfidf_ridge']:.4f} | {v['frozen_lgbm_content_only']:.4f} |")
(OUT / "test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))

if args.split == "test":
    (MAN / "test_opened.json").write_text(json.dumps({"opened_at_utc": result["evaluated_at_utc"], "freeze_id": freeze["freeze_id"], "result_sha256": sha256_file(OUT / "test.json")}, indent=2) + "\n", encoding="utf-8")
    print("test_opened.json written; the sealed test must not be opened again for this freeze")
