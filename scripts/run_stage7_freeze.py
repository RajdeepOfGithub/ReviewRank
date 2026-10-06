"""Stage 7: write the model/policy FREEZE record before the sealed test is opened.

Run from the repo root:  python scripts/run_stage7_freeze.py --primary content_only
It records the exact model file and its sha256, the feature list, the metric and comparator, the
confidence-interval method, config and code hashes, and the validation numbers the decision rests on.
It REFUSES to run if the sealed test was already opened (artifacts/manifests/test_opened.json exists).
Output: artifacts/manifests/freeze.json (committed). The Stage 8 script refuses to run without it and
re-checks every hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import sha256_file, write_manifest  # noqa: E402

MAN = ROOT / "artifacts" / "manifests"
ap = argparse.ArgumentParser()
ap.add_argument("--primary", default="content_only")
ap.add_argument("--freeze-id", default="FREEZE-20260930-01")
args = ap.parse_args()

if (MAN / "test_opened.json").exists():
    sys.exit("the sealed test was already opened; a new freeze needs a new sealed evaluation design (new version)")
if (MAN / "freeze.json").exists():
    sys.exit("freeze.json already exists; do not overwrite a freeze record (create a new version instead)")

primary = json.loads((MAN / "primary_model.json").read_text(encoding="utf-8"))
if primary["primary"] != args.primary:
    sys.exit(f"primary_model.json says {primary['primary']!r}, not {args.primary!r}")
ranker = json.loads((ROOT / "reports" / "ranker" / "ranker.json").read_text(encoding="utf-8"))
baselines = json.loads((ROOT / "reports" / "baselines" / "baselines.json").read_text(encoding="utf-8"))

src_hashes = {p.name: sha256_file(p) for p in sorted((ROOT / "src" / "reviewrank").glob("*.py"))}
code_sha = hashlib.sha256(json.dumps(src_hashes, sort_keys=True).encode()).hexdigest()

freeze = {
    "freeze_id": args.freeze_id,
    "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "chosen_model": {
        "primary": primary["primary"],
        "model_file": "artifacts/models/lgbm_primary.txt",
        "model_sha256": sha256_file(ROOT / "artifacts" / "models" / "lgbm_primary.txt"),
        "primary_manifest": "artifacts/manifests/primary_model.json",
        "primary_manifest_sha256": sha256_file(MAN / "primary_model.json"),
        "feature_columns": primary["feature_columns"],
        "lightgbm_params": primary["lightgbm_params"],
        "uses_tfidf_score": primary["uses_tfidf_score"],
    },
    "comparators": {
        "primary_comparator": "longest_first",
        "other_baselines": ["random_seeded", "oldest_first", "tfidf_ridge (artifacts/models/tfidf_ridge_alpha30.joblib)"],
        "tfidf_model_sha256": sha256_file(ROOT / "artifacts" / "models" / "tfidf_ridge_alpha30.joblib"),
    },
    "metric": {"primary": "mean NDCG@3 over eligible groups (gain 2^rel-1, log2 discount)", "also": ["NDCG@5", "Precision@3"], "coverage_reported": True},
    "uncertainty": {"method": "product-clustered bootstrap", "replicates": 2000, "ci_level": 0.95, "seed": 42, "paired_vs_comparator": True},
    "group_definition": "parent_asin x calendar half-year, size>=5, >=1 voted review, grades 0/1/2-4/5+ (LABEL-v1)",
    "sensitivity_definitions": ["min_size_3", "min_size_10", "require_two_voted", "quarter_window", "binary_grade"],
    "tie_rule": "seeded random permutation, seed 42",
    "split": "test = 2021-01-01..2021-12-31 UTC; text_seen_in_train rows excluded (ADR-009); maturity 18 months (ADR-013)",
    "decision_basis_validation": {
        "run_ids": [ranker["run"]["run_id"], baselines["run"]["run_id"], "EXP-20260930-T01"],
        "primary_ndcg_at_3_validation": ranker["summary"]["lgbm_content_only"]["ndcg_at_3"],
        "longest_first_ndcg_at_3_validation": ranker["summary"]["longest_first"]["ndcg_at_3"],
        "paired_primary_minus_longest_first_validation": ranker["paired_ndcg_at_3"]["primary_content_only_minus_longest_first"],
    },
    "rejected_candidates": {
        "tfidf_stacking": "paired -0.0014 [-0.0073, +0.0041] (ADR-021)",
        "zero_shot_sentiment_features": "paired +0.0000 / -0.0006 / -0.0023, CIs include 0 (ADR-022)",
        "exposure_age_features": "excluded by design (ADR-019)",
    },
    "configs": {"eval_yaml_sha256": sha256_file(ROOT / "configs" / "eval.yaml"), "splits_yaml_sha256": sha256_file(ROOT / "configs" / "splits.yaml")},
    "evaluation": {"script": "scripts/run_stage8_sealed_test.py", "script_sha256": sha256_file(ROOT / "scripts" / "run_stage8_sealed_test.py")},
    "data": {"clean_manifest_sha256": sha256_file(MAN / "clean_manifest.json"), "source_manifest_sha256": sha256_file(MAN / "source_manifest.json")},
    "code": {"src_files_sha256": src_hashes, "src_combined_sha256": code_sha, "git_commit": "none (repository has no commits yet)"},
    "rules": [
        "The sealed test is opened exactly once by scripts/run_stage8_sealed_test.py.",
        "The result is reported whatever it is, including a loss to the comparator.",
        "No tuning or model change after the test is opened; a later change needs a new version and a new sealed evaluation design.",
    ],
    "test_rows_untouched": True,
}
write_manifest(freeze, MAN / "freeze.json")
print(json.dumps({k: freeze[k] for k in ("freeze_id", "frozen_at_utc")}, indent=2))
print("freeze written:", MAN / "freeze.json")
