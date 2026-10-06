# ReviewRank

Cold-start ranking of newly posted Amazon product reviews by how many helpful votes they will collect, evaluated with a temporal backtest designed to avoid leakage and a sealed test year.

## In plain language

- **Question:** a brand-new review has no helpful votes, so it cannot be ranked by votes. Can a model, using only what is known when the review is posted, order the reviews on a product page so that those that later collect votes come first?
- **Data:** 494,121 Amazon reviews in Health and Personal Care (Amazon Reviews 2023), split by posting time: train 2015-01 to 2020-06, validation 2020-H2, a sealed test year 2021.
- **Compared:** random order, oldest-first, longest-review-first (the best simple rule), TF-IDF + Ridge, and a LightGBM LambdaRank model with 20 content features.
- **Result:** on the 2021 test (2,164 product-and-half-year groups, 1,696 products) the model scored NDCG@3 0.6558 versus 0.6223 for longest-first, a paired gain of +0.0335 with a 95% interval of [+0.0236, +0.0429].
- **What did not work:** FinBERT, a general sentiment model, fine-tuned DistilBERT and FinBERT, and a stacked TF-IDF score gave no detectable gain (every paired interval includes 0).
- **A shortcut I removed:** features that reveal how early a review was posted were left out because early reviews collect more votes through exposure.
- **Limits:** helpful votes are a proxy label shaped by exposure; one product category and one test year; a single training seed; retrospective evidence only; nothing is deployed.

## Headline result (sealed 2021 test, opened once)

2,164 evaluation groups across 1,696 products. Intervals are product-clustered bootstrap, 95%.

| Model | NDCG@3 |
|---|---|
| Random | 0.3398 [0.3254, 0.3538] |
| Oldest-first | 0.4442 [0.4285, 0.4602] |
| Longest-review-first (best simple baseline) | 0.6223 [0.6083, 0.6364] |
| TF-IDF + Ridge | 0.6124 [0.5987, 0.6272] |
| **LightGBM LambdaRank, content-only (frozen)** | **0.6558 [0.6424, 0.6699]** |

**What is scored.** The test split has 68,463 reviews, but only 35,022 of them (51.2%) fall in the 2,164 evaluated groups. 344 test reviews were removed because their text also appears in training, and a group counts only if it has at least 5 reviews and at least one review with a vote (2,164 of 3,026 size-eligible groups, 71.5%). NDCG is therefore measured on product pages where at least one review has a vote; the same rule applies to every model.

Paired gain over the best baseline: **+0.0335 [+0.0236, +0.0429]** (about 5.4% relative; a ratio of two point estimates). It is positive with an interval above 0 in every segment checked and in both halves of 2021, and the frozen model also scores higher than longest-first under five alternative group definitions (point estimates only, no intervals). On validation the gain looked larger (+0.0580), mainly because the longest-first baseline was weaker in 2020-H2 (0.5991 on validation vs 0.6223 on test); the model's own score barely moved (0.6571 on validation, 0.6558 on the test).

**What did not work (all reported, none tuned away):** a stacked TF-IDF score; zero-shot FinBERT and a general sentiment model as features; fine-tuned DistilBERT and FinBERT (standalone 0.5816 and 0.5840 on validation, below the length baseline) and their blends with the ranker. None of these additions showed a detectable gain: every paired interval includes 0. The fine-tuning was a single 60,000-review, 2-epoch run per model with no search, so this is a result for that setup, not a statement about transformers in general. FinBERT labels 80.8% of validation reviews neutral (domain mismatch), and sentiment largely duplicates the star rating.

**A shortcut I removed:** the number of earlier reviews and the month/hour likely encode how early a review was posted, and early reviews collect votes through exposure. Adding them raised validation NDCG@3 by +0.0177 [+0.0101, +0.0251] (+0.0152 [+0.0076, +0.0232] in the ablation table of `reports/ranker/ranker.md`, which uses a different fit), so they were excluded from the primary model to avoid rewarding position bias. The decision was made after seeing the validation ablation and lowers the reported score.

**Known bias:** the model's top-3 picks average 0.44 images per review versus 0.25 for the most-voted reviews, and run longer (about 75 vs 56 words). See `docs/model_card.md`.

## How the evaluation is protected

- Temporal split by posting time: train 2015-01 to 2020-06, validation 2020-H2, sealed test 2021, unlabeled stream 2022 onward.
- Labels: grades 0/1/2–4/5+ from absolute votes, compared only inside (product, half-year) groups of at least 5 reviews; reviews must be at least 18 months old at the dataset snapshot.
- Tests show that votes never change a feature and that a later review never changes an earlier review's features; exact duplicates removed; text seen in train excluded from evaluation. These checks reduce the risk of leakage; they cannot prove its absence.
- A freeze record with SHA-256 hashes (`artifacts/manifests/freeze.json`) was written before the test; the test script refuses to run if the model, configs, source files, evaluation script or data differ, and refuses a second run while `artifacts/manifests/test_opened.json` exists. This is a file-existence guard in a repository that has no commit history (`freeze.json` records `git_commit: none`), so it does not independently prove that the freeze preceded the test.
- 119 automated tests: 110 verified in a fresh environment, 9 not yet verified on an unrestricted machine (see "Known issues and caveats").

## Pipeline

| Stage | Script / module | Output |
|---|---|---|
| 1 Acquire | `scripts/run_stage1.py` | checksum-verified source, typed Parquet, `source_manifest.json` |
| 2 Audit, EDA, clean | `scripts/stage2_audit_eda.py`, `python -m reviewrank.clean` | `reports/audit`, `reports/eda`, cleaned Parquet |
| 3 Baselines | `scripts/run_stage3_baselines.py` | `reports/baselines` |
| 4 Features | `scripts/run_stage4_features.py` | `reports/features` |
| 5 Ranker | `scripts/run_stage5_ranker.py` | `reports/ranker`, model files |
| 6 Text models | `scripts/gpu_smoke_test.py`, `run_stage6a`–`6d` (GPU environment) | `reports/text_models`, `reports/gpu` |
| 7 Freeze | `scripts/run_stage7_freeze.py` | `artifacts/manifests/freeze.json` |
| 8 Sealed test | `scripts/run_stage8_sealed_test.py` | `reports/test` |
| 10 Monitor | `scripts/run_stage10_drift_replay.py` | `reports/monitoring` |

Every decision and number is traced in `docs/planning/03_METRICS_AND_DECISIONS.md` (run IDs, input hashes, ADR-001 to ADR-025). Contract: `docs/problem_contract.md`; `docs/model_card.md`; `docs/deploy_aws.md` (a runbook that was not executed).

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.lock
pip install -e . --no-deps
pytest
```

`requirements.lock` is the package set of the environment used for the reported runs (Python 3.12.10). `pyproject.toml` itself pins nothing.

**Windows:** create the virtual environment in a short path (for example `C:\rr\.venv`). pyarrow contains deeply nested files, and `pip install` fails with "No such file or directory ... Windows Long Path support" when the full path is long, as it is for this repository's default location.

To regenerate the data products, run the stage scripts in order (Stage 1 downloads about 227 MB). The sealed-test script will not run a second time on this freeze; deleting `test_opened.json` voids the one-time property. Stage 6 (GPU) needs a separate PyTorch environment; see `requirements-stage6.txt` for the versions recorded in `reports/text_models`. On Windows with an AMD Ryzen AI Max chip I used AMD's ROCm 7.2.1 wheels in a separate short-path environment, after a forward-and-backward smoke test.

## Getting the data and regenerating excluded files

The repository does not contain the data, the cleaned Parquet files, the cached transformer weights or most model files; they are git-ignored (`.gitignore`).

- **Data:** `python scripts/run_stage1.py` downloads the Health and Personal Care review file (about 227 MB) from Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`; the revision and SHA-256 are pinned in `artifacts/manifests/source_manifest.json`) into `data/raw/` and converts it to Parquet in `data/processed/`. The data is not redistributed here (see "Data and license").
- **Cleaned data and features:** `python -m reviewrank.clean` (Stage 2) writes `data/processed/reviews_clean.parquet`; `scripts/run_stage4_features.py` writes the feature table.
- **Models:** `scripts/run_stage3_baselines.py` and `scripts/run_stage5_ranker.py` retrain the baselines and the LightGBM models into `artifacts/models/`. Only `lgbm_primary.txt` (the frozen model, under 1 MB) is committed, so the API example works. `tfidf_ridge_alpha30.joblib` is not committed, so the sealed-test script cannot be re-verified from a fresh clone (it is also designed to run only once).
- **Transformer weights and Stage 6 score files:** downloaded into `hf_cache/` and written to `data/processed/` when `run_stage6a`–`6d` run (see `requirements-stage6.txt`).
- A rerun is not guaranteed to reproduce the frozen model's file hash, so it will not necessarily match `freeze.json`. The committed reports are the reference results.

## Known issues and caveats

- **Seeds.** Single training seed 42 (LightGBM, bootstrap and tie-breaking). Seed variance and variance across different splits were not measured.
- **Tests.** 110 of 119 tests were verified in a fresh virtual environment. The other 9 (`tests/test_baselines.py`, `tests/test_ranker.py`) could not be collected on the audit machine because Windows Application Control blocked a scikit-learn DLL; they are not yet verified on an unrestricted machine.
- **First reviews and residual exposure.** The count of earlier reviews was removed, but the product-history features (`prior_mean_rating`, `prior_mean_words`, `words_vs_prior_mean`, `rating_minus_prior_mean`) are missing for the first review of a product, which still marks it as early. Removing the product-history group costs −0.0227 NDCG@3 on validation, and `words_vs_prior_mean` carries 41.2% of the model's gain (`reports/ranker/ranker.md`). Whether this reflects residual exposure or real signal was not tested.
- **`primary_model.json` says "not frozen".** `artifacts/manifests/primary_model.json` still reads "candidate on validation only; not frozen" and "(git-ignored)", although this is the frozen model that the API and Dockerfiles load. The file is covered by the SHA-256 recorded in `freeze.json`, so editing it would invalidate the freeze; the freeze record and this README are the authority.
- **Configs.** `configs/eval.yaml` and `configs/splits.yaml` still say `status: provisional` and list baselines (`newest_first`, `lowest_rating_first`) that differ from the baselines in the test report (`oldest_first`). Both are covered by the freeze hashes and were left unedited.
- **Stage 6 environment.** `torch` and `transformers` are not in `requirements.lock`; see `requirements-stage6.txt`.

## Serve the model

```bash
uvicorn reviewrank.serve:app_from_env --factory --port 8000
curl -X POST localhost:8000/rank -H "content-type: application/json" -d '{
  "product_history": {"n_reviews": 5, "mean_rating": 4.2, "mean_words": 30},
  "reviews": [{"id": "a", "text": "Short.", "rating": 5}, {"id": "b", "text": "A longer, more detailed review ...", "rating": 3}]
}'
```

`GET /health`, `GET /model-info`, `POST /rank`. Unknown fields, including any vote field, are rejected with HTTP 422; a parity test (in-process) shows the API builds the same features as the batch pipeline. The server itself has not been started as part of the verification described here, and there are no latency measurements.

**Not done:** the Docker image has not been built and nothing is deployed on AWS (the runbook is `docs/deploy_aws.md`; the AWS CLI had no credentials configured). Offline drift replay on 54,540 unlabeled 2022–2023 reviews: no month with at least 1,000 reviews exceeds PSI 0.25 (maximum 0.218). Five sparse months (fewer than 1,000 reviews each) are excluded from that statement; across all 21 months the maximum PSI is 1.854 (2023-09, 17 reviews). This is input and score drift only, not an accuracy claim.

## Limitations

Votes are a proxy label affected by exposure; one product category and one year of test data; retrospective evidence, no online experiment; evaluation covers only product pages with at least one voted review; short reviews dominate; the model over-prefers images and length; no fairness analysis (no reviewer attributes). Details in `docs/model_card.md`.

## Data and license

Amazon Reviews 2023 (McAuley Lab), category Health and Personal Care, pinned to a dataset revision and checksum-verified. No dataset licence is assigned; the maintainers permit academic, non-commercial use, so this project is non-commercial and does not redistribute raw data or review text. Cite: Hou et al., "Bridging Language and Items for Retrieval and Recommendation", arXiv:2403.03952 (2024).
