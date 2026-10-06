# ReviewRank — Metrics, experiment and decision ledger

**Working ledger.** Only the feasibility measurements below are real results. Everything else is a blank template. Append a record after each completed stage; keep failed and negative results. Link machine-generated artifacts rather than typing a favorable number from memory.

## Evidence rules

Every result needs: source manifest sha256, category, split manifest hash, label-definition ID, query-group rule, run ID, git commit, dependency lock, feature/transform version, model/config, seed, evaluated rows and groups, coverage, baseline on identical rows, metric with CI method, artifact path, date.

Distinguish: **observed metric on a temporal backtest**, **paired difference vs baseline with CI**, **hypothetical product impact** (none is measured here). Helpful votes are a proxy label, not truth and not purchase impact.

## Stage progress (append)

| Date | Stage | Status | Evidence path / commit | Finding, blocker, next action |
|---|---|---|---|---|
| 2026-09-30 | Feasibility | Done | private feasibility notes (not in this repository) | Data usable; label-age confound found; contract next |
| 2026-09-30 | 0 Contract | Done (no data, no model) | `docs/problem_contract.md`, `configs/*.yaml`, `tests/` (24 passed) | Label window/cutoffs/maturity and split dates stay PROVISIONAL until Stage 2 ADR |
| 2026-09-30 | 1 Acquire | Done | `artifacts/manifests/source_manifest.json`, `scripts/run_stage1.py`, `tests/test_acquire.py`, `tests/test_ingest.py` (34 passed) | License verified (none assigned; non-commercial research use); sha256 matches source; 494,121 rows; max timestamp 2023-09-11. Risk for Stage 2: ~8 reviews/product may make product-window groups tiny |
| 2026-09-30 | 2 Audit + EDA | Done | `reports/audit/data_quality.md`, `reports/eda/eda_findings.md`, `artifacts/manifests/clean_manifest.json`, `src/reviewrank/clean.py` (40 tests passed) | ADR-008..014 recorded; label and split frozen; sealed test untouched. Next: label builder + tests, baselines |
| 2026-09-30 | 3 Baselines | Done | `reports/baselines/baselines.md` + `baselines.json` (run EXP-20260930-B02; B01 kept as `baselines_B01.*`), `src/reviewrank/{labels,bootstrap,evaluate,baselines}.py`, 71 tests passed | Best baseline = longest-first, NDCG@3 0.5991 [0.5819, 0.6172] on validation. TF-IDF Ridge indistinguishable. Sealed test not loaded |
| 2026-09-30 | 4 Features | Done | `src/reviewrank/features.py`, `tests/test_features.py` (11 tests; 87 total passing), `reports/features/feature_report.md`, `data/processed/features_dev.parquet` (git-ignored) | 23 as-of features in 5 groups; sealed test and stream not loaded; ADR-017/018 |
| 2026-09-30 | 5 Ranker | Done (candidate on validation; not frozen) | `reports/ranker/ranker.md/.json` (R02; R01 kept as `ranker_R01.*`), `reports/baselines/baselines.md` (B03; B02 kept), `artifacts/manifests/primary_model.json`, `src/reviewrank/ranker.py`, 87 tests passing | Primary candidate = content-only LightGBM LambdaRank, NDCG@3 0.6571 [0.6403, 0.6736]; +0.0580 [+0.0451, +0.0709] over longest-first. TF-IDF stacking rejected by rule. Sealed test not loaded. Next: Stage 6 text models (FinBERT etc.) |
| 2026-09-30 | 6 Text models — GPU smoke test | Done (passed) | `scripts/gpu_smoke_test.py`, `reports/gpu/smoke_test.json`; env `<HOME>\rrg` (short path, outside repo) | torch 2.9.1+rocm7.2.1; Radeon 8060S visible (37.7 GB addressable); matmul matches CPU; transformer forward+backward+optimizer step works (loss 1.38 → 0.0041 in 5 steps); FinBERT forward+backward works. AMD warns flash/mem-efficient attention is experimental. First attempt inside the repo failed on the Windows long-path limit |
| 2026-09-30 | 6a/6b Zero-shot sentiment features | Done | `scripts/run_stage6a_sentiment_features.py`, `run_stage6b_text_features.py`, `reports/text_models/stage6b.md/.json`, `sent_*_manifest.json` | FinBERT and a general SST-2 model scored all 329,274 train+validation reviews (fp16); neither adds ranking signal (paired gains +0.0000 / −0.0006 / −0.0023, all CIs include 0); all rejected by rule |
| 2026-09-30 | 6c Fine-tuned transformers | In progress | `scripts/run_stage6c_finetune.py` (DistilBERT then FinBERT, 60k train rows, 2 epochs, fixed hyperparameters; attempt 1 with 100k rows was killed at the 2-hour background limit and produced no output) | DistilBERT done (section "Stage 6c/6d results"); FinBERT attempt 2 hung mid-epoch-2 when the laptop slept overnight (log kept as `ft_finbert_attempt2_sleep_hang.log`); attempt 3 runs with a process-level Windows keep-awake request |
| 2026-09-30 | 10 Serve (partial) | Code + tests done; Docker image NOT built yet | `src/reviewrank/scoring.py`, `serve.py`, `tests/test_scoring.py`, `tests/test_api.py` (109 tests passing), `Dockerfile`, `requirements-serve.txt`, `.dockerignore` | FastAPI `/health`, `/model-info`, `/rank`; features match the batch pipeline (parity test); vote fields and unknown fields rejected (422). Docker daemon was not running (Docker Desktop not started), so no image yet. Points at the CANDIDATE primary model; re-point to the frozen model after Stage 7. No latency numbers yet |
| 2026-09-30 | 10 Monitor (offline drift replay) | Done (candidate model) | `src/reviewrank/monitor.py`, `tests/test_monitor.py`, `scripts/run_stage10_drift_replay.py`, `reports/monitoring/drift.md/.json/.png` (119 tests passing) | No monitored input or score exceeds PSI 0.25 in any month with ≥1,000 reviews (max 0.218). The only flagged month is 2023-09 with 17 reviews. Input/score drift only; not an accuracy claim |
| 2026-10-01 | 7 Freeze | **DONE: FREEZE-20260930-01 at 2026-10-01T14:18:27Z** (section "Model freeze") | `artifacts/manifests/freeze.json`, `scripts/run_stage7_freeze.py` | Writes `artifacts/manifests/freeze.json` (model sha256, features, params, metric, comparators, config/code hashes, validation basis, rejected candidates). Refuses if the test was opened or a freeze exists |
| 2026-09-30 | 8 Sealed test | **DONE 2026-10-01: sealed test opened ONCE** (result in section "Stage 8 results — SEALED TEST"); earlier dry run on validation verified the pipeline | `scripts/run_stage8_sealed_test.py`, `reports/test_dryrun/test.md/.json` | The dry run reproduces every known validation number exactly. Test mode refuses without a freeze record with matching hashes and refuses a second run (`test_opened.json`) |
| 2026-10-01 | 9 Bias + error analysis | Done (part of the sealed test report) | `reports/test/test.md` (segments, top-3 profile) | Model beats longest-first in every segment (+0.030 to +0.039) and both 2021 halves; over-prefers images (0.44 images per top-3 review vs 0.246 for the most-voted; mean image count, not a share) and length (74.9 vs 56.0 words). Fairness not evaluated (no reviewer attributes) |
| 2026-10-01 | 11 Communicate | Done | `README.md`, `docs/model_card.md`, this ledger | README and model card use sealed-test numbers only. **Not done:** Docker image not built, nothing deployed on AWS (no CLI credentials), no commits yet (the GitHub repository has not been created) |

## Source/version and license record

```yaml
dataset_name: Amazon Reviews 2023 (McAuley Lab)
official_page: https://amazon-reviews-2023.github.io/
hf_repo: McAuley-Lab/Amazon-Reviews-2023
category: Health_and_Personal_Care
file: raw/review_categories/Health_and_Personal_Care.jsonl
download_url: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Health_and_Personal_Care.jsonl
hf_revision: 2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e   # download pinned to this dataset commit
retrieved_at_utc: 2026-09-30T21:53:27Z
raw_bytes: 226990293
raw_sha256: b532849dcf8e42910ce688ad423b3c1622ce860e7bf88a3641358bb89f345d02   # equals the checksum published by the source (HF LFS metadata)
rows: 494121                        # parquet rows == independent JSONL line count
parquet_sha256: 8cf635113b8a539a9a62ec33c93dc72d7e81620f72c32aeb6564ecadcc4b1cac
parquet_bytes: 55615024
manifest_path: artifacts/manifests/source_manifest.json
license: >
  VERIFIED 2026-09-30: no dataset license is stated on the project site, the HF card or HF API metadata.
  MIT covers the GitHub code only. Maintainers say they cannot assign a license and that academic,
  non-commercial use is permitted (HF discussion #1). Our use: non-commercial portfolio/research,
  cite the paper, do not redistribute raw data or review text.
raw_text_published_in_repo: no
```

## Stage 1 measurements (real, official file, 2026-09-30)

| Measure | Value |
|---|---|
| Rows | 494,121 |
| Time range | 2001-02-04 to 2023-09-11 (snapshot is about Sept 2023) |
| Nulls in any of the 10 columns | 0 |
| Empty (whitespace-only) text | 249 |
| Distinct products (`parent_asin`) | 60,274 |
| Distinct users | 461,656 |
| Reviews per product (mean) | about 8.2 — **query groups may be small; check in Stage 2** |
| helpful_vote range / rating range | 0–7,326 / 1.0–5.0 |

## Feasibility measurements (real, 2026-09-30, scratch copy, not yet the official split)

| Measure | Value |
|---|---|
| Rows / years | 494,121 / 2001–2023 |
| helpful_vote ≥1 / ≥3 / ≥10 | 28.2% / 8.7% / 1.9% |
| Verified purchase | 90.1% |
| Ratings 1–5 | 69,564 / 28,895 / 36,949 / 57,000 / 301,713 |
| Words per review | median 20, mean 35, p90 80; 249 empty |
| Share with ≥1 vote by year | 2010 ~65%; 2015 ~29%; 2020 ~24%; 2023 ~19% |

## Quality and split audit (Stage 2, after cleaning; source `artifacts/manifests/clean_manifest.json`)

| Split | Date range (UTC) | Reviews | Half-year groups ≥5 (counts only, before cleaning) | Notes |
|---|---|---:|---:|---|
| Excluded (before train_start) | to 2014-12-31 | 36,713 | — | ADR-010 |
| Train | 2015-01-01 to 2020-06-30 | 289,187 | — | — |
| Validation | 2020-07-01 to 2020-12-31 | 40,087 | 1,914 | 229 rows flagged text-seen-in-train |
| Test (sealed) | 2021-01-01 to 2021-12-31 | 68,463 | 3,070 | 344 rows flagged; **labels not inspected** |
| Stream (drift replay only) | 2022-01-01 onward | 54,540 | — | unlabeled |

Rows in 494,121 → out 488,990 (5,131 exact duplicates removed, ADR-008). Eligible-group counts with votes and coverage come from the Stage 3 label builder. Full tables: `reports/audit/data_quality.md`.

## Stage 2 decisions (ADR-008 to ADR-014; all frozen 2026-09-30)

| ID | Decision | Evidence | Reverse only if |
|---|---|---|---|
| ADR-008 | Remove exact duplicates on (user_id, parent_asin, timestamp_ms) | 5,131 rows (1.04%); in development rows 100% identical text and votes | never needed |
| ADR-009 | Flag validation/test reviews whose normalized text (≥20 chars) is in train; exclude from evaluation | 344 test, 229 validation after cleaning | Stage 4 finds near-duplicate leakage |
| ADR-010 | Training starts 2015-01-01 | EDA-01: steep vote-rate decline before 2014, flat 27–33% over 2015–2019 | Stage 6 ablation shows pre-2015 rows help |
| ADR-011 | Query group = (parent_asin, calendar half-year), min size 5 | EDA-06: 62–65% of reviews covered; 1,914 validation / 3,070 test groups ≥5 | Too few eligible test groups (decide before the freeze, not after) |
| ADR-012 | Relevance = vote bins 0 / 1 / 2–4 / 5+ → grades 0–3 | EDA-04: 70.0 / 14.9 / 9.9 / 5.3% of dev rows | A degenerate Stage 3 result, before the freeze |
| ADR-013 | Maturity 18 months at snapshot 2023-09-11 | Convention only; one snapshot cannot show saturation | — (sensitivity: H1 vs H2 2021 in Stage 8) |
| ADR-014 | Longest-first is a mandatory baseline; **expectation recorded before any model:** it will be hard to beat | EDA-02: share with ≥1 vote 14% (1–5 words) to 64% (101+ words). **Confirmed in Stage 3** (best baseline, NDCG@3 0.5991) | — |
| ADR-015 | Tune the TF-IDF baseline until its best setting is interior to the grid (alphas 3–100); keep B01 on record | B01 best alpha sat at the grid edge; B02 best alpha 30, difference to longest-first no longer significant | — |
| ADR-016 | Baseline "rating-extremity" replaced by "lowest-rating-first" | EDA-05: 1★ 38.2% vs 5★ 28.1% with ≥1 vote (extremity would tie 1★ and 5★) | — |
| ADR-017 | Calendar features are month-of-year and hour (UTC) only; no year or raw timestamp. `prior_reviews` and calendar are ablated in Stage 5 as age/exposure proxies | Review age at the snapshot must not be recoverable (problem contract, leakage rule 3) | Ablation shows they matter and a leak-free reason exists |
| ADR-018 | The TF-IDF score enters the ranker as a stacked feature built out-of-fold (5 folds split by product) on train; the full-train TF-IDF model scores validation | An in-sample TF-IDF score would carry the training label into the ranker | — |
| ADR-019 | Exposure/age proxies (`prior_reviews`, month, hour) are **excluded from the primary candidate models**; the all-features model is reported only for context. TF-IDF stacking is used only if its paired NDCG@3 gain over content-only has a CI lower bound above 0 | Run R01 (all features): removing `prior_reviews` + calendar cost −0.0177 [−0.0260, −0.0095] NDCG@3. `prior_reviews` encodes how early in the window a review was posted (EDA-03 first-mover/exposure effect), so a ranker using it partly learns position bias instead of review quality. **Decision made after seeing the validation ablation in R01; it is conservative (lowers the apparent score)** | — |
| ADR-020 | Oldest-first added as a baseline (run B03) | Stage 3 had only newest-first, the wrong direction per EDA-03; a ranker can exploit earliness, so that shortcut must be measured | — |
| ADR-021 | Primary candidate = content-only LightGBM LambdaRank (20 features, num_leaves 31, min_data_in_leaf 200, 214 iterations). Text models in Stage 6 enter as an added feature or standalone and are kept only if their paired NDCG@3 gain over this model has a CI lower bound above 0 (same rule as TF-IDF) | R02: TF-IDF stacking −0.0014 [−0.0073, +0.0041] so the simpler model stays | A later stage shows a model passing the rule |
| ADR-022 | Zero-shot sentiment probabilities (FinBERT, DistilBERT SST-2) are not used as ranker features | Stage 6b paired NDCG@3 gains over the primary: +0.0000 [−0.0058, +0.0063] (FinBERT), −0.0006 [−0.0054, +0.0046] (SST-2), −0.0023 [−0.0088, +0.0044] (both); rule requires a CI lower bound above 0 | A fine-tuned variant passes the rule |
| ADR-023 | Fine-tuning protocol (6c): regression on log1p(votes), train split only, 60,000 randomly sampled rows (seed 42; **revised from 100,000 on 2026-10-01, before any result existed**, because attempt 1 was killed at the 2-hour background limit with no output), 2 epochs, lr 2e-5, batch 32, max length 128, fp16 autocast, no validation-based early stopping; same protocol for DistilBERT and FinBERT | Throughput benchmark about 65 rows/s on an idle machine; 60,000 rows × 2 epochs is about 31 min (DistilBERT) and about 60 min (FinBERT). Attempt 1 (100k rows, both models in one background job with output piped to `tail`) produced no files: it ran concurrently with CPU-heavy work (tokenization is on the CPU) and hit the 2-hour limit; progress was invisible because of the pipe. Attempt 2 runs one model per job with a live log (`reports/text_models/ft_<model>.log`) and no other heavy jobs | — |
| ADR-024 | Serving contract: `POST /rank` takes one product page's new reviews plus an optional `product_history` (n_reviews, mean_rating, mean_words of EARLIER reviews); returns scores and ranks. Request models forbid unknown fields, so votes cannot be sent. Only content-only models are served (no TF-IDF stacking, no exposure/age features). Ties keep input order | As-of features need product history; API/batch parity is tested; ADR-019/021 | A different primary model is frozen |
| ADR-025 | No transformer in the final system: fine-tuned DistilBERT and FinBERT (60k rows, 2 epochs) are rejected standalone (0.5816, 0.5840 vs longest-first 0.5991) and as cross-fitted blends (+0.0013 [−0.0032, +0.0057], +0.0023 [−0.0022, +0.0068]). Primary stays the content-only LightGBM; Stage 6 is closed | Pre-stated rule (ADR-021): keep only if the paired CI lower bound is above 0 | A larger/longer fine-tune or a different task formulation is run as a NEW version with its own validation, before a new freeze |

## EDA findings (Stage 2; development rows only; details and plots in `reports/eda/eda_findings.md`)

| ID | Finding | Decision it changed |
|---|---|---|
| EDA-01 | Share with ≥1 vote: 64.9% (2010), 44.9% (2013), 33.7% (2014), then 28.9 / 31.2 / 32.6 / 28.6 / 27.4% (2015–2019), 23.7% (2020). My earlier "steady decline" impression was wrong; the effect is strong before 2015 and weak after | Train from 2015-01-01 (ADR-010); compare within groups (ADR-011) |
| EDA-02 | Length is the strongest simple signal: ≥1 vote 14.0% (1–5 words) → 63.9% (101+ words); mean votes 0.27 → 6.41 | Longest-first baseline mandatory (ADR-014) |
| EDA-03 | Order among a product's reviews: 36.5% (1st) → 34.0% (6–10) → 25.6% (11+); the drop at 11+ is entangled with time and popularity | As-of order allowed only with an ablation |
| EDA-04 | 70.0% zero votes; top 1% of reviews hold 40.0% of votes; median top-review share of a product's votes 42.9% | Graded relevance bins (ADR-012) |
| EDA-05 | 1★ 38.2% vs 5★ 28.1% with ≥1 vote; verified+image 52.3% vs verified no image 29.0% | Image presence is a candidate feature; rating extremity is not assumed to help |
| EDA-06 | Half-year groups ≥5 cover 62–65% of reviews (validation 1,914; test 3,070 groups) | Evaluation unit (ADR-011) |

**Disclosure:** the feasibility check printed the share of reviews with ≥1 vote for every year (2021: 22.1%, 2022: 24.9%, 2023: 19.2%) before the split existed. That was an aggregate descriptive rate; no model used it and no row-level test labels have been inspected.

## EDA finding (copy)

```text
EDA-ID/date:
Question:
Split and sample:
Plot/table path, numerator/denominator:
Observation:
Alternative explanation / artifact:
Decision changed or no change:
Validation check to challenge conclusion:
```

## Label-design ADR — COMPLETED 2026-09-30 (LABEL-v1, frozen)

```text
Query group: (parent_asin, calendar half-year). Minimum size 5. NDCG is defined only if at least one review in the group has >=1 vote; coverage is reported with every metric.
Why half-year: covers 62-65% of reviews (validation 1,914 / test 3,070 groups >=5, counts only). Quarter covers 52-55%, month 33-35%, year about 70% but mixes wider ages. Evidence: EDA-06.
Maturity: M = 18 months at snapshot 2023-09-11. A convention (one snapshot cannot show saturation). Excludes no row of train/validation/test (latest test review, 2021-12-31, is about 20 months old); forbids labeling reviews after 2022-03-11. Sensitivity: H1 vs H2 2021 in Stage 8.
Grades: votes 0 -> 0, 1 -> 1, 2-4 -> 2, 5+ -> 3. Development shares 70.0 / 14.9 / 9.9 / 5.3 %. NDCG gain 2^rel - 1.
Excluded groups: size < 5; no voted review. Dev estimate: about 73% of half-year groups >=5 had >=1 voted review (2020 windows); test coverage unknown until the sealed evaluation.
Known biases left: position/exposure (no impression logs); up to 6 months of age difference inside a group; vote competition changes over years; popular products dominate counts.
Sensitivity analyses (all reported): binary >=1 vote; require >=2 voted reviews; min size 3 and 10; quarter windows.
Tests that pin this definition: NOT yet written. tests/test_labels.py is part of Stage 3 and must exist before any model result.
```

Template for any later revision (a revision before the freeze needs a new ADR; after the freeze it needs a new sealed test):

```text
LABEL-ID/date:
Query group definition (product × window):
Window length and why (evidence: EDA-ID):
Maturity months M and why (vote-age curve path):
Relevance grades and vote-quantile cutoffs:
Groups excluded (no positive / too small) and coverage after exclusion:
Known biases left (position/exposure, age within window):
Tests that pin this definition:
```

## Model experiment (copy per run)

```yaml
run_id: EXP-YYYYMMDD-001
timestamp_utc:
question_and_hypothesis:
source_manifest_sha256:
git_commit:
split_manifest_sha256:
label_id:
feature_set_version:
text_encoder: none | tfidf | finbert-zeroshot | general-encoder | finetuned:<name>
model: heuristic | tfidf-linear | lightgbm-lambdarank | transformer
parameters_path:
seed:
compute_environment:          # CPU/GPU actually used
train_seconds:
prediction_artifact:
metrics_json:
validation:
  groups_evaluated:
  reviews_evaluated:
  coverage:
  ndcg_at_3:
  ndcg_at_5:
  precision_at_3:
  ci_method: product-clustered bootstrap, B=
  ci_lower:
  ci_upper:
  paired_diff_vs_best_baseline:
  paired_ci:
decision: keep | reject | investigate
reason_and_rejected_option:
limitations:
```

**Comparison table** (identical validation rows and groups; no invented values):

| Run | Model | NDCG@3 [CI] | NDCG@5 | P@3 | Train s | Decision |
|---|---|---|---:|---:|---:|---|
| EXP-20260930-B02 | Random (seeded) | 0.3282 [0.3114, 0.3436] | 0.4257 | 0.3608 | 0 | reference |
| EXP-20260930-B02 | Newest-first | 0.2319 [0.2162, 0.2465] | 0.3308 | 0.2821 | 0 | reject (below random) |
| EXP-20260930-B02 | Longest-first | **0.5991 [0.5819, 0.6172]** | 0.6625 | 0.5254 | 0 | **keep: the bar to beat** |
| EXP-20260930-B02 | Lowest-rating-first (EDA-05; replaces "rating-extremity") | 0.3919 [0.3735, 0.4103] | 0.4810 | 0.4127 | 0 | keep as weak baseline |
| EXP-20260930-B02 | TF-IDF Ridge (alpha 30) | 0.5897 [0.5715, 0.6083] | 0.6559 | 0.5237 | 25.8 | keep; not better than longest-first (paired −0.0094 [−0.0209, +0.0016]) |
| EXP-20260930-B03 | Oldest-first (added ADR-020) | 0.4774 [0.4576, 0.4963] | 0.5571 | 0.4540 | 0 | keep as exposure baseline (below longest-first by 0.1217) |
| EXP-20260930-R02 | LightGBM LambdaRank, all 23 features (includes exposure/age proxies) | 0.6748 [0.6584, 0.6909] | 0.7310 | 0.5734 | 1.7 | context only (ADR-019) |
| EXP-20260930-R02 | **LightGBM LambdaRank, content-only (20 features) = primary candidate** | **0.6571 [0.6403, 0.6736]** | 0.7115 | 0.5649 | 1.6 | **keep: +0.0580 [+0.0451, +0.0709] over longest-first** |
| EXP-20260930-R02 | LightGBM, content + stacked TF-IDF score | 0.6557 [0.6392, 0.6725] | 0.7105 | 0.5625 | 1.2 | reject by rule (paired −0.0014 [−0.0073, +0.0041]) |
| EXP-… | + FinBERT zero-shot features | — | — | — | — | — |
| EXP-… | + general encoder features | — | — | — | — | — |
| EXP-… | Fine-tuned transformer | — | — | — | — | — |

## Stage 3 results — baseline ladder (validation only; sealed test NOT loaded)

Runs: `EXP-20260930-B01` (TF-IDF alphas 1, 3, 10) and `EXP-20260930-B02` (alphas 3, 10, 30, 100). Artifacts: `reports/baselines/baselines.md`, `baselines.json` (B02) and `baselines_B01.md/.json`. Inputs: `data/processed/reviews_clean.parquet` (sha256 in `baselines.json`), `configs/eval.yaml`, `configs/splits.yaml`. Git commit: none yet (no commits). Seeds: ties 42, bootstrap 42 (2000 replicates, 95%, clustered by product). Environment: Python 3.12.10, scikit-learn, pandas, numpy (versions in JSON).

- **Evaluated:** validation 2020-07-01..2020-12-31; **1,373 groups** (one per product, because validation is one half-year), 21,500 reviews; 1,883 size-ok groups, 72.9% of them had a voted review (coverage).
- **Result:** longest-first NDCG@3 **0.5991 [0.5819, 0.6172]**; TF-IDF Ridge 0.5897 [0.5715, 0.6083]; lowest-rating-first 0.3919; random 0.3282; newest-first 0.2319.
- **Paired, NDCG@3:** TF-IDF − longest-first = −0.0094 [−0.0209, +0.0016] (CI includes zero); longest-first − random = +0.2709 [+0.2499, +0.2925].
- **ADR-014 confirmed:** longest-first was recorded as hard to beat before any model ran, and it is the best baseline.
- **B01 → B02 (ADR-015):** in B01 the best TF-IDF alpha (10) sat at the edge of the grid and TF-IDF looked significantly worse than longest-first (−0.0157 [−0.0279, −0.0036]). The grid was extended; best alpha is now 30 (interior) and the difference is not significant. Alpha is selected on validation, which slightly favors the baseline.
- **Sensitivity (all models keep the same ordering):** min size 3 → 2,080 groups, longest-first 0.6803; min size 10 → 623 groups, 0.5042; two voted reviews → 1,016 groups, 0.6017; quarter windows → 1,305 groups, 0.6241; binary grade → 1,373 groups, 0.6549. **NDCG values cannot be compared across group definitions** (random scores 0.2072 at size 10 vs 0.4705 at size 3).
- **Limits:** one half-year of validation; baselines only, not the project claim. The learned model must beat **longest-first** on identical groups with a paired interval above zero.
- **Tests that pin this stage:** `tests/test_labels.py`, `test_bootstrap.py`, `test_evaluate.py`, `test_baselines.py` (71 tests passing in total).

## Stage 4 results — features (sealed test and stream NOT loaded)

Artifacts: `reports/features/feature_report.md` and `.json`, `data/processed/features_dev.parquet` (excluded + train + validation rows; excluded pre-2015 rows are loaded only as history for as-of aggregates). Code: `src/reviewrank/features.py`; tests: `tests/test_features.py`.

- **23 features, 5 groups:** length (7), style (5), media_meta (4: image_count, has_image, rating, verified), asof_product (5), calendar (2: month, hour).
- **As-of rule, tested:** changing votes never changes a feature; changing a later review never changes an earlier review's features; same-timestamp reviews do not see each other; shuffling input rows does not change features; no output column is named like a vote.
- **Missing:** 12.07% of train rows have no `prior_*` features (first review of a product) by design; other features under 0.2%.
- **Descriptive Spearman with votes (train):** chars +0.316, words +0.313, sentences +0.281, words_vs_prior_mean +0.232, digits +0.195, title_chars +0.173, image_count +0.114, prior_reviews −0.111, rating −0.067, calendar about 0. `unique_word_ratio` −0.281 is mostly a length artifact (longer texts repeat words). Not evidence a feature helps a ranker.

## Stage 5 results — LightGBM LambdaRank (validation only; sealed test NOT loaded)

Runs: R01 (all features; kept as `reports/ranker/ranker_R01.*`), **R02 (final design)** and baseline run B03 (adds oldest-first; B02 kept as `reports/baselines/baselines_B02.*`). Artifacts: `reports/ranker/ranker.md/.json`, `artifacts/manifests/primary_model.json`, model files in `artifacts/models/` (git-ignored; sha256 in `ranker.json`). Git commit: none yet. Same 1,373 validation groups; product-clustered bootstrap (2000 replicates, 95%); ties seed 42.

- **Baseline update (B03):** oldest-first NDCG@3 0.4774 [0.4576, 0.4963], well above random (0.3282) but below longest-first (−0.1217 [−0.1474, −0.0962]). Earliness alone is worth about +0.149 over random: the exposure advantage is real. **Longest-first (0.5991) remains the best baseline.**
- **Primary candidate by pre-stated rule:** content-only (TF-IDF stacking gain −0.0014 [−0.0073, +0.0041], so not used). NDCG@3 **0.6571 [0.6403, 0.6736]**, NDCG@5 0.7115, Precision@3 0.5649.
- **Paired, NDCG@3:** primary − longest-first **+0.0580 [+0.0451, +0.0709]**; primary − oldest-first +0.1797 [+0.1556, +0.2035]; primary − TF-IDF Ridge +0.0674 [+0.0532, +0.0817].
- **Exposure proxies (excluded by design):** the all-features model scores 0.6748, +0.0177 [+0.0101, +0.0251] over the primary; adding the proxies back to the primary gives +0.0152 [+0.0076, +0.0232]. That gain is the position-bias advantage ADR-019 removes.
- **Ablations from the primary (drop → change in NDCG@3):** media_meta (image_count, has_image, rating, verified) −0.0269 [−0.0375, −0.0168]; asof_product −0.0227 [−0.0314, −0.0140]; length −0.0111 [−0.0185, −0.0036]; style −0.0079 [−0.0139, −0.0017]; words_vs_prior_mean alone −0.0071 [−0.0144, +0.0002]. Group removals understate importance where features are redundant (words and chars appear in the length group and in words_vs_prior_mean).
- **Top gain features:** words_vs_prior_mean 41.2%, chars 22.9%, words 7.7%, rating_minus_prior_mean 5.4%, image_count 4.9%, verified 4.0%.
- **Grid:** 6 configs per feature set; spread across configs is small (content-only 0.6489 to 0.6571). Training takes about 1–2 seconds per model on CPU.
- **Sensitivity:** the primary beats longest-first under all five alternative group definitions (for example min size 10: 0.5785 vs 0.5042; quarter windows: 0.6808 vs 0.6241). Point estimates only, no intervals; values are not comparable across definitions.
- **Limits:** validation only, one half-year; hyperparameters and early stopping were selected on validation, so the figure is optimistic until the sealed test; nothing is frozen yet (Stage 7).
- **Stage 6 rule:** FinBERT/encoder/transformer outputs are kept only if their paired NDCG@3 gain over the primary has a CI lower bound above 0.

## Stage 6a/6b results — zero-shot sentiment features (validation only; sealed test NOT loaded)

Run `EXP-20260930-T01`; artifacts `reports/text_models/stage6b.md/.json` and `sent_finbert_manifest.json`, `sent_sst2_manifest.json`; the scores are in `data/processed/sent_*.parquet` (git-ignored). Models: `ProsusAI/finbert` (revision `4556d13015211d73dccd3fdd39d39232506f3e43`) and `distilbert-base-uncased-finetuned-sst-2-english` (revision `714eb0fa89d2f80546fda750413ed43d93601a13`); fp16 autocast, max length 128, input = title + text. A 2,000-review benchmark showed fp16 agrees with fp32 on 100% of labels (max probability difference 0.0012) at about 9x the speed. Scoring 329,274 rows took 1,172 s (FinBERT, 281 rows/s) and 609 s (SST-2, 541 rows/s).

| Model | NDCG@3 [95% CI] | Paired gain vs primary [95% CI] | Verdict by rule |
|---|---|---|---|
| content-only (primary) | 0.6571 [0.6403, 0.6736] | — | reference |
| + FinBERT probabilities | 0.6571 [0.6410, 0.6737] | +0.0000 [−0.0058, +0.0063] | reject |
| + SST-2 probabilities | 0.6565 [0.6400, 0.6726] | −0.0006 [−0.0054, +0.0046] | reject |
| + both | 0.6548 [0.6377, 0.6705] | −0.0023 [−0.0088, +0.0044] | reject |

- **Polarity vs star rating** (37,260 validation reviews with rating != 3; rating used as a noisy sentiment label): FinBERT 87.6% accurate, Spearman(net score, rating) +0.593; SST-2 86.3%, +0.679.
- **Domain-mismatch evidence:** FinBERT's top class is *neutral* for 80.8% of validation reviews; its positive/negative order is still informative. FinBERT and SST-2 net scores correlate at +0.733.
- **Votes:** SST-2 negative probability has Spearman +0.157 with votes on train (negative reviews draw more votes, as the rating EDA showed), but adds nothing beyond the existing features.
- **Decision:** ADR-022. Selection among the 24 grid fits is on validation (optimistic); even so none passes the rule.

## Stage 6c/6d results — fine-tuned transformers (validation only; sealed test NOT loaded) — final

Protocol (ADR-023): regression on log1p(votes), train split only, 60,000 sampled rows (seed 42), 2 epochs, lr 2e-5, batch 32, max length 128, fp16 autocast + GradScaler, no validation-based early stopping. Manifest `reports/text_models/ft_distilbert_manifest.json`; backbone `distilbert-base-uncased` (revision `12040accade4e8a0f71eabdb258fecc2e7e948be`). Evaluation run `EXP-20260930-T02` (`reports/text_models/stage6d.md/.json`); blend weight chosen by 2-fold cross-fitting over validation products.

- **DistilBERT training:** 1,477 s for 2 epochs (81 rows/s); mean train loss 0.362 (epoch 1) → 0.327 (epoch 2); scoring 40,087 validation rows took 78.6 s.
- **Standalone:** NDCG@3 **0.5816 [0.5635, 0.5992]** — below longest-first (0.5991) and TF-IDF Ridge (0.5897); paired vs primary −0.0755 [−0.0903, −0.0608].
- **Blend with the primary (cross-fitted):** 0.6584 [0.6419, 0.6747]; paired gain **+0.0013 [−0.0032, +0.0057]** → rejected by rule (CI lower bound not above 0). The weight search always chose the smallest weight (0.1): validation NDCG@3 falls steadily as the transformer's weight grows (0.6584 at 0.1 to 0.5965 at 0.9).
- **Reading:** a transformer trained on 60,000 reviews and the first 128 tokens, reading text alone, did not learn more than the length and metadata features capture. Possible reasons (not tested): it cannot see length beyond the truncation point, it has no metadata, 60,000 rows is small, and the regression target is noisy and zero-inflated.
- **Operational lessons:** (1) attempt 1 (100k rows, chained jobs, output piped through `tail`) was killed at the 2-hour limit with no files; (2) FinBERT attempt 2 stopped logging 2,292 s into training and hung until the limit — the laptop went to sleep overnight; (3) fix: a process-level `SetThreadExecutionState` keep-awake request in the script (no system setting changed). Background jobs cap at 2 hours.
- **FinBERT fine-tune (attempt 3, same protocol, keep-awake on):** backbone `ProsusAI/finbert` (revision `4556d13015211d73dccd3fdd39d39232506f3e43`); 2,945 s for 2 epochs (40.7 rows/s, about half DistilBERT's speed); mean train loss 0.364 → 0.327; scoring 40,087 validation rows 158 s. Standalone NDCG@3 **0.5840 [0.5668, 0.6019]** (DistilBERT 0.5816: no advantage from finance pretraining after fine-tuning); paired vs primary −0.0730 [−0.0884, −0.0574]. Cross-fitted blend with the primary 0.6594 [0.6429, 0.6756], paired gain **+0.0023 [−0.0022, +0.0068]** → rejected by rule; blend weight again 0.1 in both folds.
- **Decision (ADR-025):** neither fine-tuned transformer is used. The primary stays the content-only LightGBM LambdaRank. Combined Stage 6 conclusion: every text-model addition (stacked TF-IDF, zero-shot FinBERT, zero-shot SST-2, fine-tuned DistilBERT, fine-tuned FinBERT, blends) has a paired NDCG@3 interval that includes zero; engineered length, media/metadata and as-of product features carry the signal.

## Stage 8 results — SEALED TEST (opened once, 2026-10-01, freeze FREEZE-20260930-01)

Command `python scripts/run_stage8_sealed_test.py` (all freeze hashes matched; `artifacts/manifests/test_opened.json` written afterwards with the result file's sha256). Artifacts: `reports/test/test.md`, `reports/test/test.json`, stdout log `reports/test_run.log`. Test = reviews posted 2021-01-01 to 2021-12-31 (68,463 reviews after cleaning). Groups: **2,164** eligible (parent_asin × half-year, ≥5 reviews, ≥1 voted review) across **1,696 products**; 3,026 size-ok groups, coverage 71.5%. Intervals: product-clustered bootstrap, 2000 replicates, 95%, paired. Git commit: none yet.

| Model | NDCG@3 [95% CI] | NDCG@5 | Precision@3 |
|---|---|---|---|
| Random (seeded) | 0.3398 [0.3254, 0.3538] | 0.4292 | 0.3561 |
| Oldest-first | 0.4442 [0.4285, 0.4602] | 0.5254 | 0.4250 |
| Longest-first (primary comparator) | 0.6223 [0.6083, 0.6364] | 0.6812 | 0.5382 |
| TF-IDF + Ridge | 0.6124 [0.5987, 0.6272] | 0.6701 | 0.5325 |
| **Frozen LightGBM LambdaRank, content-only** | **0.6558 [0.6424, 0.6699]** | **0.7131** | **0.5578** |

- **Paired NDCG@3 differences:** frozen − longest-first **+0.0335 [+0.0236, +0.0429]**; frozen − TF-IDF +0.0434 [+0.0324, +0.0544]; frozen − oldest-first +0.2116 [+0.1934, +0.2303]; frozen − random +0.3160 [+0.2982, +0.3341]; longest-first − TF-IDF +0.0099 [+0.0003, +0.0194] (on the test the length rule is slightly but significantly better than TF-IDF; on validation they were indistinguishable).
- **Validation vs test (for the model card):** the frozen model scored 0.6571 on validation and 0.6558 on the test, so its absolute score held. The best baseline moved from 0.5991 to 0.6223, so the margin shrank, mainly because the baseline was weaker on validation: +0.0580 on validation (one half-year) vs **+0.0335 on the sealed test** (relative gain 5.4%). Validation overstated the margin.
- **Segments (all favor the frozen model):** groups of 5–6 reviews +0.0344 [+0.0161, +0.0526]; 7–9 +0.0388 [+0.0168, +0.0592]; 10+ +0.0307 [+0.0167, +0.0442]; group mean rating below 4 +0.0381 [+0.0227, +0.0531]; 4 or above +0.0300 [+0.0178, +0.0432]; window 2021-H1 +0.0356 [+0.0225, +0.0476]; 2021-H2 +0.0308 [+0.0160, +0.0448]. (Another maturity check: H2 reviews are younger at the snapshot and the gain is similar.)
- **Sensitivity (mean NDCG@3; frozen vs longest-first):** min size 3 (3,272 groups) 0.7186 vs 0.6901; min size 10 (1,060) 0.5881 vs 0.5574; two voted reviews (1,611) 0.6628 vs 0.6318; quarter windows (2,040) 0.6822 vs 0.6505; binary grade (2,164) 0.6998 vs 0.6719. Values are not comparable across definitions.
- **Bias analysis (top-3 picks, mean over groups):** frozen model 74.9 words, 0.44 images per review (mean image count, not a share), rating 3.60, 90.4% verified, 2.71 votes; longest-first 81.5 words, 0.297 images; the most-voted reviews (oracle) 56.0 words, 0.246 images, rating 3.73, 3.94 votes; all reviews 34.6 words, 0.128 images. **The model over-prefers images and length relative to what readers vote for** (same pattern as validation).
- **Caveats:** retrospective evaluation on a helpful-vote proxy, one product category, one year; no online experiment; scores are not a measure of truthfulness. The test was opened once; no change is allowed for this freeze.

## Stage 8 pipeline check — dry run on VALIDATION (historical; done before the test was opened)

`python scripts/run_stage8_sealed_test.py --split validation` (artifacts `reports/test_dryrun/test.md/.json`). Purpose: show that the evaluation pipeline used for the test (features built in one pass from all earlier rows' non-label columns, saved model files, baselines, bootstrap) is equivalent to the development pipeline.

- **Reproduction (all identical to Stages 3 and 5):** random 0.3282, oldest-first 0.4774, longest-first 0.5991 [0.5819, 0.6172], TF-IDF Ridge 0.5897, candidate primary 0.6571 [0.6403, 0.6736]; paired primary − longest-first +0.0580 [+0.0451, +0.0709].
- **Segments (validation, group level):** the candidate beats longest-first in every segment: groups of 5–6 reviews +0.0420 [+0.0198, +0.0638]; 7–9 +0.0469; 10+ +0.0743 [+0.0530, +0.0956]; group mean rating below 4 +0.0704; 4 or above +0.0493.
- **Bias analysis — what the top-3 look like (mean over groups):** candidate picks average 72.7 words, 0.431 images per review (mean image count, not a share), rating 3.70, 91.7% verified, 2.70 votes; longest-first picks average 80.2 words, 0.268 images; the oracle (truly most-voted) picks average 54.3 words, 0.266 images, rating 3.85, 4.13 votes; all reviews average 36.1 words, 0.136 images. **The candidate over-prefers images and length relative to what readers actually vote for**; this is a limitation to state in the model card.
- **Guards:** test mode requires `freeze.json` with matching hashes for the model, manifest, TF-IDF baseline, both configs and every source file; refuses a second run; reads test labels only in test mode.

## Stage 10 results — offline drift replay (unlabeled stream; candidate primary model)

Artifacts: `reports/monitoring/drift.md`, `drift.json`, `drift.png`; code `src/reviewrank/monitor.py` (PSI), `scripts/run_stage10_drift_replay.py`; 10 tests in `tests/test_monitor.py`. The script never reads `helpful_vote`; earlier rows, including the sealed test year, are loaded only as non-label history for the stream's as-of features. Stream: 54,540 reviews, 2022-01 to 2023-09. Reference: validation period (40,087 reviews). Git commit: none yet.

- **Result:** whole-stream PSI vs the validation period is small for every monitored column and for the model score (max 0.057, prior_mean_rating). In the 16 months with ≥1,000 reviews the maximum PSI over all monitored columns is 0.218 (moderate; below the conventional 0.25). The only month above 0.25 is 2023-09 with 17 reviews (partial month); 2023-05 to 2023-08 have 210–705 reviews, so their PSI is noisy.
- **Meaning:** inputs and scores look stable relative to 2020-H2. This says nothing about whether ranking quality changed; that needs delayed votes (maturity rule, ADR-013).
- **Bug caught in my own code:** the first replay printed image_count PSI of exactly 0.000 in every month. Cause: for zero-inflated counts with more than 10 distinct values the reference quantile edges collapsed to one point, and `psi` returned 0 silently. Fixed (extra edges from the non-zero values), with a regression test (`test_zero_inflated_counts_with_many_distinct_values_are_not_silently_zero`). After the fix the image_count maximum is 0.438, in the 17-review month.
- **Thresholds** (0.1 / 0.25) are conventions, not laws.

## FinBERT notes — in progress (evidence only; template below)

- **Model:** `ProsusAI/finbert`, BERT-base encoder with a 3-class head, 109,484,547 parameters. Revision hash (Hub, read 2026-09-30): `4556d13015211d73dccd3fdd39d39232506f3e43` (last modified 2023-05-23).
- **Label order, verified 2026-09-30 by loading the model and reading its own config:** `id2label = {0: positive, 1: negative, 2: neutral}`. The v1 code assumed index 1 = negative; that was correct.
- **Smoke-test sample (3 sentences, not evidence):** "This product is great and works perfectly." → neutral 0.683, positive 0.306; "Terrible, it broke after two days." → negative 0.706; "It is a bottle of vitamins." → neutral 0.942. A clearly positive product review scored mostly neutral, a first hint of domain mismatch (FinBERT was trained on financial text).
- **Hardware:** forward and backward both ran on the Radeon 8060S (`reports/gpu/smoke_test.json`).
- **Measured (Stage 6b):** zero-shot polarity matches star-rating polarity 87.6% of the time (SST-2: 86.3%); top class is neutral for 80.8% of reviews; Spearman with rating +0.593 (SST-2 +0.679); adds no ranking signal over the primary (paired +0.0000 [−0.0058, +0.0063]). Why: sentiment largely duplicates rating, which the ranker already has, and helpfulness depends on detail and length more than on polarity.
- **Fine-tuned (Stage 6c/6d):** FinBERT fine-tuned on 60k reviews scores NDCG@3 0.5840 [0.5668, 0.6019], DistilBERT 0.5816 [0.5635, 0.5992]: no advantage from finance pretraining after fine-tuning; both below the longest-first rule (0.5991); cross-fitted blends with the primary +0.0023 [−0.0022, +0.0068] (FinBERT) and +0.0013 [−0.0032, +0.0057] (DistilBERT), both rejected. FinBERT trained at about half DistilBERT's speed (40.7 vs 81.2 rows/s).
- **FinBERT summary:** it is BERT-base pretrained on financial text; its head outputs positive/negative/neutral (order 0/1/2 from its config); on product reviews it labels 80.8% neutral zero-shot (domain mismatch) but its polarity still agrees with star ratings 87.6% of the time; neither zero-shot nor fine-tuned FinBERT improved the ranker; a likely reason is that sentiment duplicates the rating and that helpfulness depends on detail and length (untested).

## FinBERT notes (fill as learned; evidence only) — template

```text
Model id / revision hash:
Label order from model config (id2label) — verified how:
Input max length and truncation behavior:
Zero-shot score definition used:
Correlation of score with rating on validation:
Did it add ranking signal beyond TF-IDF? (paired CI):
Why it did or did not help (domain mismatch evidence):
```

## Model freeze — RECORDED 2026-10-01T14:18:27Z, BEFORE the sealed test was opened

```text
FREEZE-ID: FREEZE-20260930-01 (the id keeps the original planning date; frozen 2026-10-01T14:18:27Z UTC)
Source of truth: artifacts/manifests/freeze.json (committed). The test script re-checks every hash below and refuses to run on any mismatch.
Chosen model: content-only LightGBM LambdaRank, 20 features, num_leaves 31, min_data_in_leaf 200, 214 trees.
  model file artifacts/models/lgbm_primary.txt, sha256 4e282804b63a174ba536eac522e7058d17243f1c9af2457b3bd75569aaf9f4b9
  primary manifest sha256 11650d1d8cb3429c23621c76bbab1a72d4e6d2e48f3a221538ef6ee886a09522
  source files combined sha256 a3aa3878d87a5934bcd41973e3982b6fd953c7a443a7c0bc699d3b818fc4517e (16 files in src/reviewrank, each hash in freeze.json)
  evaluation script scripts/run_stage8_sealed_test.py sha256 cf0f4e6b32192f7d... (full value in freeze.json)
Features: words, chars, title_chars, title_words, avg_word_len, sentences, unique_word_ratio, exclaims, questions, digits, caps_ratio, is_empty_text, image_count, has_image, rating, verified, prior_mean_rating, rating_minus_prior_mean, prior_mean_words, words_vs_prior_mean. No exposure/age features (ADR-019), no TF-IDF score, no sentiment or transformer score (ADR-021/022/025).
Reason from validation: NDCG@3 0.6571 [0.6403, 0.6736]; paired +0.0580 [+0.0451, +0.0709] over longest-first (EXP-20260930-R02, B03, T01); rejected: stacked TF-IDF, zero-shot FinBERT/SST-2 features, fine-tuned DistilBERT/FinBERT blends.
Primary comparator: longest-first. Other baselines: random (seeded), oldest-first, TF-IDF Ridge (artifacts/models/tfidf_ridge_alpha30.joblib).
Metric: mean NDCG@3 over eligible groups (gain 2^rel-1, log2 discount), NDCG@5 and Precision@3 also; coverage reported.
Groups: (parent_asin, calendar half-year), size >= 5, >= 1 voted review, grades 0/1/2-4/5+ (LABEL-v1); text_seen_in_train rows excluded (ADR-009); maturity 18 months (ADR-013).
CI: product-clustered bootstrap, 2000 replicates, 95%, seed 42, paired against each comparator. Ties: seeded random permutation, seed 42.
Sensitivity (all reported): min size 3 and 10, two voted reviews, quarter windows, binary grade.
Test rows untouched at freeze time: yes (artifacts/manifests/test_opened.json did not exist; the dry run used validation only).
One-time evaluation command: python scripts/run_stage8_sealed_test.py   (writes test_opened.json when done)
Rules: report the result whatever it is, including a loss to the comparator; no tuning or model change after opening; a later change is a NEW version with a new sealed-evaluation design.
Deviations: none so far.
```

Template for any later revision:

```text
FREEZE-ID/date/time:
Chosen run, model, source hash, split hash, commit:
Reason from validation and rejected candidates:
Label ID and exact metric formula:
Baselines and tie rule/seed:
CI method and replicates:
Sealed test untouched? evidence:
One-time evaluation command:
Deviations and new version if reopened:
```

## Sealed test report (fill only after freeze)

| Model | Groups | Reviews | NDCG@3 [CI] | Paired diff vs best baseline [CI] | P@3 | Caveat |
|---|---:|---:|---|---|---|---|
| Random | — | — | — | — | — | — |
| Best baseline | — | — | — | — | — | — |
| Selected model | — | — | — | — | — | — |

Segments: rating (1–5), length bucket, verified vs not, group size. Report whether the selected model mostly prefers long or extreme reviews. If the CI of the paired difference spans zero, say so.

## Serving and monitoring replay

| Run/date | Schema failures | Feature shift (PSI) | Score shift | API/batch parity | Latency p50/p95 | Cold start | Action |
|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | — | — | — | — | — | — | — |

Replay uses unlabeled 2022–2023 reviews. It shows input/score drift, not accuracy decay. Any claim about performance over time needs labels and the maturity rule.

## AWS / cost record — status 2026-09-30: NOT DEPLOYED, nothing created, nothing spent

- **Terms read from official pages (2026-09-30):** Free plan = up to $200 credits ($100 immediately + up to $100 more), valid 6 months, **account closes on its own after 6 months or when credits run out**, no charge unless converted to a Paid plan (https://aws.amazon.com/free/). Lambda free tier = 1 million requests and 400,000 GB-seconds per month, not time-limited (https://aws.amazon.com/lambda/pricing/). ECR private storage $0.10 per GB-month, free 500 MB for one year for new ECR customers (https://aws.amazon.com/ecr/pricing/). Function URL charges: not stated on the pricing page; unverified.
- **Blockers:** AWS CLI 2.35.15 is installed but **no credentials/profile/region configured** (the repository owner must run `aws configure --profile reviewrank` personally; secrets must never go into chat or the repo). Docker Desktop daemon was not running, so no image has been built. The model is a candidate until the Stage 7 freeze.
- **Prepared:** `Dockerfile.lambda` (Lambda Web Adapter 1.1.0, port 8080, verified at https://github.com/awslabs/aws-lambda-web-adapter), `docs/deploy_aws.md` runbook (budget alarm first, reserved concurrency 2, teardown steps).
- **Deployment rule:** do not describe the project as deployed on AWS until the runbook is done and the live URL is checked; a Free-plan demo is temporary.

Template for when resources are created:

```text
Date, service, resource ID, purpose:
Free-tier terms as read on official page (URL, date):
Budget alarm set (amount, email):
Spend so far:
Teardown done:
```

## Decision record (copy)

```text
ADR-ID/date:
Question and alternatives:
Validity / feasibility:
Evidence: EDA-ID, EXP-ID, CI:
Choice and why:
Counterargument / tradeoff:
What evidence would reverse this:
```

### Initial rationales (update with evidence)

| Why / why not? | Provisional answer |
|---|---|
| Why rank instead of classify? | The product decision is which reviews to show first; ranking metrics match that decision. |
| Why not predict raw vote counts? | Votes depend on how long a review has existed and where it was shown; raw counts would reward age. |
| Why temporal split? | New reviews arrive after training; random splits leak time. |
| Why product-clustered bootstrap? | Reviews within one product are correlated; row-level resampling understates uncertainty. |
| Why compare FinBERT? | It is a finance-domain model; testing it on product text shows whether domain pretraining transfers. |
| Why serve and monitor? | Shows what happens after training; drift replay demonstrates the monitoring idea without claiming live performance. |