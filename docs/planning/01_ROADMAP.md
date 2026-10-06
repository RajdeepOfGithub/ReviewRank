# ReviewRank — Cold-Start Review Helpfulness Ranking

**Working brief · 2026-09-30 · status: planned; all results pending. "ReviewRank" is a working name.**

## One-sentence claim

Build a reproducible, time-honest pipeline that ranks **newly posted Amazon product reviews by how helpful they will turn out to be**, *before any votes exist*, compare it with simple baselines on later months, and serve it as a tested API with an offline drift replay.

**Why this problem:** a new review has zero votes, so "sort by helpful votes" cannot rank it (the cold-start problem). A text/metadata model can. This is a real ranking decision, not a classification exercise.

**What this project does NOT claim:** that a model measures truthfulness or purchase impact; that results reflect Amazon's live ranking; any revenue effect. Helpful votes are a *proxy* shaped by display position and exposure.

## Why this task (decision record)

Reviews give real text, real timestamps and a real (if imperfect) outcome, helpful votes. CFPB complaint text was evaluated as an alternative source and rejected: narratives are no longer in the bulk file or API.

## Data (verified 2026-09-30 unless marked)

- Source: McAuley Lab **Amazon Reviews 2023** (Hugging Face `McAuley-Lab/Amazon-Reviews-2023`, `raw/review_categories/*.jsonl`). Cite Hou et al., "Bridging Language and Items for Retrieval and Recommendation", ACL 2024. Terms: Amazon's terms apply **[re-verify at source before publishing anything]**; keep raw text out of the public repo.
- Primary category for development: `Health_and_Personal_Care` — 494,121 reviews, 2001–2023, 226,990,293 bytes.
- Fields: `asin, parent_asin, user_id, rating, title, text, images, timestamp (ms), helpful_vote, verified_purchase`.
- Measured: `helpful_vote>=1` 28.2%; `>=3` 8.7%; `>=10` 1.9%; verified 90.1%; 5-star ~61%; median 20 words.
- **Label-age confound (measured):** share of reviews with ≥1 vote falls from ~65% (2010) to ~29% (2015), ~24% (2020), ~19% (2023). Raw votes partly measure how long a review has existed.
- Rows per year: ≤2019: 293,037; 2020: 76,753; 2021: 69,202; 2022: 43,667; 2023 (partial): 11,462.

## Scope and evaluation contract (initial; frozen in Stage 0 before modeling)

| Item | Initial choice | Change rule |
|---|---|---|
| Unit | One review | — |
| Prediction moment | Review posting time. Only information available then | Anything known later is forbidden |
| Query group | Reviews of the same `parent_asin` in the same calendar half-year, at least 5 reviews **(frozen in Stage 2, ADR-011)** | Only before the freeze, with a new ADR |
| Relevance label | Grades 0–3 from absolute vote bins 0 / 1 / 2–4 / 5+, compared only within a group **(frozen, ADR-012)** | Only before the freeze, with a new ADR |
| Maturity rule | Reviews at least 18 months old at the 2023-09-11 snapshot are labeled **(frozen, ADR-013; a convention)** | Never loosen after seeing test |
| Split | **Temporal**: train 2015-01-01..2020-06-30 (ADR-010), validation 2020-07-01..2020-12-31, sealed test 2021 **(boundaries frozen in Stage 2)** | Changing it after modeling starts requires a new sealed test and a logged reason |
| 2022–2023 reviews | Unlabeled "stream" for drift replay only, never for scoring claims | — |
| Main metric | Mean **NDCG@3** over eligible query groups (groups with ≥1 vote-bearing review), with coverage reported; also NDCG@5, Precision@3 | Metric fixed at freeze |
| Uncertainty | Product-clustered bootstrap CI on metric and on paired difference vs each baseline | No significance claim if CI spans zero |
| Baselines | Newest-first, longest-first, rating-extremity, random (seeded), TF-IDF linear | Must be beaten at same groups and rows |
| Candidates | LightGBM LambdaRank on engineered features; + frozen-encoder features; fine-tuned transformer on a documented subsample | Complex model only if a documented error pattern justifies it |
| Text encoders compared | TF-IDF; zero-shot **FinBERT** sentiment (domain-mismatch test); general sentiment/sentence encoder; fine-tuned small transformer | FinBERT stays only if it earns its place; result reported either way |
| Delivery | Batch scorer, Dockerized FastAPI, offline drift replay, AWS free-tier deployment **only after** local reproducibility | "Deployed" only once verified live |

**Minimum credible finish (the core):** validated data + label-design ADR, temporal split with sealed test, baseline ladder, LightGBM ranker, one transformer experiment (incl. FinBERT), product-clustered CIs, error/bias analysis, served API with tests, drift replay, honest README/model card, verified evidence.

**Extensions only after the core is done:** train on one category and test on another (transfer check); probability calibration; SHAP/permutation importance with correlation caveat; LLM-judge sanity check on a small hand-labeled sample; DuckDB/Parquet scale run on a second, larger category.

## Stages and exit gates

| Stage | Work | Artifact / gate | What it shows |
|---|---|---|---|
| 0. Contract | Problem, label, split, metric, leakage rules, license | `docs/v2/problem_contract.md`, `configs/*.yaml`, tiny synthetic fixture; no data downloaded | Why cold start; why ranking not classification |
| 1. Acquire | Download category, record URL, bytes, sha256, rows, schema; Parquet conversion | Immutable source manifest; raw data out of Git | Source/version discipline |
| 2. Audit + EDA | Duplicates (same user/product/text), empty text, language, time coverage, vote-age curve, product concentration, near-duplicate/spam reviews, rating mix | Data-quality + EDA reports with ≥3 findings that changed a decision; **label-design ADR** | Finding → decision chain |
| 3. Baselines | Heuristic and TF-IDF baselines on validation | Baseline table with CIs | Simple model sets the bar |
| 4. Features | Text stats, rating extremity vs product-as-of mean, verified, images, calendar; as-of product aggregates only | Point-in-time feature table + leakage unit tests | Leakage prevention |
| 5. Ranker | LightGBM LambdaRank, recorded search, fixed seeds | Model comparison table with cost (time) | Why this model |
| 6. Text models | Frozen FinBERT/general encoder features; fine-tuned small transformer on subsample | Ablation table; FinBERT understanding notes | Domain mismatch, compute vs gain |
| 7. Select + freeze | Choose on validation only; write freeze record | Freeze record before test is opened | Protecting the test set |
| 8. Sealed test | One-time evaluation; segments (rating, length, verified, product size); paired CIs | `reports/test.md`, row-level predictions | Honest result, including losses |
| 9. Bias + error analysis | Does model just promote long or extreme reviews? position-bias caveat; worst errors reviewed | Error analysis report | Dive deep, limitations |
| 10. Serve + monitor | FastAPI, Docker, tests, parity, drift replay 2022–2023, AWS free-tier deploy with budget alarm | One-command demo; live URL only if verified; monitoring replay report | Production thinking |
| 11. Communicate | README, model card, evidence table | Independent reviewer can reproduce headline metrics | Every number defensible |

## Time and leakage rules

1. Features may use only what existed at posting time. Product-level aggregates (mean rating, review count) must be computed from reviews posted **before** the row's timestamp.
2. `helpful_vote` is never a feature. Neither is anything derived from it (including cohort ranks, vote-based product stats).
3. Review age at snapshot must not enter features or be recoverable (e.g., via raw timestamp interactions that let the model learn "older ⇒ more votes"). Calendar features are tested for this in validation ablations.
4. Same-user or near-duplicate reviews across the train/test boundary are checked; decision logged (drop, group-split, or keep).
5. Fit vocabularies, scalers, and encoders on train only.
6. The sealed test is opened once, after freeze. Any later change gets a new version and a new independent evaluation design.

## Evidence gate

A number is reportable only if its metric definition, query-group rule, evaluated rows/groups, baseline, split dates, CI, artifact path, run ID and commit are recorded in `03_METRICS_AND_DECISIONS.md`. Say "temporal backtest", "offline ranking metric", "helpful-vote proxy". Never say "improved conversion", "reduced fake reviews", or "deployed" unless that happened.

## Risks (known now)

- **Weak signal:** helpfulness from text is a modest-signal task; the honest result may be small gains. Plan for it; report it.
- **Compute:** GPU availability on this Windows machine is **not yet checked** (old notes describe a different Linux AMD box). Fallback: fine-tune on a documented subsample, or use free hosted GPUs; record what was actually used.
- **Position/exposure bias:** early or top-displayed reviews collect more votes regardless of quality; cannot be removed from this data, only acknowledged and partly controlled by grouping.
- **AWS free tier:** terms changed recently (new accounts: credits and a limited-duration free plan per third-party sources). **Verify on the official AWS page before deploying**; set a budget alarm first.
- **Licensing:** verify Amazon Reviews terms at source before any public artifact beyond code and aggregate metrics.

## Sources

- Amazon Reviews 2023: https://amazon-reviews-2023.github.io/ · Hugging Face card: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- Amazon Leadership Principles (16): https://www.amazon.jobs/content/en/our-workplace/leadership-principles
- Feasibility measurements: private notes, not in this repository.
