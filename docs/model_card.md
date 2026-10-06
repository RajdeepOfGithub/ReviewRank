# ReviewRank — Model card

**Status: frozen (FREEZE-20260930-01, 2026-10-01) and evaluated once on the sealed 2021 test year.** Validation figures below were used for selection and are optimistic about the margin; the sealed-test result is the headline.

## Headline result (sealed test, 2,164 groups across 1,696 products; product-clustered bootstrap 95% intervals)

| Model | NDCG@3 |
|---|---|
| Random | 0.3398 [0.3254, 0.3538] |
| Oldest-first | 0.4442 [0.4285, 0.4602] |
| Longest-first (best baseline) | 0.6223 [0.6083, 0.6364] |
| TF-IDF + Ridge | 0.6124 [0.5987, 0.6272] |
| **LightGBM LambdaRank, content-only (frozen)** | **0.6558 [0.6424, 0.6699]** |

Paired gain over longest-first: **+0.0335 [+0.0236, +0.0429]** (relative +5.4%); it is positive with an interval above 0 in every segment checked (+0.030 to +0.039) and in both halves of 2021, and the frozen model also scores higher than longest-first under all five alternative group definitions (point estimates only, no intervals). On validation the gain looked larger (+0.0580), mainly because the longest-first baseline was weaker in 2020-H2 (0.5991 vs 0.6223 on test); the model's own score barely changed (0.6571 validation, 0.6558 test). Only 35,022 of the 68,463 test reviews (51.2%) are in the 2,164 evaluated groups: 344 were removed for text overlap with train, and a group counts only if it has at least 5 reviews and at least one voted review (2,164 of 3,026 size-eligible groups, 71.5%). Retrospective evaluation on a helpful-vote proxy, one category, one year, one training seed. Evidence for every number: `docs/planning/03_METRICS_AND_DECISIONS.md` and `reports/`.

## Intended use

Rank newly posted product reviews on a single product page by their predicted helpful-vote outcome, using only information that exists when the review is posted (cold start). The evaluation ranks all reviews of a product within a half-year after the fact. Intended as a portfolio/research demonstration of an end-to-end ML workflow. Not intended for: judging a review's truthfulness, detecting fake or incentivized reviews, moderating or removing reviews, or any decision about a person.

## Model

- LightGBM gradient-boosted trees with the LambdaRank objective, trained to rank reviews inside (product, half-year) groups.
- 20 content features, computable from the review itself and the product's earlier reviews: text length and style statistics, title length, image count, star rating, verified-purchase flag, and the product's earlier-review summary (mean rating, mean words) with deviations from it. Votes are never an input. Exposure/age features (count of earlier reviews, month, hour) are **excluded by design** because they likely let a ranker learn the first-mover/position advantage (ADR-019). The exclusion is partial: the product-history features are missing for a product's first review, which still marks it as early, and `words_vs_prior_mean` carries 41.2% of the model's gain (`reports/ranker/ranker.md`); whether this is residual exposure or real signal was not tested.
- Selected hyperparameters: num_leaves 31, min_data_in_leaf 200, 214 trees (validation selection).
- Rejected additions (paired NDCG@3 interval includes 0): stacked TF-IDF score (−0.0014 [−0.0073, +0.0041]); zero-shot FinBERT and general-sentiment probabilities (+0.0000, −0.0006, −0.0023). Fine-tuned DistilBERT and FinBERT (60k reviews, 2 epochs): standalone NDCG@3 0.5816 and 0.5840 (below longest-first), and cross-fitted blends with the primary +0.0013 [−0.0032, +0.0057] and +0.0023 [−0.0022, +0.0068], so no transformer is in the final system.

## Data

Amazon Reviews 2023 (McAuley Lab), category Health and Personal Care: 494,121 reviews, 2001-02 to 2023-09; 488,990 after removing 5,131 exact duplicates. Temporal split by posting time: train 2015-01-01 to 2020-06-30 (289,187 reviews), validation 2020-07 to 2020-12 (40,087), sealed test 2021 (68,463), unlabeled stream 2022 onward (54,540). Source file pinned and checksum-verified (`artifacts/manifests/source_manifest.json`). No dataset licence is assigned; maintainers permit academic, non-commercial use; this project is non-commercial and does not redistribute raw text. Cite: Hou et al., "Bridging Language and Items for Retrieval and Recommendation", arXiv:2403.03952 (2024).

## Metrics (validation, 1,373 groups; product-clustered bootstrap 95% intervals)

| Model | NDCG@3 |
|---|---|
| Random | 0.3282 [0.3114, 0.3436] |
| Oldest-first | 0.4774 [0.4576, 0.4963] |
| Longest-first (best baseline) | 0.5991 [0.5819, 0.6172] |
| TF-IDF + Ridge | 0.5897 [0.5715, 0.6083] |
| **LightGBM LambdaRank, content-only** | **0.6571 [0.6403, 0.6736]** |

Paired gain over longest-first: +0.0580 [+0.0451, +0.0709]. Groups are (product, half-year) with at least 5 reviews and at least one voted review (72.9% of size-eligible groups). NDCG values are not comparable across group definitions.

## Known limitations and biases

- **Proxy label.** Helpful votes depend on position, exposure and timing; no impression logs exist, so this cannot be corrected, only controlled by grouping.
- **Over-preference for images and length.** In a validation check, the model's top-3 picks average 0.43 images per review (the most-voted reviews: 0.27; `reports/test_dryrun/test.json`) and 72.7 words (most-voted: 54.3). Readers' votes favor detail but less strongly than the model does.
- **One category, one marketplace snapshot.** No claim about other categories or sites.
- **Retrospective evidence only.** No online experiment; no claim about purchases, satisfaction or revenue.
- **Language.** Text is almost entirely ASCII (0.09% mostly non-ASCII); behavior on other languages is untested.
- **Short reviews dominate.** Median review is 20 words; 10.3% have 3 words or fewer.
- **Fairness.** Not evaluated: no reviewer or product demographic attributes are used or available. Amplifying long, image-bearing reviews may disadvantage reviewers who write short text or cannot post images.

## Serving and monitoring

FastAPI service (`/health`, `/model-info`, `/rank`) that rejects unknown fields, including vote data; API and batch features match (parity test). Offline drift replay on 54,540 unlabeled 2022–2023 reviews: whole-stream PSI ≤ 0.057 against the validation period; no month with ≥1,000 reviews above PSI 0.25 (max 0.218; five sparse months with fewer than 1,000 reviews are excluded, and across all 21 months the maximum is 1.854, in 2023-09 with 17 reviews). PSI shows input and score drift only, not accuracy. Not deployed to AWS (see `docs/deploy_aws.md`).

## Personal information and ethics

Review text can contain personal information. Raw text and data files are not committed or published. The model gives no score for individual people and should not be used to rank or evaluate reviewers.

## How to reproduce

`docs/planning/03_METRICS_AND_DECISIONS.md` lists run IDs, input hashes and artifacts; scripts `scripts/run_stage1.py` through `run_stage10_drift_replay.py` regenerate every report; `pytest` runs 119 tests (110 verified in a fresh environment; 9 not yet verified on an unrestricted machine, see the README).
