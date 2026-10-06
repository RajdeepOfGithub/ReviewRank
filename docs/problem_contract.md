# ReviewRank — Problem contract (Stage 0)

**Status:** frozen items are marked FROZEN. Items marked PROVISIONAL are confirmed by evidence in Stage 2 and frozen before the first model run (Stage 3). Nothing here reports a result. No data has been downloaded.

## Decision and user

- **Decision:** which newly posted reviews to show first on a product page.
- **Problem (cold start):** a new review has zero votes, so vote-sorting cannot place it. The system scores a review **at posting time** from its own text and metadata and from product information that existed then.
- **User:** a shopper deciding on a product; indirectly, the team that owns review ranking.

## Prediction target

- **Unit:** one review. **Prediction moment:** the review's posting time. Only information available then may be used. FROZEN
- **Ranking task:** within a query group (same `parent_asin`, same time window), order reviews so that the ones that later earn the most helpful votes come first. FROZEN
- **Relevance label:** grades 0–3 from absolute vote bins (0, 1, 2–4, 5+ votes), compared only inside a query group. Query group = same `parent_asin` and same calendar half-year, at least 5 reviews; NDCG needs at least one voted review in the group and coverage is reported. Decided in Stage 2 (ADR-011/012). FROZEN
- **Maturity rule:** only reviews at least 18 months old at the 2023-09-11 snapshot are labeled (ADR-013). This is a convention, not a measured saturation point (one snapshot cannot show it); it excludes no row of our splits and forbids labeling reviews posted after 2022-03-11. FROZEN
- **Why not raw votes:** the share of reviews with at least one vote is about 65% for 2010, 45% for 2013, 34% for 2014, then roughly flat at 27–33% for 2015–2019 and 23.7% for 2020 (development rows, Stage 2 EDA-01). The effect is strong in old years and weaker later; raw votes still depend on age, volume and exposure, so we compare reviews only within a group. See `docs/planning/03_METRICS_AND_DECISIONS.md`.

## Data

- Amazon Reviews 2023 (McAuley Lab), category `Health_and_Personal_Care` for development. Fields: `asin, parent_asin, user_id, rating, title, text, images, timestamp, helpful_vote, verified_purchase`.
- Terms: Amazon's terms apply; cite Hou et al., ACL 2024. **Not yet verified at source.** Raw text is never committed or published. FROZEN (rule); verification is an open item.

## Split and evaluation

- **Temporal split** by posting time, UTC, half-open intervals (`configs/splits.yaml`): train before 2020-07-01, validation 2020-07-01 to 2020-12-31, sealed test 2021-01-01 to 2021-12-31, stream from 2022-01-01 (unlabeled, drift replay only). Training uses reviews from 2015-01-01 onward (ADR-010). **Boundaries confirmed in Stage 2 (2026-09-30) and FROZEN from now:** never changed after modeling starts without a new sealed test and a logged reason. FROZEN
- **Primary metric:** mean NDCG@3 over eligible groups with exponential gain and log2 discount, reported with coverage (share of groups with a defined NDCG). Also NDCG@5 and Precision@3. FROZEN
- **Uncertainty:** bootstrap over products (clusters), paired differences against each baseline, 95% intervals. FROZEN
- **Baselines (same rows and groups as every model):** seeded random, newest-first, longest-first, rating-extremity, TF-IDF linear. FROZEN
- **Selection:** on validation only. A freeze record is written before the test is opened; the test is opened once. FROZEN

## Leakage rules

1. `helpful_vote` and anything derived from it are never features.
2. Product-level features (mean rating, review count) use only reviews posted strictly before the row.
3. Review age at the snapshot must not be recoverable from features; calendar features are ablated to check this.
4. Vocabularies, scalers and encoders are fit on train only.
5. Same-user and near-duplicate reviews across the train/test boundary are checked and the decision logged.

## What this project will and will not claim

| Allowed | Not allowed |
|---|---|
| "Temporal backtest on [N] reviews, NDCG@3 [value, CI] vs [baseline]" | "Improves conversion / sales / shopper satisfaction" |
| "Helpful-vote proxy" | "Finds the truly helpful or truthful reviews" |
| "Offline drift replay on unlabeled 2022–2023 reviews" | "Monitored model accuracy in production" |
| "Deployed on AWS" only after it is live and verified | "Deployed" for a local Docker demo |

## Known limitations (today)

- Votes depend on display position and exposure; no impression logs exist, so this bias can be acknowledged and partly controlled by grouping but not removed.
- Helpfulness from text is a modest-signal task; small gains over simple baselines are a plausible honest result.
- One category is used for development; generalization to other categories is an extension, not a claim.
- Text may contain personal information; raw text is not redistributed.

## Open items before Stage 1

- ~~Verify Amazon Reviews licence terms at source.~~ **Done 2026-09-30:** no dataset licence is assigned; maintainers permit academic, non-commercial use; we cite and do not redistribute raw data or text (see `artifacts/manifests/source_manifest.json`).
- Stage 2 risk: about 8 reviews per product on average; if product-window groups are too small to rank, the query-group definition must change (for example, group by product over a longer window or by a coarser key), decided by EDA before any model.
- Verify AWS free-tier terms on the official page before any deployment; set a budget alarm first.
- GPU smoke test for PyTorch on this machine (ASUS ROG Flow Z13, Ryzen AI Max+ 395, Radeon 8060S). Native Windows ROCm support for this chip is reported by third-party sources and must be verified; v1's ROCm backpropagation crash on Linux is the reason to test first.
- Disk: about 18 GB free on C: at Stage 0; budget data, model caches and Docker images accordingly.
