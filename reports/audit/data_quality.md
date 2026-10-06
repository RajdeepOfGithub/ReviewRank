# ReviewRank — Data quality report (Stage 2, 2026-09-30)

Machine-generated inputs: `reports/audit/audit.json`, `reports/eda/followups.json`, `artifacts/manifests/clean_manifest.json`. Produced by `scripts/stage2_audit_eda.py` and `src/reviewrank/clean.py`. Checks here are label-free except where noted (duplicate comparison used development rows only).

## Checks on the full file (494,121 rows, before cleaning)

| Check | Result | Decision |
|---|---|---|
| Nulls in the 10 source columns | none (Stage 1) | none needed |
| Empty (whitespace-only) text | 249 (0.05%) | keep rows; features must handle empty text |
| Exact duplicates on (user_id, parent_asin, timestamp_ms) | 4,382 keys, 5,131 extra rows (1.04%). In development rows 100% of these had identical text and identical votes | **remove extras, keep one (ADR-008)** |
| Same user, same product, different times | 5,082 pairs, 11,015 rows involved (2.2%) | keep (can be legitimate); checked again at Stage 4 |
| Identical normalized text, ≥20 chars, across reviews | 5,362 texts, 12,970 rows (2.6%), largest repeat 47 | keep in train; flag in validation/test if also in train |
| Validation/test text also present in train (before dedupe) | test 353 of 62,855 (0.56%); validation 233 of 36,955 (0.63%) | **flag and exclude from evaluation (ADR-009)** |
| Users in validation/test who also posted in train | test 3,307 of 69,202 (4.8%); validation 2,152 of 40,549 (5.3%) | keep; low overlap; noted |
| Mostly non-ASCII text | 0.09% | no language filter needed now |
| Words per review | median 20, p75 43, p90 80, p95 117, p99 237, max 3,455; ≤3 words 10.3%; >100 words 6.7% | sequence length chosen in Stage 6 |
| Reviews per product | median 2, p75 5, p90 14, p99 97, max 3,192 | see group sizes below |
| Products with ≥5 / ≥10 / ≥20 / ≥50 reviews | 16,956 / 8,734 / 4,199 / 1,414 (of 60,274) | — |
| Share of reviews in the top 1% of products | 33.7% | popularity is concentrated; grouping by product controls it |

## Cleaning result (`clean_manifest.json`)

- Rows in 494,121 → rows out **488,990** (5,131 duplicates removed; matches the audit).
- Rows per split after cleaning: excluded (before 2015-01-01) **36,713**; train **289,187** (2015-01-01 to 2020-06-30); validation **40,087** (2020-07-01 to 2020-12-31); test **68,463** (sealed, 2021); stream **54,540** (2022-01-01 onward, unlabeled).
- Flagged `text_seen_in_train`: test **344**, validation **229** (excluded from evaluation, kept in the file).

## Group sizes (review counts only, no votes; before cleaning)

Query group = (`parent_asin`, window). Share = share of that split's reviews that sit in a group with at least 5 reviews.

| Window | Validation groups ≥5 | Validation share | Test groups ≥5 | Test share |
|---|---:|---:|---:|---:|
| Month | 1,475 | 35.3% | 2,307 | 32.9% |
| Quarter | 1,870 | 54.6% | 2,937 | 51.6% |
| **Half-year (chosen)** | **1,914** | **64.7%** | **3,070** | **62.1%** |
| Year | 1,914 (same as half-year; validation is only 6 months) | 64.7% | 2,987 | 69.8% |

Half-year groups with ≥10 reviews: validation 744, test 1,260. Many thousands of groups remain, so confidence intervals will be usable.

## Not done / risks

- Near-duplicate (non-exact) text is not detected; revisit with MinHash only if Stage 3–4 results look suspicious.
- No language or spam/incentivized-review filter.
- Group eligibility with votes (at least one voted review) is computed by the Stage 3 label builder; on development data, about 73% of half-year groups with ≥5 reviews had at least one voted review (`followups.json`).
- The share of groups with votes falls with window year (about 80% in 2015 to 73% in 2020); the test year may be lower. Coverage will be reported.
