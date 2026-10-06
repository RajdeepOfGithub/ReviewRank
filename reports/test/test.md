# ReviewRank — SEALED TEST (one-time evaluation)

Freeze `FREEZE-20260930-01`. Split: **test**. Groups: **2164** (products 1696); size-ok groups 3026, with a voted review 2164 (coverage 71.5%). Intervals: product-clustered bootstrap (2000 replicates, 95%). Source: `reports\test/test.json`.

| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |
|---|---|---|---|
| random_seeded | 0.3398 [0.3254, 0.3538] | 0.4292 [0.4153, 0.4429] | 0.3561 [0.3428, 0.3702] |
| oldest_first | 0.4442 [0.4285, 0.4602] | 0.5254 [0.5106, 0.5403] | 0.4250 [0.4100, 0.4398] |
| longest_first | 0.6223 [0.6083, 0.6364] | 0.6812 [0.6691, 0.6933] | 0.5382 [0.5242, 0.5521] |
| tfidf_ridge | 0.6124 [0.5987, 0.6272] | 0.6701 [0.6581, 0.6829] | 0.5325 [0.5182, 0.5464] |
| frozen_lgbm_content_only | 0.6558 [0.6424, 0.6699] | 0.7131 [0.7020, 0.7244] | 0.5578 [0.5440, 0.5725] |

## Paired differences in NDCG@3

| Comparison | Difference [95% CI] |
|---|---|
| frozen_minus_longest_first | +0.0335 [+0.0236, +0.0429] |
| frozen_minus_tfidf_ridge | +0.0434 [+0.0324, +0.0544] |
| frozen_minus_oldest_first | +0.2116 [+0.1934, +0.2303] |
| frozen_minus_random_seeded | +0.3160 [+0.2982, +0.3341] |
| longest_first_minus_tfidf_ridge | +0.0099 [+0.0003, +0.0194] |

## Segments (group level)

| Segment | Groups | Frozen NDCG@3 | Longest-first NDCG@3 | Paired difference [95% CI] |
|---|---:|---:|---:|---|
| group_size_5_6 | 628 | 0.7392 | 0.7048 | +0.0344 [+0.0161, +0.0526] |
| group_size_7_9 | 476 | 0.6967 | 0.6579 | +0.0388 [+0.0168, +0.0592] |
| group_size_10_plus | 1060 | 0.5881 | 0.5574 | +0.0307 [+0.0167, +0.0442] |
| group_mean_rating_below_4 | 946 | 0.6356 | 0.5975 | +0.0381 [+0.0227, +0.0531] |
| group_mean_rating_4_plus | 1218 | 0.6715 | 0.6415 | +0.0300 [+0.0178, +0.0432] |
| window_2021H1 | 1245 | 0.6537 | 0.6181 | +0.0356 [+0.0225, +0.0476] |
| window_2021H2 | 919 | 0.6587 | 0.6279 | +0.0308 [+0.0160, +0.0448] |

## What the top-3 reviews look like (mean over groups)

| Selection | words | rating | images | verified | votes |
|---|---:|---:|---:|---:|---:|
| random_seeded | 36.701 | 3.9 | 0.158 | 0.902 | 1.411 |
| oldest_first | 38.903 | 3.996 | 0.191 | 0.893 | 1.796 |
| longest_first | 81.486 | 3.806 | 0.297 | 0.865 | 2.534 |
| tfidf_ridge | 77.233 | 3.681 | 0.291 | 0.867 | 2.464 |
| frozen_lgbm_content_only | 74.922 | 3.595 | 0.44 | 0.904 | 2.712 |
| oracle_by_votes | 55.973 | 3.725 | 0.246 | 0.909 | 3.938 |
| all_reviews_in_eligible_groups | 34.617 | 3.941 | 0.128 | 0.923 | 1.176 |

## Sensitivity (mean NDCG@3, other group definitions, same frozen model)

| Definition | Groups | random | oldest_first | longest_first | tfidf_ridge | frozen_lgbm |
|---|---:|---:|---:|---:|---:|---:|
| min_size_3 | 3272 | 0.4699 | 0.5575 | 0.6901 | 0.6843 | 0.7186 |
| min_size_10 | 1060 | 0.2205 | 0.3248 | 0.5574 | 0.5369 | 0.5881 |
| require_two_voted | 1611 | 0.3437 | 0.4492 | 0.6318 | 0.6220 | 0.6628 |
| quarter_window | 2040 | 0.3537 | 0.4211 | 0.6505 | 0.6365 | 0.6822 |
| binary_grade | 2164 | 0.4158 | 0.5145 | 0.6719 | 0.6618 | 0.6998 |
