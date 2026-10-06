# ReviewRank — DRY RUN ON VALIDATION (pipeline check, not a result)

Freeze `None`. Split: **validation**. Groups: **1373** (products 1373); size-ok groups 1883, with a voted review 1373 (coverage 72.9%). Intervals: product-clustered bootstrap (2000 replicates, 95%). Source: `reports\test_dryrun/test.json`.

| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |
|---|---|---|---|
| random_seeded | 0.3282 [0.3114, 0.3436] | 0.4257 [0.4099, 0.4408] | 0.3608 [0.3430, 0.3778] |
| oldest_first | 0.4774 [0.4576, 0.4963] | 0.5571 [0.5395, 0.5741] | 0.4540 [0.4365, 0.4717] |
| longest_first | 0.5991 [0.5819, 0.6172] | 0.6625 [0.6477, 0.6776] | 0.5254 [0.5084, 0.5426] |
| tfidf_ridge | 0.5897 [0.5715, 0.6083] | 0.6559 [0.6405, 0.6713] | 0.5237 [0.5076, 0.5407] |
| frozen_lgbm_content_only | 0.6571 [0.6403, 0.6736] | 0.7115 [0.6967, 0.7249] | 0.5649 [0.5484, 0.5812] |

## Paired differences in NDCG@3

| Comparison | Difference [95% CI] |
|---|---|
| frozen_minus_longest_first | +0.0580 [+0.0451, +0.0709] |
| frozen_minus_tfidf_ridge | +0.0674 [+0.0532, +0.0817] |
| frozen_minus_oldest_first | +0.1797 [+0.1556, +0.2035] |
| frozen_minus_random_seeded | +0.3289 [+0.3083, +0.3488] |
| longest_first_minus_tfidf_ridge | +0.0094 [-0.0016, +0.0209] |

## Segments (group level)

| Segment | Groups | Frozen NDCG@3 | Longest-first NDCG@3 | Paired difference [95% CI] |
|---|---:|---:|---:|---|
| group_size_5_6 | 387 | 0.7616 | 0.7196 | +0.0420 [+0.0198, +0.0638] |
| group_size_7_9 | 363 | 0.6805 | 0.6336 | +0.0469 [+0.0229, +0.0723] |
| group_size_10_plus | 623 | 0.5785 | 0.5042 | +0.0743 [+0.0530, +0.0956] |
| group_mean_rating_below_4 | 563 | 0.6543 | 0.5839 | +0.0704 [+0.0508, +0.0914] |
| group_mean_rating_4_plus | 810 | 0.6590 | 0.6096 | +0.0493 [+0.0327, +0.0661] |
| window_2020H2 | 1373 | 0.6571 | 0.5991 | +0.0580 [+0.0451, +0.0709] |

## What the top-3 reviews look like (mean over groups)

| Selection | words | rating | images | verified | votes |
|---|---:|---:|---:|---:|---:|
| random_seeded | 37.045 | 3.982 | 0.153 | 0.908 | 1.317 |
| oldest_first | 39.473 | 4.071 | 0.181 | 0.905 | 1.756 |
| longest_first | 80.228 | 3.87 | 0.268 | 0.874 | 2.343 |
| tfidf_ridge | 75.57 | 3.76 | 0.269 | 0.877 | 2.324 |
| frozen_lgbm_content_only | 72.729 | 3.698 | 0.431 | 0.917 | 2.696 |
| oracle_by_votes | 54.306 | 3.853 | 0.266 | 0.919 | 4.133 |
| all_reviews_in_eligible_groups | 36.14 | 3.958 | 0.136 | 0.927 | 1.164 |

## Sensitivity (mean NDCG@3, other group definitions, same frozen model)

| Definition | Groups | random | oldest_first | longest_first | tfidf_ridge | frozen_lgbm |
|---|---:|---:|---:|---:|---:|---:|
| min_size_3 | 2080 | 0.4705 | 0.5876 | 0.6803 | 0.6709 | 0.7226 |
| min_size_10 | 623 | 0.2072 | 0.3603 | 0.5042 | 0.4839 | 0.5785 |
| require_two_voted | 1016 | 0.3617 | 0.4820 | 0.6017 | 0.5920 | 0.6596 |
| quarter_window | 1305 | 0.3717 | 0.4521 | 0.6241 | 0.6134 | 0.6808 |
| binary_grade | 1373 | 0.4085 | 0.5484 | 0.6549 | 0.6504 | 0.7092 |
