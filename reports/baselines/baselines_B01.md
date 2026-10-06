# ReviewRank — Baseline ladder on validation (Stage 3)

Run `EXP-20260930-B01`. **Validation only (2020-07-01 to 2020-12-31); the sealed test year was not loaded.** Source: `reports/baselines/baselines.json`.

Groups evaluated: **1373** (products: 1373); eligible reviews: 21500. Size-ok groups 1883, with a voted review 1373 (coverage 72.9%). Intervals: product-clustered bootstrap, 2000 replicates, 95%.

| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |
|---|---|---|---|
| random_seeded | 0.3282 [0.3114, 0.3436] | 0.4257 [0.4099, 0.4408] | 0.3608 [0.3430, 0.3778] |
| newest_first | 0.2319 [0.2162, 0.2465] | 0.3308 [0.3138, 0.3465] | 0.2821 [0.2644, 0.2998] |
| lowest_rating_first | 0.3919 [0.3735, 0.4103] | 0.4810 [0.4639, 0.4980] | 0.4127 [0.3952, 0.4307] |
| longest_first | 0.5991 [0.5819, 0.6172] | 0.6625 [0.6477, 0.6776] | 0.5254 [0.5084, 0.5426] |
| tfidf_ridge | 0.5834 [0.5649, 0.6008] | 0.6511 [0.6355, 0.6662] | 0.5193 [0.5025, 0.5361] |

## Paired differences in NDCG@3 (same groups, same resampled products)

| Comparison | Mean difference [95% CI] |
|---|---|
| random_seeded_minus_longest_first | -0.2709 [-0.2925, -0.2499] |
| newest_first_minus_longest_first | -0.3672 [-0.3877, -0.3469] |
| lowest_rating_first_minus_longest_first | -0.2071 [-0.2309, -0.1843] |
| tfidf_ridge_minus_longest_first | -0.0157 [-0.0279, -0.0036] |
| longest_first_minus_random_seeded | 0.2709 [0.2499, 0.2925] |

## Sensitivity (mean NDCG@3 on the same validation rows, different group definitions)

| Definition | Groups | random_seeded | newest_first | lowest_rating_first | longest_first | tfidf_ridge |
|---|---:|---|---|---|---|---|
| min_size_3 | 2080 | 0.4705 | 0.3743 | 0.5105 | 0.6803 | 0.6672 |
| min_size_10 | 623 | 0.2072 | 0.1259 | 0.2840 | 0.5042 | 0.4799 |
| require_two_voted | 1016 | 0.3617 | 0.2558 | 0.4020 | 0.6017 | 0.5872 |
| quarter_window | 1305 | 0.3717 | 0.2736 | 0.4140 | 0.6241 | 0.6138 |
| binary_grade | 1373 | 0.4085 | 0.3091 | 0.4863 | 0.6549 | 0.6435 |

TF-IDF Ridge: alpha chosen on validation from [1.0, 3.0, 10.0] (best 10.0); this slightly favors the baseline. Trained on the training split only (2015-01-01 to 2020-06-30).

Caveat: a single half-year (2020H2) of validation; these are baselines, not the final claim. The primary comparison for the project is against the best baseline here.
