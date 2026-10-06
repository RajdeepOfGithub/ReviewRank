# ReviewRank — LightGBM LambdaRank on validation (Stage 5, run R02)

Run `EXP-20260930-R02`. **Validation only (2020-07-01 to 2020-12-31); the sealed test year was not loaded.** Source: `reports/ranker/ranker.json`. The first run (`R01`) is kept in `ranker_R01.*`.
Groups evaluated: **1373**; intervals are product-clustered bootstrap (2000 replicates, 95%). Training groups: (product, half-year) with >=2 reviews and >=1 voted review (20868 groups, 198331 rows).
Selection is on validation, so these numbers are **candidates, not a final result**.

**Primary candidate: `content_only`.** Exposure/age proxies (prior_reviews, month, hour) are excluded from the primary candidates (ADR-019); `features_all` includes them for context.

| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |
|---|---|---|---|
| oldest_first | 0.4774 [0.4576, 0.4963] | 0.5571 [0.5395, 0.5741] | 0.4540 [0.4365, 0.4717] |
| longest_first | 0.5991 [0.5819, 0.6172] | 0.6625 [0.6477, 0.6776] | 0.5254 [0.5084, 0.5426] |
| tfidf_ridge | 0.5897 [0.5715, 0.6083] | 0.6559 [0.6405, 0.6713] | 0.5237 [0.5076, 0.5407] |
| lgbm_features_all | 0.6748 [0.6584, 0.6909] | 0.7310 [0.7178, 0.7447] | 0.5734 [0.5572, 0.5895] |
| lgbm_content_only | 0.6571 [0.6403, 0.6736] | 0.7115 [0.6967, 0.7249] | 0.5649 [0.5484, 0.5812] |
| lgbm_content_plus_tfidf | 0.6557 [0.6392, 0.6725] | 0.7105 [0.6965, 0.7243] | 0.5625 [0.5458, 0.5785] |

## Paired differences in NDCG@3 (same groups, same resampled products)

| Comparison | Difference [95% CI] |
|---|---|
| primary_content_only_minus_longest_first | +0.0580 [+0.0451, +0.0709] |
| primary_content_only_minus_oldest_first | +0.1797 [+0.1556, +0.2035] |
| primary_content_only_minus_tfidf_ridge | +0.0674 [+0.0532, +0.0817] |
| content_plus_tfidf_minus_content_only | -0.0014 [-0.0073, +0.0041] |
| features_all_minus_primary_content_only | +0.0177 [+0.0101, +0.0251] |
| oldest_first_minus_longest_first | -0.1217 [-0.1474, -0.0962] |

## Hyperparameter grid (validation NDCG@3)

| Feature set | num_leaves | min_data_in_leaf | best iteration | NDCG@3 | seconds |
|---|---:|---:|---:|---:|---:|
| features_all | 15 | 50 | 461 | 0.6748 **(selected)** | 1.7 |
| features_all | 15 | 200 | 119 | 0.6632 | 0.8 |
| features_all | 31 | 50 | 372 | 0.6740 | 2.0 |
| features_all | 31 | 200 | 337 | 0.6679 | 1.9 |
| features_all | 63 | 50 | 141 | 0.6624 | 1.7 |
| features_all | 63 | 200 | 280 | 0.6666 | 2.6 |
| content_only | 15 | 50 | 358 | 0.6563 | 1.6 |
| content_only | 15 | 200 | 156 | 0.6533 | 1.0 |
| content_only | 31 | 50 | 139 | 0.6567 | 1.1 |
| content_only | 31 | 200 | 214 | 0.6571 **(selected)** | 1.6 |
| content_only | 63 | 50 | 53 | 0.6537 | 1.0 |
| content_only | 63 | 200 | 63 | 0.6489 | 1.1 |
| content_plus_tfidf | 15 | 50 | 288 | 0.6544 | 1.6 |
| content_plus_tfidf | 15 | 200 | 206 | 0.6529 | 1.3 |
| content_plus_tfidf | 31 | 50 | 127 | 0.6557 **(selected)** | 1.2 |
| content_plus_tfidf | 31 | 200 | 139 | 0.6539 | 1.2 |
| content_plus_tfidf | 63 | 50 | 51 | 0.6532 | 1.0 |
| content_plus_tfidf | 63 | 200 | 109 | 0.6534 | 1.3 |

## Ablations from the primary candidate (`content_only`); difference in NDCG@3 vs the primary

| Variant | Features | NDCG@3 | Difference vs primary [95% CI] |
|---|---:|---:|---|
| minus_length | 13 | 0.6460 | -0.0111 [-0.0185, -0.0036] |
| minus_style | 15 | 0.6492 | -0.0079 [-0.0139, -0.0017] |
| minus_media_meta | 16 | 0.6302 | -0.0269 [-0.0375, -0.0168] |
| minus_asof_product | 16 | 0.6343 | -0.0227 [-0.0314, -0.0140] |
| minus_words_vs_prior_mean | 19 | 0.6499 | -0.0071 [-0.0144, +0.0002] |
| plus_exposure_proxies | 23 | 0.6723 | +0.0152 [+0.0076, +0.0232] |

## Top features by gain (primary)

| Feature | Share of gain |
|---|---:|
| words_vs_prior_mean | 0.412 |
| chars | 0.229 |
| words | 0.077 |
| rating_minus_prior_mean | 0.054 |
| image_count | 0.049 |
| verified | 0.040 |
| prior_mean_words | 0.029 |
| prior_mean_rating | 0.022 |
| avg_word_len | 0.017 |
| rating | 0.012 |
| caps_ratio | 0.012 |
| title_chars | 0.009 |
| has_image | 0.009 |
| digits | 0.008 |
| unique_word_ratio | 0.008 |

## Sensitivity (mean NDCG@3, different group definitions)

| Definition | Groups | oldest_first | longest_first | tfidf_ridge | lgbm_features_all | lgbm_primary_content_only |
|---|---:|---:|---:|---:|---:|---:|
| min_size_3 | 2080 | 0.5876 | 0.6803 | 0.6709 | 0.7361 | 0.7226 |
| min_size_10 | 623 | 0.3603 | 0.5042 | 0.4839 | 0.6030 | 0.5785 |
| require_two_voted | 1016 | 0.4820 | 0.6017 | 0.5920 | 0.6755 | 0.6596 |
| quarter_window | 1305 | 0.4521 | 0.6241 | 0.6134 | 0.6919 | 0.6808 |
| binary_grade | 1373 | 0.5484 | 0.6549 | 0.6504 | 0.7232 | 0.7092 |
