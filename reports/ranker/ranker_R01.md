# ReviewRank — LightGBM LambdaRank on validation (Stage 5)

Run `EXP-20260930-R01`. **Validation only (2020-07-01 to 2020-12-31); the sealed test year was not loaded.** Source: `reports/ranker/ranker.json`.
Groups evaluated: **1373**, same as the baseline run; intervals are product-clustered bootstrap (2000 replicates, 95%).
Training groups: (product, half-year) with >=2 reviews and >=1 voted review (20868 groups, 198331 rows). Selection is on validation, so the numbers are **not** a final result.

| Model | NDCG@3 [95% CI] | NDCG@5 [95% CI] | Precision@3 [95% CI] |
|---|---|---|---|
| longest_first | 0.5991 [0.5819, 0.6172] | 0.6625 [0.6477, 0.6776] | 0.5254 [0.5084, 0.5426] |
| tfidf_ridge | 0.5897 [0.5715, 0.6083] | 0.6559 [0.6405, 0.6713] | 0.5237 [0.5076, 0.5407] |
| lgbm_features_only | 0.6748 [0.6584, 0.6909] | 0.7310 [0.7178, 0.7447] | 0.5734 [0.5572, 0.5895] |
| lgbm_features_plus_tfidf | 0.6716 [0.6547, 0.6879] | 0.7284 [0.7149, 0.7422] | 0.5747 [0.5579, 0.5904] |

## Paired differences in NDCG@3 (same groups, same resampled products)

| Comparison | Difference [95% CI] |
|---|---|
| lgbm_features_only_minus_longest_first | +0.0757 [+0.0622, +0.0890] |
| lgbm_features_plus_tfidf_minus_longest_first | +0.0725 [+0.0598, +0.0857] |
| lgbm_features_plus_tfidf_minus_tfidf_ridge | +0.0819 [+0.0687, +0.0962] |
| lgbm_features_plus_tfidf_minus_lgbm_features_only | -0.0032 [-0.0098, +0.0036] |

## Hyperparameter grid (validation NDCG@3)

| Feature set | num_leaves | min_data_in_leaf | best iteration | NDCG@3 | seconds |
|---|---:|---:|---:|---:|---:|
| features_only | 15 | 50 | 461 | 0.6748 **(selected)** | 2.7 |
| features_only | 15 | 200 | 119 | 0.6632 | 1.2 |
| features_only | 31 | 50 | 372 | 0.6740 | 2.9 |
| features_only | 31 | 200 | 337 | 0.6679 | 3.0 |
| features_only | 63 | 50 | 141 | 0.6624 | 2.2 |
| features_only | 63 | 200 | 280 | 0.6666 | 3.4 |
| features_plus_tfidf | 15 | 50 | 143 | 0.6637 | 1.3 |
| features_plus_tfidf | 15 | 200 | 267 | 0.6644 | 1.9 |
| features_plus_tfidf | 31 | 50 | 396 | 0.6690 | 2.9 |
| features_plus_tfidf | 31 | 200 | 386 | 0.6716 **(selected)** | 3.1 |
| features_plus_tfidf | 63 | 50 | 181 | 0.6637 | 2.8 |
| features_plus_tfidf | 63 | 200 | 267 | 0.6656 | 3.4 |

## Ablations (full model = features + TF-IDF score; difference in NDCG@3 vs the full model)

| Removed | Features left | NDCG@3 | Difference vs full [95% CI] |
|---|---:|---:|---|
| minus_tfidf_score | 23 | 0.6679 | -0.0037 [-0.0103, +0.0028] |
| minus_length | 17 | 0.6624 | -0.0092 [-0.0163, -0.0022] |
| minus_style | 19 | 0.6664 | -0.0052 [-0.0115, +0.0007] |
| minus_media_meta | 20 | 0.6492 | -0.0224 [-0.0331, -0.0121] |
| minus_asof_product | 19 | 0.6458 | -0.0258 [-0.0344, -0.0169] |
| minus_calendar | 22 | 0.6707 | -0.0009 [-0.0072, +0.0054] |
| minus_prior_reviews_and_calendar | 21 | 0.6539 | -0.0177 [-0.0260, -0.0095] |

## Top features by gain (full model)

| Feature | Share of gain |
|---|---:|
| words_vs_prior_mean | 0.374 |
| chars | 0.190 |
| tfidf_score | 0.071 |
| rating_minus_prior_mean | 0.047 |
| prior_reviews | 0.045 |
| image_count | 0.039 |
| verified | 0.035 |
| prior_mean_words | 0.028 |
| words | 0.024 |
| avg_word_len | 0.020 |
| month | 0.020 |
| caps_ratio | 0.015 |
| prior_mean_rating | 0.015 |
| has_image | 0.013 |
| title_chars | 0.011 |

## Sensitivity (mean NDCG@3, different group definitions)

| Definition | Groups | longest_first | tfidf_ridge | lgbm_features_only | lgbm_features_plus_tfidf |
|---|---:|---:|---:|---:|---:|
| min_size_3 | 2080 | 0.6803 | 0.6709 | 0.7361 | 0.7344 |
| min_size_10 | 623 | 0.5042 | 0.4839 | 0.6030 | 0.6036 |
| require_two_voted | 1016 | 0.6017 | 0.5920 | 0.6755 | 0.6737 |
| quarter_window | 1305 | 0.6241 | 0.6134 | 0.6919 | 0.6912 |
| binary_grade | 1373 | 0.6549 | 0.6504 | 0.7232 | 0.7222 |
