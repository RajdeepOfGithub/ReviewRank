# ReviewRank — Feature report (Stage 4)

Run `STAGE4-20260930`. Features for excluded + train + validation rows; **the sealed test year and the stream were not loaded**. 23 features in 5 groups. Source: `reports/features/feature_report.json`.

Safety checks (unit tests, `tests/test_features.py`): changing votes never changes any feature; changing a later review never changes an earlier review's features; reviews with the same timestamp do not see each other; the output has no column named like a vote.

| Group | Feature | Missing % (train) | Spearman with votes (train) |
|---|---|---:|---:|
| length | words | 0.0 | +0.313 |
| length | chars | 0.0 | +0.316 |
| length | title_chars | 0.0 | +0.173 |
| length | title_words | 0.0 | +0.161 |
| length | avg_word_len | 0.01 | -0.033 |
| length | sentences | 0.0 | +0.281 |
| length | unique_word_ratio | 0.12 | -0.281 |
| style | exclaims | 0.0 | +0.056 |
| style | questions | 0.0 | +0.073 |
| style | digits | 0.0 | +0.195 |
| style | caps_ratio | 0.01 | -0.092 |
| style | is_empty_text | 0.0 | +0.006 |
| media_meta | image_count | 0.0 | +0.114 |
| media_meta | has_image | 0.0 | +0.114 |
| media_meta | rating | 0.0 | -0.067 |
| media_meta | verified | 0.0 | -0.008 |
| asof_product | prior_reviews | 0.0 | -0.111 |
| asof_product | prior_mean_rating | 12.07 | +0.027 |
| asof_product | rating_minus_prior_mean | 12.07 | -0.083 |
| asof_product | prior_mean_words | 12.07 | +0.071 |
| asof_product | words_vs_prior_mean | 12.07 | +0.232 |
| calendar | month | 0.0 | +0.007 |
| calendar | hour | 0.0 | +0.003 |

Notes: Spearman values are descriptive, on the training split only, and are not evidence a feature will help a ranker. `prior_*` features are missing for a product's first review by design (no earlier reviews). `prior_reviews` and `calendar` are a proxy for age/exposure and are ablated in Stage 5.
