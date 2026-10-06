# ReviewRank — Zero-shot sentiment features (Stage 6b)

Run `EXP-20260930-T01`. **Validation only; the sealed test year was not loaded.** 1373 groups; product-clustered bootstrap (2000 replicates, 95%). Source: `reports/text_models/stage6b.json`.

| Model | NDCG@3 [95% CI] | Paired gain vs primary [95% CI] | Verdict by rule |
|---|---|---|---|
| content_only (primary) | 0.6571 [0.6403, 0.6736] | — | reference |
| plus_finbert | 0.6571 [0.6410, 0.6737] | +0.0000 [-0.0058, +0.0063] | reject (CI lower bound not above 0) |
| plus_sst2 | 0.6565 [0.6400, 0.6726] | -0.0006 [-0.0054, +0.0046] | reject (CI lower bound not above 0) |
| plus_both | 0.6548 [0.6377, 0.6705] | -0.0023 [-0.0088, +0.0044] | reject (CI lower bound not above 0) |

## Polarity agreement with star ratings (validation rows with rating != 3; rating used as a noisy sentiment label)

| Model | Rows | Polarity accuracy (positive vs negative) | Spearman(net score, rating) |
|---|---:|---:|---:|
| finbert | 37260 | 0.876 | +0.593 |
| sst2 | 37260 | 0.863 | +0.679 |

FinBERT predicts *neutral* as its top class for 80.8% of validation reviews. Spearman between FinBERT and SST-2 net scores: +0.733.

## Spearman of each score with votes (train; descriptive only)

| Score | Spearman |
|---|---:|
| finbert_positive | -0.104 |
| finbert_negative | +0.123 |
| finbert_neutral | +0.025 |
| sst2_negative | +0.157 |
| sst2_positive | -0.157 |
