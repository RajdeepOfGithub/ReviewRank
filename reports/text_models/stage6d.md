# ReviewRank — Fine-tuned transformers (Stage 6d)

Run `EXP-20260930-T02`. **Validation only; the sealed test year was not loaded.** 1373 groups; product-clustered bootstrap (2000 replicates, 95%).
Blend weight chosen by 2-fold cross-fitting over validation products.

| Model | NDCG@3 [95% CI] | Paired vs primary [95% CI] | Verdict |
|---|---|---|---|
| longest_first | 0.5991 [0.5819, 0.6172] | -0.0580 [-0.0709, -0.0451] |  |
| primary_content_only | 0.6571 [0.6403, 0.6736] | — | reference |
| distilbert_standalone | 0.5816 [0.5635, 0.5992] | -0.0755 [-0.0903, -0.0608] |  |
| finbert_standalone | 0.5840 [0.5668, 0.6019] | -0.0730 [-0.0884, -0.0574] |  |
| blend_distilbert_crossfit | 0.6584 [0.6419, 0.6747] | +0.0013 [-0.0032, +0.0057] | reject (CI lower bound not above 0) |
| blend_finbert_crossfit | 0.6594 [0.6429, 0.6756] | +0.0023 [-0.0022, +0.0068] | reject (CI lower bound not above 0) |

## Blend weights

- distilbert: weights chosen per fold {0: np.float64(0.1), 1: np.float64(0.1)}; full-validation NDCG@3 by weight: 0.1:0.6584, 0.2:0.6551, 0.3:0.6489, 0.4:0.6461, 0.5:0.6398, 0.6:0.6312, 0.7:0.6191, 0.8:0.6096, 0.9:0.5965
- finbert: weights chosen per fold {0: np.float64(0.1), 1: np.float64(0.1)}; full-validation NDCG@3 by weight: 0.1:0.6594, 0.2:0.6565, 0.3:0.6561, 0.4:0.6504, 0.5:0.6422, 0.6:0.6327, 0.7:0.6236, 0.8:0.6076, 0.9:0.5977
