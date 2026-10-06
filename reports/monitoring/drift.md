# ReviewRank — Offline drift replay (Stage 10)

Stream: 54540 unlabeled reviews, 2022-01 to 2023-09 (the last month may be partial). Reference: validation period (2020-07 to 2020-12), 40087 reviews. **Input and score drift only; no accuracy claim.** The script reads no vote column. Source: `reports/monitoring/drift.json`, plot `drift.png`.

| Monitored column | Max monthly PSI | Level of max | Whole stream vs validation | Whole stream vs train | Months above 0.25 |
|---|---:|---|---:|---:|---|
| words | 0.849 | large | 0.004 | 0.039 | 2023-09 |
| chars | 0.204 | moderate | 0.004 | 0.034 | none |
| rating | 0.626 | large | 0.012 | 0.040 | 2023-09 |
| image_count | 0.438 | large | 0.003 | 0.051 | 2023-09 |
| verified | 0.218 | moderate | 0.037 | 0.029 | none |
| prior_mean_rating | 1.844 | large | 0.057 | 0.083 | 2023-09 |
| prior_mean_words | 1.548 | large | 0.026 | 0.027 | 2023-09 |
| words_vs_prior_mean | 1.854 | large | 0.002 | 0.019 | 2023-09 |
| score | 1.654 | large | 0.003 | 0.033 | 2023-09 |

Months with fewer than 1,000 reviews (PSI is noisy at small n): {'2023-05': 705, '2023-06': 530, '2023-07': 382, '2023-08': 210, '2023-09': 17}.

PSI thresholds (0.1 / 0.25) are conventions. A flagged month means the inputs or scores look different from the reference; it does not say the ranking got worse. Checking accuracy would need votes, which arrive late (maturity rule, ADR-013).
