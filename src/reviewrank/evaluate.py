"""Per-group ranking evaluation.

Every model scores exactly the same rows. Within each group, items are ordered by score
with the same seeded tie rule (configs/eval.yaml: ties), then NDCG@k and Precision@k are
computed from the relevance grades in ranked order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ndcg_at_k, order_by_score, precision_at_k


def per_group_metrics(
    rows: pd.DataFrame,
    scores,
    *,
    seed: int = 42,
    ndcg_ks: tuple[int, ...] = (3, 5),
    precision_k: int = 3,
) -> pd.DataFrame:
    """One row per group: group_id, parent_asin, size, ndcg_at_{k}, precision_at_{k}.

    `rows` needs group_id, parent_asin and grade. Groups with undefined NDCG are an error
    here because eligibility (labels.build_groups) guarantees at least one voted review.
    """
    rows = rows.reset_index(drop=True)
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(rows):
        raise ValueError("scores must align with rows")
    grades = rows["grade"].to_numpy()
    parents = rows["parent_asin"].to_numpy()

    out = []
    for group_id, idx in rows.groupby("group_id").indices.items():
        order = order_by_score(scores[idx], seed)
        ranked = grades[idx][order].tolist()
        record = {"group_id": group_id, "parent_asin": parents[idx[0]], "size": len(idx)}
        for k in ndcg_ks:
            value = ndcg_at_k(ranked, k)
            if value is None:
                raise ValueError(f"NDCG undefined for group {group_id}; it should not be eligible")
            record[f"ndcg_at_{k}"] = value
        record[f"precision_at_{precision_k}"] = precision_at_k(ranked, precision_k)
        out.append(record)
    return pd.DataFrame(out)
