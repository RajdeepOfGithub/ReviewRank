"""Ranking metrics.

Definitions (fixed in configs/eval.yaml):
    gain(rel)        = 2**rel - 1
    discount(pos)    = log2(pos + 1), pos is 1-based
    DCG@k            = sum over the first k ranked items of gain / discount
    NDCG@k           = DCG@k / DCG@k of the ideal (relevance-sorted) order
A group whose ideal DCG is 0 (no relevant item) has NDCG undefined; it is
excluded and the coverage is reported, never silently scored as 0 or 1.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _check_relevances(relevances: Sequence[float]) -> None:
    if any(r < 0 for r in relevances):
        raise ValueError("relevance grades must be non-negative")


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """DCG of relevances listed in ranked order (best-ranked first)."""
    if k <= 0:
        raise ValueError("k must be positive")
    _check_relevances(relevances)
    return sum((2.0**r - 1.0) / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float | None:
    """NDCG@k for relevances in ranked order; None when undefined (ideal DCG == 0)."""
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0.0:
        return None
    return dcg_at_k(relevances, k) / ideal


def precision_at_k(relevances: Sequence[float], k: int, threshold: float = 1.0) -> float:
    """Share of the first k ranked items with relevance >= threshold."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = relevances[:k]
    if not top:
        raise ValueError("empty ranking")
    return sum(1 for r in top if r >= threshold) / len(top)


def order_by_score(scores: Sequence[float], seed: int) -> np.ndarray:
    """Indices that rank items by descending score; ties broken by a seeded permutation.

    Deterministic for a given (scores, seed). The same tie rule must be used for
    every model being compared (configs/eval.yaml: ties).
    """
    scores = np.asarray(scores, dtype=float)
    tiebreak = np.random.default_rng(seed).permutation(len(scores))
    # np.lexsort sorts by the LAST key first; negate score for descending order.
    return np.lexsort((tiebreak, -scores))


def mean_ndcg_with_coverage(values: Sequence[float | None]) -> tuple[float, float]:
    """Mean NDCG over defined groups and the share of groups that were defined."""
    if len(values) == 0:
        raise ValueError("no groups")
    defined = [v for v in values if v is not None]
    coverage = len(defined) / len(values)
    if not defined:
        raise ValueError("NDCG undefined for every group")
    return float(np.mean(defined)), coverage
