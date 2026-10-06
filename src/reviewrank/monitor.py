"""Drift monitoring helpers (offline replay; no labels).

PSI (population stability index) compares a current distribution with a reference:
    PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)   over bins i (proportions, floored at EPS).
Common reading: < 0.1 small, 0.1-0.25 moderate, > 0.25 large. These are conventions, not laws; with
few distinct values the feature is treated as categorical (one bin per value).
PSI measures INPUT or SCORE drift. It says nothing about accuracy; that needs labels (delayed votes).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-4


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(float)
    return np.maximum(counts / max(counts.sum(), 1.0), EPS)


def psi(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("empty reference or current sample")

    uniq = np.unique(ref)
    if len(uniq) <= bins:  # categorical-like (for example 0/1 flags, star ratings)
        cats = np.union1d(uniq, np.unique(cur))
        edges = np.sort(np.unique(np.concatenate([cats - 0.5, [cats[-1] + 0.5]])))  # one bin per value
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < bins // 2:  # heavy ties (zero-inflated counts): quantile edges collapse, so add
            top = ref[ref > edges[0]]  # edges from the values above the most common lowest value
            if len(top):
                edges = np.unique(np.concatenate([edges, np.quantile(top, np.linspace(0, 1, bins))]))
        if len(edges) < 3:
            raise ValueError("reference has too little variation to bin")
        edges[0], edges[-1] = -np.inf, np.inf  # catch values outside the reference range
    p_ref, p_cur = _proportions(ref, edges), _proportions(cur, edges)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def psi_level(value: float) -> str:
    if value < 0.1:
        return "small"
    if value < 0.25:
        return "moderate"
    return "large"
