"""Cluster bootstrap over products.

Reviews and groups of one product are correlated, so we resample PRODUCTS (clusters) with
replacement, not individual groups. The statistic is the mean of per-group values over the
resampled products (each group counts once). Paired differences use the same resamples for
both models, so the interval reflects the difference, not two separate intervals.
"""

from __future__ import annotations

import numpy as np


def _cluster_sums(values: np.ndarray, clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, inverse = np.unique(clusters, return_inverse=True)
    n = inverse.max() + 1
    sums = np.bincount(inverse, weights=values, minlength=n)
    counts = np.bincount(inverse, minlength=n).astype(float)
    return sums, counts


def _resampled_means(sums: np.ndarray, counts: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(replicates, len(sums)))
    return sums[idx].sum(axis=1) / counts[idx].sum(axis=1)


def _interval(samples: np.ndarray, ci_level: float) -> tuple[float, float]:
    tail = (1.0 - ci_level) / 2.0
    lo, hi = np.quantile(samples, [tail, 1.0 - tail])
    return float(lo), float(hi)


def cluster_bootstrap_mean(
    values, clusters, *, replicates: int = 2000, ci_level: float = 0.95, seed: int = 42
) -> dict:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if len(values) != len(clusters) or len(values) == 0:
        raise ValueError("values and clusters must be non-empty and the same length")
    sums, counts = _cluster_sums(values, clusters)
    lo, hi = _interval(_resampled_means(sums, counts, replicates, seed), ci_level)
    return {
        "mean": float(values.mean()),
        "ci_lower": lo,
        "ci_upper": hi,
        "n_groups": int(len(values)),
        "n_clusters": int(len(sums)),
        "replicates": replicates,
        "ci_level": ci_level,
    }


def paired_cluster_bootstrap_diff(
    a, b, clusters, *, replicates: int = 2000, ci_level: float = 0.95, seed: int = 42
) -> dict:
    """Mean of (a - b) with a product-clustered CI. a and b are per-group values, aligned."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("a and b must be aligned")
    result = cluster_bootstrap_mean(a - b, clusters, replicates=replicates, ci_level=ci_level, seed=seed)
    result["mean_diff"] = result.pop("mean")
    return result
