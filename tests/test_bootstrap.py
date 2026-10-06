import numpy as np
import pytest

from reviewrank.bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap_diff


def test_constant_values_have_zero_width_interval():
    r = cluster_bootstrap_mean([0.5] * 10, list("aabbccddee"), replicates=200)
    assert r["mean"] == pytest.approx(0.5)
    assert r["ci_lower"] == pytest.approx(0.5) and r["ci_upper"] == pytest.approx(0.5)
    assert r["n_clusters"] == 5 and r["n_groups"] == 10


def test_same_seed_is_deterministic():
    rng = np.random.default_rng(0)
    v, c = rng.random(100), rng.integers(0, 20, 100)
    assert cluster_bootstrap_mean(v, c, seed=1) == cluster_bootstrap_mean(v, c, seed=1)
    assert cluster_bootstrap_mean(v, c, seed=1) != cluster_bootstrap_mean(v, c, seed=2)


def test_identical_models_have_zero_difference_and_zero_width():
    v = np.random.default_rng(1).random(60)
    c = np.repeat(np.arange(20), 3)
    r = paired_cluster_bootstrap_diff(v, v, c, replicates=300)
    assert r["mean_diff"] == 0.0 and r["ci_lower"] == 0.0 and r["ci_upper"] == 0.0


def test_clear_improvement_has_interval_above_zero():
    rng = np.random.default_rng(2)
    base = rng.random(300)
    better = base + 0.1 + rng.normal(0, 0.01, 300)
    c = np.repeat(np.arange(100), 3)
    r = paired_cluster_bootstrap_diff(better, base, c, replicates=500)
    assert r["ci_lower"] > 0.0


def test_interval_coverage_is_close_to_nominal():
    rng = np.random.default_rng(3)
    true_mean, hits, reps = 0.5, 0, 60
    for rep in range(reps):
        cluster_effect = rng.normal(0, 0.1, 100)
        values = true_mean + np.repeat(cluster_effect, 3) + rng.normal(0, 0.1, 300)
        clusters = np.repeat(np.arange(100), 3)
        r = cluster_bootstrap_mean(values, clusters, replicates=400, seed=rep)
        hits += r["ci_lower"] <= true_mean <= r["ci_upper"]
    assert hits / reps >= 0.85


def test_clustering_widens_the_interval_when_groups_within_a_product_are_correlated():
    rng = np.random.default_rng(4)
    cluster_effect = rng.normal(0, 1.0, 50)
    values = np.repeat(cluster_effect, 10) + rng.normal(0, 0.1, 500)
    clusters = np.repeat(np.arange(50), 10)
    clustered = cluster_bootstrap_mean(values, clusters, replicates=500)
    naive = cluster_bootstrap_mean(values, np.arange(500), replicates=500)
    assert (clustered["ci_upper"] - clustered["ci_lower"]) > 2 * (naive["ci_upper"] - naive["ci_lower"])


def test_misaligned_inputs_rejected():
    with pytest.raises(ValueError):
        cluster_bootstrap_mean([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        paired_cluster_bootstrap_diff([1, 2], [1], [1, 2])
