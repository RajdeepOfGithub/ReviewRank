import numpy as np
import pytest

from reviewrank.monitor import psi, psi_level


def test_identical_distributions_have_near_zero_psi():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_same_distribution_different_sample_is_small():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_shifted_distribution_is_large():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(size=5000), rng.normal(1.5, 1.0, 5000)) > 0.25


def test_values_outside_the_reference_range_are_counted():
    ref = np.random.default_rng(3).normal(size=2000)
    assert psi(ref, np.full(500, 50.0)) > 0.25


def test_categorical_like_features_use_one_bin_per_value():
    ref = np.array([1] * 800 + [0] * 200)
    same = np.array([1] * 80 + [0] * 20)
    flipped = np.array([1] * 20 + [0] * 80)
    assert psi(ref, same) < 0.01
    assert psi(ref, flipped) > 0.5


def test_unseen_category_in_current_is_handled():
    ref = np.array([1, 2, 3] * 100)
    assert psi(ref, np.array([1, 2, 3, 4] * 50)) > 0.0


def test_nan_values_are_ignored_and_empty_input_raises():
    ref = np.array([1.0, 2.0, np.nan, 3.0] * 50)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        psi(ref, np.array([np.nan]))


def test_zero_inflated_counts_with_many_distinct_values_are_not_silently_zero():
    rng = np.random.default_rng(5)
    # 97% zeros plus counts 1..15 (more than 10 distinct values): quantile edges collapse to one point.
    ref = np.where(rng.random(20000) < 0.97, 0, rng.integers(1, 16, 20000))
    same = np.where(rng.random(5000) < 0.97, 0, rng.integers(1, 16, 5000))
    more_images = np.where(rng.random(5000) < 0.80, 0, rng.integers(1, 16, 5000))
    assert psi(ref, same) < 0.05
    assert psi(ref, more_images) > 0.25


def test_constant_reference_versus_a_different_constant_is_large():
    assert psi(np.zeros(1000), np.ones(1000), bins=10) > 1.0


def test_psi_level_labels():
    assert psi_level(0.05) == "small" and psi_level(0.15) == "moderate" and psi_level(0.4) == "large"
