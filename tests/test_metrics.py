import math

import pytest

from reviewrank.metrics import (
    dcg_at_k,
    mean_ndcg_with_coverage,
    ndcg_at_k,
    order_by_score,
    precision_at_k,
)


def test_dcg_matches_hand_calculation():
    # ranked relevances [3, 0, 2, 1, 0], k=3
    # (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4) = 7 + 0 + 1.5 = 8.5
    assert dcg_at_k([3, 0, 2, 1, 0], 3) == pytest.approx(8.5)


def test_ndcg_matches_hand_calculation():
    # ideal order [3, 2, 1]: 7 + 3/log2(3) + 1/log2(4)
    ideal = 7 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k([3, 0, 2, 1, 0], 3) == pytest.approx(8.5 / ideal)


def test_ndcg_is_one_for_perfect_order():
    assert ndcg_at_k([3, 2, 1, 0], 3) == pytest.approx(1.0)


def test_ndcg_undefined_when_no_relevant_item():
    assert ndcg_at_k([0, 0, 0, 0], 3) is None


def test_k_larger_than_list_is_allowed():
    assert ndcg_at_k([2, 1], 5) == pytest.approx(1.0)


def test_negative_relevance_rejected():
    with pytest.raises(ValueError):
        dcg_at_k([1, -1], 2)


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        dcg_at_k([1, 2], 0)


def test_precision_at_k():
    assert precision_at_k([3, 0, 2, 0], 3) == pytest.approx(2 / 3)


def test_order_by_score_descending():
    order = order_by_score([0.1, 0.9, 0.5], seed=0)
    assert list(order) == [1, 2, 0]


def test_order_by_score_ties_are_deterministic_and_seed_dependent():
    scores = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    a = list(order_by_score(scores, seed=7))
    b = list(order_by_score(scores, seed=7))
    assert a == b
    assert sorted(a) == list(range(6))
    assert any(list(order_by_score(scores, seed=s)) != a for s in range(20))


def test_mean_ndcg_reports_coverage():
    mean, coverage = mean_ndcg_with_coverage([1.0, None, 0.5, None])
    assert mean == pytest.approx(0.75)
    assert coverage == pytest.approx(0.5)


def test_mean_ndcg_all_undefined_raises():
    with pytest.raises(ValueError):
        mean_ndcg_with_coverage([None, None])
