import numpy as np
import pandas as pd
import pytest

from reviewrank.evaluate import per_group_metrics


def rows(grades_by_group):
    recs = []
    for gid, grades in grades_by_group.items():
        for i, g in enumerate(grades):
            recs.append({"group_id": gid, "parent_asin": gid.split("|")[0], "grade": g})
    return pd.DataFrame(recs)


def test_ideal_scores_give_ndcg_one():
    df = rows({"P1|2020H2": [3, 0, 2, 1, 0]})
    scores = df["grade"].to_numpy(float)  # score equals relevance -> ideal order
    out = per_group_metrics(df, scores)
    assert out["ndcg_at_3"].iloc[0] == pytest.approx(1.0)
    assert out["ndcg_at_5"].iloc[0] == pytest.approx(1.0)
    assert out["precision_at_3"].iloc[0] == pytest.approx(1.0)


def test_reversed_scores_are_worse_than_ideal():
    df = rows({"P1|2020H2": [3, 0, 2, 1, 0]})
    good = per_group_metrics(df, df["grade"].to_numpy(float))["ndcg_at_3"].iloc[0]
    bad = per_group_metrics(df, -df["grade"].to_numpy(float))["ndcg_at_3"].iloc[0]
    assert bad < good


def test_known_value_for_a_fixed_ordering():
    # scores force the order [index0, index2, index1, ...]: grades [3, 2, 0, 1, 0]; k=3
    df = rows({"P1|2020H2": [3, 0, 2, 1, 0]})
    scores = np.array([5.0, 3.0, 4.0, 2.0, 1.0])
    out = per_group_metrics(df, scores)
    # ranked grades [3, 2, 0, 1, 0]: DCG@3 = 7 + 3/log2(3) + 0 ; ideal [3,2,1]: 7 + 3/log2(3) + 1/2
    import math

    dcg = 7 + 3 / math.log2(3)
    ideal = dcg + 0.5
    assert out["ndcg_at_3"].iloc[0] == pytest.approx(dcg / ideal)


def test_multiple_groups_and_parent_asin_are_returned():
    df = rows({"P1|2020H2": [1, 0, 0, 0, 0], "P2|2020H2": [0, 2, 0, 0, 0]})
    out = per_group_metrics(df, np.arange(len(df), dtype=float))
    assert sorted(out["group_id"]) == ["P1|2020H2", "P2|2020H2"]
    assert set(out["parent_asin"]) == {"P1", "P2"}


def test_ties_are_deterministic_for_the_same_seed():
    df = rows({"P1|2020H2": [3, 0, 2, 1, 0]})
    same = np.zeros(5)
    a = per_group_metrics(df, same, seed=11)["ndcg_at_3"].iloc[0]
    b = per_group_metrics(df, same, seed=11)["ndcg_at_3"].iloc[0]
    assert a == b


def test_group_without_relevant_item_raises():
    df = rows({"P1|2020H2": [0, 0, 0, 0, 0]})
    with pytest.raises(ValueError):
        per_group_metrics(df, np.zeros(5))


def test_misaligned_scores_raise():
    df = rows({"P1|2020H2": [1, 0, 0, 0, 0]})
    with pytest.raises(ValueError):
        per_group_metrics(df, np.zeros(3))
