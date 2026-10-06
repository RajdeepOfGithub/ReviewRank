import numpy as np
import pandas as pd

from reviewrank.evaluate import per_group_metrics
from reviewrank.ranker import (
    group_sizes,
    oof_tfidf_scores,
    predict,
    product_folds,
    sort_by_group,
    train_lambdarank,
)


def test_sort_by_group_and_sizes():
    df = pd.DataFrame({"group_id": ["b", "a", "b", "a", "a"], "x": [1, 2, 3, 4, 5]})
    s = sort_by_group(df)
    assert s["group_id"].tolist() == ["a", "a", "a", "b", "b"]
    assert group_sizes(s).tolist() == [3, 2]


def test_product_folds_put_each_product_in_one_fold_and_are_deterministic():
    products = pd.Series(["p%d" % (i % 37) for i in range(500)])
    f1 = product_folds(products, 5, seed=1)
    f2 = product_folds(products, 5, seed=1)
    assert np.array_equal(f1, f2)
    frame = pd.DataFrame({"p": products, "f": f1})
    assert (frame.groupby("p")["f"].nunique() == 1).all()
    assert set(f1) == {0, 1, 2, 3, 4}
    assert not np.array_equal(f1, product_folds(products, 5, seed=2))


def synthetic_groups(n_groups=200, size=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        signal = rng.normal(size=size)
        noise = rng.normal(size=size)
        grade = np.digitize(signal, [0.3, 1.0, 1.6])  # 0..3 driven by `signal`
        for i in range(size):
            rows.append({"group_id": f"P{g}|2020H2", "parent_asin": f"P{g}", "signal": signal[i], "noise": noise[i], "grade": grade[i]})
    return pd.DataFrame(rows)


def test_lambdarank_learns_a_planted_signal_better_than_noise():
    train = sort_by_group(synthetic_groups(300, seed=1))
    valid = sort_by_group(synthetic_groups(100, seed=2))
    cols = ["signal", "noise"]
    booster = train_lambdarank(
        train[cols], train["grade"], group_sizes(train), valid[cols], valid["grade"], group_sizes(valid),
        params={"num_leaves": 7, "min_data_in_leaf": 10}, num_boost_round=100, early_stopping_rounds=10,
    )
    valid = valid.assign(_pos=(valid["grade"] >= 1).astype(int))
    valid = valid[valid.groupby("group_id")["_pos"].transform("sum") >= 1].reset_index(drop=True)  # eligible groups only
    scores = predict(booster, valid[cols])
    pg = per_group_metrics(valid, scores)
    random_scores = np.random.default_rng(0).random(len(valid))
    baseline = per_group_metrics(valid, random_scores)
    assert pg["ndcg_at_3"].mean() > baseline["ndcg_at_3"].mean() + 0.1
    imp = dict(zip(cols, booster.feature_importance("gain")))
    assert imp["signal"] > imp["noise"]


def test_training_is_deterministic():
    train = sort_by_group(synthetic_groups(100, seed=3))
    valid = sort_by_group(synthetic_groups(40, seed=4))
    cols = ["signal", "noise"]

    def fit():
        return train_lambdarank(
            train[cols], train["grade"], group_sizes(train), valid[cols], valid["grade"], group_sizes(valid),
            params={"num_leaves": 7, "min_data_in_leaf": 10}, num_boost_round=30, early_stopping_rounds=5,
        )

    assert np.allclose(predict(fit(), valid[cols]), predict(fit(), valid[cols]))


def test_oof_scores_are_not_in_sample_scores():
    # Toy text task where memorising the training row would give a perfect score.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "parent_asin": [f"p{i % 20}" for i in range(n)],
        "title": ["t"] * n,
        "text": [f"unique token{i} words here" for i in range(n)],
        "helpful_vote": rng.integers(0, 6, n),
    })
    oof = oof_tfidf_scores(df, alpha=0.1, n_splits=4, seed=0)
    assert len(oof) == n and np.isfinite(oof).all()
    from reviewrank.baselines import TfidfRidge

    in_sample = TfidfRidge(alpha=0.1, min_df=1).fit(df).score(df)
    target = np.log1p(df["helpful_vote"].to_numpy(float))
    assert np.corrcoef(in_sample, target)[0, 1] > np.corrcoef(oof, target)[0, 1]
