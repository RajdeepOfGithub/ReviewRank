import numpy as np
import pandas as pd

from reviewrank.baselines import (
    TfidfRidge,
    score_longest_first,
    score_lowest_rating_first,
    score_newest_first,
    score_oldest_first,
    score_random,
)


def toy(n=60):
    rng = np.random.default_rng(0)
    helpful = ["detailed review with great explanation of how it works"] * (n // 2)
    useless = ["bad", "ok", "meh"] * (n // 6)
    texts = helpful + useless
    votes = [5] * (n // 2) + [0] * len(useless)
    return pd.DataFrame(
        {
            "title": ["t"] * len(texts),
            "text": texts,
            "helpful_vote": votes,
            "rating": rng.integers(1, 6, len(texts)).astype(float),
            "timestamp_ms": np.arange(len(texts)) * 1000,
            "words": [len(t.split()) for t in texts],
        }
    )


def test_heuristic_scores_rank_in_the_documented_direction():
    df = pd.DataFrame(
        {"timestamp_ms": [1, 3, 2], "words": [10, 5, 50], "rating": [5.0, 1.0, 3.0]}
    )
    assert np.argmax(score_newest_first(df)) == 1
    assert np.argmax(score_oldest_first(df)) == 0
    assert np.argmax(score_longest_first(df)) == 2
    assert np.argmax(score_lowest_rating_first(df)) == 1


def test_random_scores_are_seeded():
    df = pd.DataFrame({"x": range(10)})
    assert np.array_equal(score_random(df, 5), score_random(df, 5))
    assert not np.array_equal(score_random(df, 5), score_random(df, 6))


def test_tfidf_ridge_learns_the_toy_signal_and_is_deterministic():
    df = toy()
    m1 = TfidfRidge(alpha=1.0, min_df=1).fit(df)
    m2 = TfidfRidge(alpha=1.0, min_df=1).fit(df)
    s = m1.score(df)
    assert np.allclose(s, m2.score(df))
    assert s[df["helpful_vote"] == 5].mean() > s[df["helpful_vote"] == 0].mean()


def test_scoring_does_not_use_the_vote_column():
    df = toy()
    model = TfidfRidge(alpha=1.0, min_df=1).fit(df)
    scored = model.score(df.drop(columns=["helpful_vote"]))
    assert len(scored) == len(df)
