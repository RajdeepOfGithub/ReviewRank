"""Baselines (docs/planning/01_ROADMAP.md). Scoring functions take a DataFrame of reviews and
return one score per row (higher = ranked earlier). None of them reads `helpful_vote` at
scoring time; the TF-IDF model reads it only as its TRAINING target.

- random:              seeded random scores
- newest_first:        later timestamp first
- longest_first:       more words first (EDA-02: strongest simple signal)
- lowest_rating_first: lower star rating first (EDA-05: 1 star 38.2% vs 5 star 28.1% with >=1 vote)
- tfidf_ridge:         Ridge regression on TF-IDF of title + text, target log1p(votes), trained on
                       the training split only
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


def score_random(rows: pd.DataFrame, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(len(rows))


def score_newest_first(rows: pd.DataFrame) -> np.ndarray:
    return rows["timestamp_ms"].to_numpy(dtype=float)


def score_oldest_first(rows: pd.DataFrame) -> np.ndarray:
    """Earlier posting first: captures the exposure/first-mover advantage (EDA-03)."""
    return -rows["timestamp_ms"].to_numpy(dtype=float)


def score_longest_first(rows: pd.DataFrame) -> np.ndarray:
    return rows["words"].to_numpy(dtype=float)


def score_lowest_rating_first(rows: pd.DataFrame) -> np.ndarray:
    return -rows["rating"].to_numpy(dtype=float)


def _text(rows: pd.DataFrame) -> pd.Series:
    return rows["title"].fillna("") + " " + rows["text"].fillna("")


class TfidfRidge:
    """TF-IDF + Ridge on log1p(votes). Fit on the training split only."""

    def __init__(self, alpha: float = 3.0, max_features: int = 300_000, min_df: int = 5):
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=min_df, max_features=max_features, sublinear_tf=True, lowercase=True
        )
        self.model = Ridge(alpha=alpha)

    def fit(self, train: pd.DataFrame) -> "TfidfRidge":
        x = self.vectorizer.fit_transform(_text(train))
        y = np.log1p(train["helpful_vote"].to_numpy(dtype=float))
        self.model.fit(x, y)
        return self

    def score(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.vectorizer.transform(_text(rows)))
