"""LightGBM LambdaRank training helpers (Stage 5).

- Rows must be sorted so each query group is contiguous (LightGBM requirement); use sort_by_group.
- The TF-IDF score used as a stacked feature is built OUT-OF-FOLD by product so a row's score never
  comes from a model that saw its own product (and therefore its own label).
- Early stopping uses the validation set; model selection is on validation only (the sealed test
  is untouched until the freeze).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .baselines import TfidfRidge

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [3],
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "lambdarank_truncation_level": 10,
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}


def sort_by_group(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.sort_values("group_id", kind="mergesort").reset_index(drop=True)


def group_sizes(rows_sorted: pd.DataFrame) -> np.ndarray:
    return rows_sorted.groupby("group_id", sort=False).size().to_numpy()


def product_folds(parent_asin: pd.Series, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Deterministic fold id per row; every product lives in exactly one fold."""
    unique = pd.unique(parent_asin)
    perm = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {p: int(perm[i] % n_splits) for i, p in enumerate(unique)}
    return parent_asin.map(fold_of).to_numpy()


def oof_tfidf_scores(train: pd.DataFrame, *, alpha: float, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold TF-IDF Ridge scores for every training row (folds split by product)."""
    folds = product_folds(train["parent_asin"], n_splits, seed)
    out = np.zeros(len(train))
    for k in range(n_splits):
        fit_mask, score_mask = folds != k, folds == k
        model = TfidfRidge(alpha=alpha).fit(train.loc[fit_mask])
        out[score_mask] = model.score(train.loc[score_mask])
    return out


def train_lambdarank(
    x_train,
    y_train,
    sizes_train,
    x_valid,
    y_valid,
    sizes_valid,
    *,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    p = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(x_train, label=y_train, group=sizes_train)
    valid_set = lgb.Dataset(x_valid, label=y_valid, group=sizes_valid, reference=train_set)
    return lgb.train(
        p,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )


def predict(booster: lgb.Booster, x) -> np.ndarray:
    return booster.predict(x, num_iteration=booster.best_iteration)
