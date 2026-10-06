"""Score NEW reviews at posting time with a trained content-only model.

The features are built with the same functions as the batch pipeline (`features.py`), so API and
batch results match (tested). The product's prior-review summary (count, mean rating, mean words) is
supplied by the caller because the as-of features need product history. No vote field is accepted,
and exposure/age features (`prior_reviews`, month, hour) are not used by the primary model (ADR-019).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import media_meta_features, text_features
from .leakage import assert_no_forbidden_columns

FORBIDDEN_SUBSTRINGS = ("helpful", "vote")


@dataclass
class LoadedModel:
    booster: lgb.Booster
    feature_columns: list[str]
    manifest: dict


def load_model(model_path: str | Path, manifest_path: str | Path) -> LoadedModel:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("uses_tfidf_score"):
        raise ValueError("serving supports content-only models; this manifest uses a stacked TF-IDF score")
    columns = list(manifest["feature_columns"])
    assert_no_forbidden_columns(columns, FORBIDDEN_SUBSTRINGS)
    return LoadedModel(lgb.Booster(model_file=str(model_path)), columns, manifest)


def _images_json(n: int) -> str:
    return "[" + ",".join(['{"large_image_url": "x"}'] * int(n)) + "]"


def request_features(reviews: list[dict], product_prior: dict | None = None) -> pd.DataFrame:
    """Feature table for new reviews (one row each). `product_prior`: n_reviews, mean_rating, mean_words."""
    df = pd.DataFrame(
        {
            "title": [r.get("title") or "" for r in reviews],
            "text": [r["text"] for r in reviews],
            "rating": [float(r["rating"]) for r in reviews],
            "verified_purchase": [bool(r.get("verified_purchase", True)) for r in reviews],
            "images_json": [_images_json(r.get("image_count", 0)) for r in reviews],
        }
    )
    df["words"] = df["text"].str.split().str.len().fillna(0).astype(int)
    out = pd.concat([text_features(df), media_meta_features(df)], axis=1)

    prior_rating = prior_words = np.nan
    if product_prior and (product_prior.get("n_reviews") or 0) > 0:
        if product_prior.get("mean_rating") is not None:
            prior_rating = float(product_prior["mean_rating"])
        if product_prior.get("mean_words") is not None:
            prior_words = float(product_prior["mean_words"])
    out["prior_mean_rating"] = prior_rating
    out["rating_minus_prior_mean"] = out["rating"] - prior_rating
    out["prior_mean_words"] = prior_words
    out["words_vs_prior_mean"] = out["words"] - prior_words
    return out


def score_reviews(model: LoadedModel, reviews: list[dict], product_prior: dict | None = None) -> np.ndarray:
    feats = request_features(reviews, product_prior)
    missing = [c for c in model.feature_columns if c not in feats.columns]
    if missing:
        raise ValueError(f"model needs features this service cannot build: {missing}")
    return model.booster.predict(feats[model.feature_columns])


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Indices from best to worst; ties keep input order (deterministic)."""
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))
