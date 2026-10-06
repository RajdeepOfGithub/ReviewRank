"""FastAPI service: rank newly posted reviews on one product page.

Run locally:  uvicorn reviewrank.serve:app_from_env --factory --port 8000
Env: REVIEWRANK_MODEL (default artifacts/models/lgbm_primary.txt),
     REVIEWRANK_MANIFEST (default artifacts/manifests/primary_model.json)

The request schema forbids unknown fields, so vote data (or anything else) cannot be sent.
Scores are for ranking reviews against each other on one product page. They are model outputs from a
temporal backtest on a helpful-vote proxy, not a measure of a review's truthfulness.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .scoring import LoadedModel, load_model, rank_order, score_reviews


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(..., min_length=1, max_length=200)
    title: str = Field("", max_length=1000)
    text: str = Field(..., max_length=20000)
    rating: float = Field(..., ge=1, le=5)
    verified_purchase: bool = True
    image_count: int = Field(0, ge=0, le=100)


class ProductHistory(BaseModel):
    """Summary of the product's EARLIER reviews at posting time."""

    model_config = ConfigDict(extra="forbid")
    n_reviews: int = Field(0, ge=0)
    mean_rating: float | None = Field(None, ge=1, le=5)
    mean_words: float | None = Field(None, ge=0)


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_history: ProductHistory | None = None
    reviews: list[ReviewIn] = Field(..., min_length=1, max_length=500)

    @field_validator("reviews")
    @classmethod
    def ids_unique(cls, v: list[ReviewIn]) -> list[ReviewIn]:
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("review ids must be unique")
        return v


class RankedReview(BaseModel):
    id: str
    rank: int
    score: float


class RankResponse(BaseModel):
    model_run_id: str
    model_status: str
    ranked: list[RankedReview]


def create_app(model_path: str, manifest_path: str) -> FastAPI:
    app = FastAPI(title="ReviewRank", version="0.0.1")
    model: LoadedModel = load_model(model_path, manifest_path)
    app.state.model = model

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info() -> dict:
        m = model.manifest
        return {
            "run_id": m.get("run_id"),
            "primary": m.get("primary"),
            "status": m.get("status"),
            "n_features": len(model.feature_columns),
            "feature_columns": model.feature_columns,
            "lightgbm_params": m.get("lightgbm_params"),
            "notes": [
                "Ranks reviews on one product page by predicted helpful-vote outcome (temporal backtest, vote proxy).",
                "Not a measure of truthfulness. Exposure/age features are excluded by design.",
                "Request fields are restricted; vote data is rejected.",
            ],
        }

    @app.post("/rank", response_model=RankResponse)
    def rank(req: RankRequest) -> RankResponse:
        reviews = [r.model_dump() for r in req.reviews]
        prior = req.product_history.model_dump() if req.product_history else None
        scores = score_reviews(model, reviews, prior)
        order = rank_order(scores)
        ranked = [RankedReview(id=reviews[i]["id"], rank=pos + 1, score=float(scores[i])) for pos, i in enumerate(order)]
        return RankResponse(model_run_id=str(model.manifest.get("run_id")), model_status=str(model.manifest.get("status")), ranked=ranked)

    return app


def app_from_env() -> FastAPI:
    return create_app(
        os.environ.get("REVIEWRANK_MODEL", "artifacts/models/lgbm_primary.txt"),
        os.environ.get("REVIEWRANK_MANIFEST", "artifacts/manifests/primary_model.json"),
    )
