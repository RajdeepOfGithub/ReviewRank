import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from reviewrank.features import FEATURE_COLUMNS
from reviewrank.scoring import load_model, score_reviews
from reviewrank.serve import create_app

CONTENT = [c for c in FEATURE_COLUMNS if c not in ("prior_reviews", "month", "hour")]


@pytest.fixture
def paths(tmp_path):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, len(CONTENT))), columns=CONTENT)
    y = x.iloc[:, 0] + rng.normal(size=300)
    booster = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 4, "seed": 1}, lgb.Dataset(x, y), 10)
    mp, mf = tmp_path / "m.txt", tmp_path / "m.json"
    booster.save_model(str(mp))
    mf.write_text(json.dumps({"run_id": "TEST-RUN", "primary": "content_only", "feature_columns": CONTENT, "uses_tfidf_score": False, "status": "test", "lightgbm_params": {"num_leaves": 4}}))
    return mp, mf


@pytest.fixture
def client(paths):
    return TestClient(create_app(str(paths[0]), str(paths[1])))


def review(i, text="a reasonably long review text " * 3, rating=4.0, **extra):
    return {"id": f"r{i}", "text": text, "rating": rating, **extra}


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["run_id"] == "TEST-RUN" and info["n_features"] == len(CONTENT)
    assert "month" not in info["feature_columns"] and "prior_reviews" not in info["feature_columns"]


def test_rank_returns_consecutive_ranks_sorted_by_score_and_is_deterministic(client):
    body = {
        "product_history": {"n_reviews": 5, "mean_rating": 4.2, "mean_words": 30.0},
        "reviews": [review(1, "short"), review(2, "a much longer and more detailed review " * 6, rating=2.0), review(3, "medium length review text here")],
    }
    r1 = client.post("/rank", json=body).json()
    r2 = client.post("/rank", json=body).json()
    assert r1 == r2
    ranks = [x["rank"] for x in r1["ranked"]]
    scores = [x["score"] for x in r1["ranked"]]
    assert ranks == [1, 2, 3] and scores == sorted(scores, reverse=True)
    assert {x["id"] for x in r1["ranked"]} == {"r1", "r2", "r3"}
    assert r1["model_run_id"] == "TEST-RUN"


def test_api_scores_match_the_scoring_module(client, paths):
    reviews = [review(1, "one two three"), review(2, "a much longer review " * 5, rating=2.0)]
    prior = {"n_reviews": 3, "mean_rating": 3.5, "mean_words": 12.0}
    api = {x["id"]: x["score"] for x in client.post("/rank", json={"product_history": prior, "reviews": reviews}).json()["ranked"]}
    model = load_model(*paths)
    direct = score_reviews(model, reviews, prior)
    assert api["r1"] == pytest.approx(direct[0], abs=1e-9) and api["r2"] == pytest.approx(direct[1], abs=1e-9)


@pytest.mark.parametrize("bad", ["helpful_vote", "votes", "user_id"])
def test_unknown_or_vote_fields_are_rejected(client, bad):
    assert client.post("/rank", json={"reviews": [review(1, **{bad: 5})]}).status_code == 422


def test_unknown_top_level_and_history_fields_are_rejected(client):
    assert client.post("/rank", json={"reviews": [review(1)], "helpful_vote": 1}).status_code == 422
    assert client.post("/rank", json={"reviews": [review(1)], "product_history": {"n_reviews": 1, "total_votes": 9}}).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"reviews": []},
        {"reviews": [{"id": "a", "rating": 4}]},                         # missing text
        {"reviews": [review(1, rating=0.5)]},                            # rating out of range
        {"reviews": [review(1, rating=6)]},
        {"reviews": [review(1), review(1)]},                             # duplicate ids
        {"reviews": [review(i) for i in range(501)]},                    # too many
        {"reviews": [review(1, text="x" * 20001)]},                      # text too long
        {"reviews": [review(1)], "product_history": {"n_reviews": -1}},
    ],
)
def test_malformed_requests_get_422(client, body):
    assert client.post("/rank", json=body).status_code == 422


def test_product_history_is_optional(client):
    assert client.post("/rank", json={"reviews": [review(1)]}).status_code == 200
