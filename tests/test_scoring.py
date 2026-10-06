import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from reviewrank.features import FEATURE_COLUMNS, build_features
from reviewrank.leakage import LeakageError
from reviewrank.scoring import load_model, rank_order, request_features, score_reviews

CONTENT = [c for c in FEATURE_COLUMNS if c not in ("prior_reviews", "month", "hour")]


def ms(date):
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def tiny_model(tmp_path, columns=None, uses_tfidf=False):
    cols = columns or CONTENT
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, len(cols))), columns=cols)
    y = x.iloc[:, 0] * 2 + rng.normal(size=300)
    booster = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 4, "seed": 1}, lgb.Dataset(x, y), 10)
    model_path, manifest_path = tmp_path / "m.txt", tmp_path / "m.json"
    booster.save_model(str(model_path))
    manifest_path.write_text(json.dumps({"run_id": "T", "feature_columns": cols, "uses_tfidf_score": uses_tfidf, "status": "test"}))
    return model_path, manifest_path


def history():
    texts = ["One two three.", "A longer review! It has more words, and a number 5?", "Short", "Brand new review text here, with detail."]
    rows = []
    for i, (t, r) in enumerate(zip(texts, [5.0, 3.0, 1.0, 4.0])):
        rows.append({
            "parent_asin": "P1", "timestamp_ms": ms(f"2019-01-0{i + 1}"), "rating": r, "title": f"Title {i}",
            "text": t, "images_json": '[{"large_image_url": "x"}]' if i == 3 else "[]", "verified_purchase": i != 2,
            "words": len(t.split()),
        })
    return pd.DataFrame(rows)


def test_request_features_match_the_batch_pipeline_for_the_same_review():
    df = history()
    batch = build_features(df).iloc[3][CONTENT]
    prior = {"n_reviews": 3, "mean_rating": float(df["rating"].iloc[:3].mean()), "mean_words": float(df["words"].iloc[:3].mean())}
    new = df.iloc[3]
    served = request_features(
        [{"text": new["text"], "title": new["title"], "rating": new["rating"], "verified_purchase": bool(new["verified_purchase"]), "image_count": 1}],
        prior,
    ).iloc[0][CONTENT]
    pd.testing.assert_series_equal(served.astype(float), batch.astype(float), check_names=False)


def test_missing_product_history_gives_nan_prior_features():
    f = request_features([{"text": "hello world", "rating": 4}], None).iloc[0]
    assert np.isnan(f["prior_mean_rating"]) and np.isnan(f["words_vs_prior_mean"])
    f0 = request_features([{"text": "hello world", "rating": 4}], {"n_reviews": 0, "mean_rating": 5.0, "mean_words": 9.0}).iloc[0]
    assert np.isnan(f0["prior_mean_rating"])


def test_score_reviews_returns_one_score_per_review_and_ranking_is_stable(tmp_path):
    mp, mf = tiny_model(tmp_path)
    model = load_model(mp, mf)
    reviews = [{"text": "a b c", "rating": 5}, {"text": "a b c", "rating": 5}, {"text": "much longer text " * 20, "rating": 2}]
    scores = score_reviews(model, reviews, {"n_reviews": 2, "mean_rating": 4.0, "mean_words": 10.0})
    assert scores.shape == (3,)
    assert np.allclose(scores, score_reviews(model, reviews, {"n_reviews": 2, "mean_rating": 4.0, "mean_words": 10.0}))
    assert rank_order(np.array([1.0, 3.0, 3.0, 2.0])).tolist() == [1, 2, 3, 0]  # ties keep input order


def test_load_model_rejects_vote_like_feature_names(tmp_path):
    mp, mf = tiny_model(tmp_path, columns=CONTENT[:-1] + ["helpful_ratio"])
    with pytest.raises(LeakageError):
        load_model(mp, mf)


def test_load_model_rejects_tfidf_stacked_manifest(tmp_path):
    mp, mf = tiny_model(tmp_path, uses_tfidf=True)
    with pytest.raises(ValueError):
        load_model(mp, mf)


def test_model_asking_for_unbuildable_feature_is_reported(tmp_path):
    mp, mf = tiny_model(tmp_path, columns=CONTENT[:-1] + ["month"])
    model = load_model(mp, mf)
    with pytest.raises(ValueError):
        score_reviews(model, [{"text": "x", "rating": 3}], None)
