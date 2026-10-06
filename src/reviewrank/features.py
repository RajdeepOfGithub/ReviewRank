"""Stage 4 features, computed AS OF POSTING TIME.

Rules (docs/problem_contract.md, leakage rules):
- `helpful_vote` is never read here. build_features selects only the columns it needs, so a
  frame that still carries votes cannot influence the output (tested).
- Product aggregates use only reviews posted STRICTLY BEFORE the row (earlier timestamp).
  Reviews with the same millisecond timestamp never see each other.
- Calendar features are month-of-year and hour only. The year/timestamp itself is excluded so
  the model cannot learn "older review => more votes" (review age at snapshot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import assert_no_forbidden_columns

FORBIDDEN_SUBSTRINGS = ("helpful", "vote")

FEATURE_GROUPS: dict[str, list[str]] = {
    "length": ["words", "chars", "title_chars", "title_words", "avg_word_len", "sentences", "unique_word_ratio"],
    "style": ["exclaims", "questions", "digits", "caps_ratio", "is_empty_text"],
    "media_meta": ["image_count", "has_image", "rating", "verified"],
    "asof_product": [
        "prior_reviews",
        "prior_mean_rating",
        "rating_minus_prior_mean",
        "prior_mean_words",
        "words_vs_prior_mean",
    ],
    "calendar": ["month", "hour"],
}
FEATURE_COLUMNS = [c for group in FEATURE_GROUPS.values() for c in group]

REQUIRED_COLUMNS = ["parent_asin", "timestamp_ms", "rating", "title", "text", "images_json", "verified_purchase", "words"]


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].fillna("")
    title = df["title"].fillna("")
    chars = text.str.len()
    words = df["words"].astype(float)
    tokens = text.str.lower().str.findall(r"\w+")

    out = pd.DataFrame(index=df.index)
    out["words"] = words
    out["chars"] = chars.astype(float)
    out["title_chars"] = title.str.len().astype(float)
    out["title_words"] = title.str.split().str.len().fillna(0).astype(float)
    out["avg_word_len"] = text.str.replace(r"\s+", "", regex=True).str.len() / words.where(words > 0)
    out["sentences"] = text.str.count(r"[.!?]+").astype(float)
    out["unique_word_ratio"] = tokens.apply(lambda t: len(set(t)) / len(t) if len(t) else np.nan)
    out["exclaims"] = text.str.count("!").astype(float)
    out["questions"] = text.str.count(r"\?").astype(float)
    out["digits"] = text.str.count(r"\d").astype(float)
    out["caps_ratio"] = text.str.count(r"[A-Z]") / chars.where(chars > 0)
    out["is_empty_text"] = (text.str.strip().str.len() == 0).astype(float)
    return out


def media_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["image_count"] = df["images_json"].fillna("[]").str.count("large_image_url").astype(float)
    out["has_image"] = (out["image_count"] > 0).astype(float)
    out["rating"] = df["rating"].astype(float)
    out["verified"] = df["verified_purchase"].astype(float)
    return out


def asof_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prior-review aggregates per product using only STRICTLY EARLIER timestamps."""
    base = df[["parent_asin", "timestamp_ms", "rating", "words"]].copy()
    base["_row"] = np.arange(len(base))

    per_ts = (
        base.groupby(["parent_asin", "timestamp_ms"], sort=True)
        .agg(n=("rating", "size"), r=("rating", "sum"), w=("words", "sum"))
        .reset_index()
    )
    cum = per_ts.groupby("parent_asin")[["n", "r", "w"]].cumsum()
    prior = per_ts[["parent_asin", "timestamp_ms"]].copy()
    n = (cum["n"] - per_ts["n"]).to_numpy(dtype=float)  # cumulative incl. this timestamp minus its own = strictly before
    sum_r = (cum["r"] - per_ts["r"]).to_numpy(dtype=float)
    sum_w = (cum["w"] - per_ts["w"]).to_numpy(dtype=float)
    safe_n = np.where(n > 0, n, 1.0)
    prior["prior_reviews"] = n
    prior["prior_mean_rating"] = np.where(n > 0, sum_r / safe_n, np.nan)
    prior["prior_mean_words"] = np.where(n > 0, sum_w / safe_n, np.nan)

    merged = base.merge(prior, on=["parent_asin", "timestamp_ms"], how="left").sort_values("_row")
    merged.index = df.index
    out = pd.DataFrame(index=df.index)
    out["prior_reviews"] = merged["prior_reviews"]
    out["prior_mean_rating"] = merged["prior_mean_rating"]
    out["rating_minus_prior_mean"] = merged["rating"] - merged["prior_mean_rating"]
    out["prior_mean_words"] = merged["prior_mean_words"]
    out["words_vs_prior_mean"] = merged["words"] - merged["prior_mean_words"]
    return out


def calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    out = pd.DataFrame(index=df.index)
    out["month"] = ts.dt.month.astype(float)
    out["hour"] = ts.dt.hour.astype(float)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns for features: {missing}")
    # Select only what is needed: votes (or anything else) in `df` cannot leak through.
    needed = df[REQUIRED_COLUMNS]
    out = pd.concat(
        [text_features(needed), media_meta_features(needed), asof_product_features(needed), calendar_features(needed)],
        axis=1,
    )[FEATURE_COLUMNS]
    assert_no_forbidden_columns(out.columns, FORBIDDEN_SUBSTRINGS)
    return out
