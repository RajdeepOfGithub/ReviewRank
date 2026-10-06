import pandas as pd
import pytest

from reviewrank.clean import clean_reviews, normalize_text

CFG = {
    "train_start": "2015-01-01",
    "train_end": "2020-07-01",
    "validation_end": "2021-01-01",
    "test_end": "2022-01-01",
}


def ms(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp() * 1000)


def row(i, date, text="a perfectly ordinary review text here", user=None, parent="P1", asin=None, votes=0):
    return {
        "asin": asin or f"A{i}",
        "parent_asin": parent,
        "user_id": user or f"U{i}",
        "rating": 5.0,
        "title": "t",
        "text": text,
        "images_json": "[]",
        "timestamp_ms": ms(date),
        "helpful_vote": votes,
        "verified_purchase": True,
    }


def frame(rows):
    return pd.DataFrame(rows)


def test_exact_duplicates_removed_but_distinct_rows_kept():
    df = frame([
        row(1, "2018-01-01", user="U1"),
        row(1, "2018-01-01", user="U1"),            # exact duplicate of the first
        row(2, "2018-01-01", user="U1", parent="P2"),  # same user+time, different product
        row(3, "2018-02-01", user="U1"),             # same user+product, different time
    ])
    out, counts = clean_reviews(df, CFG)
    assert counts["exact_duplicates_removed"] == 1
    assert counts["rows_out"] == 3


def test_reviews_before_train_start_are_excluded_from_training():
    df = frame([row(1, "2014-12-31"), row(2, "2015-01-01")])
    out, _ = clean_reviews(df, CFG)
    by_asin = out.set_index("asin")["split"]
    assert by_asin["A1"] == "excluded"
    assert by_asin["A2"] == "train"


def test_text_seen_in_train_flags_only_validation_and_test():
    long_text = "This is a long enough review text for overlap detection."
    df = frame([
        row(1, "2018-01-01", text=long_text),                      # train
        row(2, "2021-03-01", text="  THIS is a long   enough review text for overlap detection. "),  # test, same text
        row(3, "2020-08-01", text="Totally different words in this validation review."),
        row(4, "2021-04-01", text="great"),                          # short text never flagged
        row(5, "2018-05-01", text="great"),
    ])
    out, counts = clean_reviews(df, CFG)
    flagged = out.set_index("asin")["text_seen_in_train"]
    assert bool(flagged["A2"]) is True
    assert not flagged["A1"] and not flagged["A3"] and not flagged["A4"] and not flagged["A5"]
    assert counts["text_seen_in_train_per_split"] == {"test": 1}


def test_votes_are_preserved_and_never_used_for_flags():
    df = frame([row(1, "2018-01-01", votes=7), row(2, "2021-03-01", votes=0)])
    out, _ = clean_reviews(df, CFG)
    assert out.set_index("asin")["helpful_vote"].to_dict() == {"A1": 7, "A2": 0}


def test_word_counts():
    df = frame([row(1, "2018-01-01", text="one two  three"), row(2, "2018-01-02", text="")])
    out, _ = clean_reviews(df, CFG)
    assert out.set_index("asin")["words"].to_dict() == {"A1": 3, "A2": 0}


def test_normalize_text_collapses_case_and_whitespace():
    s = pd.Series(["  Hello   WORLD \n"])
    assert normalize_text(s).iloc[0] == "hello world"
