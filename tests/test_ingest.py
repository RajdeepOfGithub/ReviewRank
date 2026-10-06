import json

import duckdb
import pytest

from reviewrank.ingest import PARQUET_COLUMNS, RowCountError, count_lines, jsonl_to_parquet


def review(i, **over):
    row = {
        "rating": 5.0,
        "title": f"t{i}",
        "text": f"text {i} with \"quotes\", commas and é unicode",
        "images": [],
        "asin": f"A{i}",
        "parent_asin": "P1",
        "user_id": f"U{i}",
        "timestamp": 1580950175902 + i,
        "helpful_vote": i,
        "verified_purchase": True,
    }
    row.update(over)
    return row


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_round_trip_types_and_values(tmp_path):
    rows = [
        review(1),
        review(2, images=[{"small_image_url": "u", "attachment_type": "IMAGE"}], verified_purchase=False),
        review(3, text="", helpful_vote=0),
    ]
    src, out = tmp_path / "r.jsonl", tmp_path / "r.parquet"
    write_jsonl(src, rows)
    result = jsonl_to_parquet(src, out)

    assert result["rows"] == 3
    assert list(result["columns"]) == PARQUET_COLUMNS
    assert result["columns"]["timestamp_ms"] == "BIGINT"
    assert result["columns"]["verified_purchase"] == "BOOLEAN"
    assert result["columns"]["rating"] == "DOUBLE"

    got = duckdb.connect().execute(
        f"SELECT asin, timestamp_ms, helpful_vote, verified_purchase, text, images_json "
        f"FROM read_parquet('{out.as_posix()}') ORDER BY asin"
    ).fetchall()
    assert got[0][1] == 1580950175903
    assert got[1][3] is False
    assert got[2][4] == ""
    assert "small_image_url" in got[1][5]
    assert got[0][4] == rows[0]["text"]


def test_count_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [review(i) for i in range(5)])
    assert count_lines(p) == 5


def test_row_count_mismatch_is_detected(tmp_path, monkeypatch):
    src, out = tmp_path / "r.jsonl", tmp_path / "r.parquet"
    write_jsonl(src, [review(i) for i in range(3)])
    monkeypatch.setattr("reviewrank.ingest.count_lines", lambda p: 99)
    with pytest.raises(RowCountError):
        jsonl_to_parquet(src, out)
