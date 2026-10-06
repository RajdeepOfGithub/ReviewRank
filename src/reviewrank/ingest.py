"""Convert the raw JSONL reviews to typed Parquet with explicit columns.

Explicit column types avoid silent schema inference. The row count of the Parquet
file is checked against an independent line count of the JSONL; any difference
means rows were dropped and the run fails.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# Source column -> DuckDB type. `images` is a list of objects and is stored as a JSON string.
SOURCE_COLUMNS = {
    "asin": "VARCHAR",
    "parent_asin": "VARCHAR",
    "user_id": "VARCHAR",
    "rating": "DOUBLE",
    "title": "VARCHAR",
    "text": "VARCHAR",
    "images": "JSON",
    "timestamp": "BIGINT",  # milliseconds since epoch
    "helpful_vote": "BIGINT",
    "verified_purchase": "BOOLEAN",
}

PARQUET_COLUMNS = [
    "asin",
    "parent_asin",
    "user_id",
    "rating",
    "title",
    "text",
    "images_json",
    "timestamp_ms",
    "helpful_vote",
    "verified_purchase",
]


class RowCountError(RuntimeError):
    """Parquet row count differs from the JSONL line count."""


def count_lines(path: str | Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def _sql_path(p: str | Path) -> str:
    return str(Path(p).resolve()).replace("\\", "/").replace("'", "''")


def jsonl_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> dict:
    """Write typed Parquet; return {"rows": int, "columns": {name: type}}."""
    jsonl_path, parquet_path = Path(jsonl_path), Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    expected_rows = count_lines(jsonl_path)

    cols = ", ".join(f"'{k}': '{v}'" for k, v in SOURCE_COLUMNS.items())
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT asin, parent_asin, user_id, rating, title, text,
                   CAST(images AS VARCHAR) AS images_json,
                   "timestamp" AS timestamp_ms, helpful_vote, verified_purchase
            FROM read_json('{_sql_path(jsonl_path)}', format='newline_delimited',
                           columns={{{cols}}}, maximum_object_size=67108864)
        ) TO '{_sql_path(parquet_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(parquet_path)}')").fetchone()[0]
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(parquet_path)}')").fetchall()
    con.close()

    if rows != expected_rows:
        raise RowCountError(f"parquet rows {rows} != jsonl lines {expected_rows}")
    return {"rows": rows, "columns": {name: typ for name, typ, *_ in schema}}
