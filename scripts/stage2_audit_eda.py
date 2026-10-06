"""Stage 2: data audit (label-free) and EDA (labels only from development rows).

Run from the repo root:  python scripts/stage2_audit_eda.py

Rules enforced here:
- Anything that reads `helpful_vote` goes through the `dev` view = reviews posted BEFORE the
  sealed test start (configs/splits.yaml). The sealed test year is never used with labels.
- Group-size checks use review counts only (no votes), so they may look at every period.
Outputs: reports/audit/audit.json, reports/eda/eda.json, reports/eda/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reviewrank.split import boundaries, load_split_config  # noqa: E402

PARQUET = (ROOT / "data" / "processed" / "reviews.parquet").as_posix()
OUT_AUDIT = ROOT / "reports" / "audit"
OUT_EDA = ROOT / "reports" / "eda"
SNAPSHOT_DATE = "2023-09-11"  # latest review timestamp in the file (Stage 1 manifest)

cfg = load_split_config(ROOT / "configs" / "splits.yaml")
train_end, val_end, test_end = (int(t.timestamp() * 1000) for t in boundaries(cfg))

WORDS = (
    "CASE WHEN length(trim(text)) = 0 THEN 0 "
    r"ELSE len(string_split(regexp_replace(trim(text), '\s+', ' ', 'g'), ' ')) END"
)
SPLIT = (
    f"CASE WHEN timestamp_ms < {train_end} THEN 'train' WHEN timestamp_ms < {val_end} THEN 'validation' "
    f"WHEN timestamp_ms < {test_end} THEN 'test' ELSE 'stream' END"
)

con = duckdb.connect()
con.execute(
    f"""
    CREATE VIEW r AS
    SELECT *, epoch_ms(timestamp_ms) AS ts, year(epoch_ms(timestamp_ms)) AS yr,
           {WORDS} AS words,
           (images_json IS NOT NULL AND images_json NOT IN ('[]', 'null')) AS has_image,
           {SPLIT} AS split
    FROM read_parquet('{PARQUET}')
    """
)
# Only this view may be used together with helpful_vote.
con.execute(f"CREATE VIEW dev AS SELECT * FROM r WHERE timestamp_ms < {val_end}")
assert con.execute("SELECT max(timestamp_ms) FROM dev").fetchone()[0] < val_end


def q(sql: str):
    return con.execute(sql).fetchdf()


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------------ audit (label-free)
audit: dict = {"snapshot_date": SNAPSHOT_DATE}

audit["rows"] = int(q("SELECT count(*) n FROM r").n[0])
audit["rows_per_year"] = q("SELECT yr, count(*) n FROM r GROUP BY yr ORDER BY yr").to_dict("records")
audit["rows_per_split"] = q("SELECT split, count(*) n FROM r GROUP BY split ORDER BY split").to_dict("records")

audit["exact_duplicate_key_user_product_timestamp"] = q(
    """SELECT coalesce(sum(c - 1), 0) AS extra_rows, count(*) AS keys FROM
       (SELECT count(*) c FROM r GROUP BY user_id, parent_asin, timestamp_ms HAVING count(*) > 1)"""
).to_dict("records")[0]

audit["same_user_same_product_multiple_reviews"] = q(
    """SELECT count(*) AS user_product_pairs, coalesce(sum(c), 0) AS rows_involved FROM
       (SELECT count(*) c FROM r GROUP BY user_id, parent_asin HAVING count(*) > 1)"""
).to_dict("records")[0]

audit["identical_text_across_reviews_min20chars"] = q(
    r"""WITH t AS (SELECT lower(trim(regexp_replace(text, '\s+', ' ', 'g'))) AS nt, user_id FROM r
                   WHERE length(trim(text)) >= 20)
        SELECT count(*) AS texts_repeated, coalesce(sum(c), 0) AS rows_involved, coalesce(max(c), 0) AS largest_repeat
        FROM (SELECT nt, count(*) c, count(DISTINCT user_id) u FROM t GROUP BY nt HAVING count(*) > 1)"""
).to_dict("records")[0]

audit["empty_text_rows"] = int(q("SELECT count(*) n FROM r WHERE length(trim(text)) = 0").n[0])
audit["words_percentiles"] = q(
    """SELECT quantile_cont(words, 0.5) p50, quantile_cont(words, 0.75) p75, quantile_cont(words, 0.9) p90,
              quantile_cont(words, 0.95) p95, quantile_cont(words, 0.99) p99, max(words) max,
              avg((words > 100)::INT) share_over_100_words, avg((words <= 3)::INT) share_3_words_or_fewer
       FROM r"""
).to_dict("records")[0]
audit["mostly_non_ascii_text_share"] = float(
    q(
        r"""SELECT avg((length(regexp_replace(text, '[\x00-\x7F]', '', 'g')) * 1.0 / length(text) > 0.3)::INT) v
            FROM r WHERE length(text) > 0"""
    ).v[0]
)

audit["reviews_per_product"] = q(
    """WITH p AS (SELECT parent_asin, count(*) c FROM r GROUP BY parent_asin)
       SELECT count(*) products, quantile_cont(c, 0.5) p50, quantile_cont(c, 0.75) p75, quantile_cont(c, 0.9) p90,
              quantile_cont(c, 0.99) p99, max(c) max,
              sum((c >= 5)::INT) products_ge5, sum((c >= 10)::INT) products_ge10,
              sum((c >= 20)::INT) products_ge20, sum((c >= 50)::INT) products_ge50
       FROM p"""
).to_dict("records")[0]
audit["share_of_reviews_in_top_1pct_products"] = float(
    q(
        """WITH p AS (SELECT parent_asin, count(*) c FROM r GROUP BY parent_asin),
                k AS (SELECT c, row_number() OVER (ORDER BY c DESC) rn, count(*) OVER () n FROM p)
           SELECT sum(c) * 1.0 / (SELECT sum(c) FROM p) v FROM k WHERE rn <= n * 0.01"""
    ).v[0]
)

# User overlap across the temporal boundaries (ids only, no votes).
audit["test_period_reviews_by_users_seen_in_train"] = q(
    f"""SELECT split, count(*) AS reviews,
               sum((user_id IN (SELECT user_id FROM r WHERE timestamp_ms < {train_end}))::INT) AS by_train_users
        FROM r WHERE split IN ('validation', 'test') GROUP BY split ORDER BY split"""
).to_dict("records")

# ------------------------------------------------------------------ group sizes (counts only)
WINDOWS = {
    "month": "date_trunc('month', ts)",
    "quarter": "date_trunc('quarter', ts)",
    "half_year": "make_date(year(ts), CASE WHEN month(ts) <= 6 THEN 1 ELSE 7 END, 1)",
    "year": "date_trunc('year', ts)",
}
group_sizes = {}
for name, expr in WINDOWS.items():
    group_sizes[name] = q(
        f"""WITH g AS (SELECT split, parent_asin, {expr} AS w, count(*) c FROM r GROUP BY ALL),
                 tot AS (SELECT split, count(*) n FROM r GROUP BY split)
            SELECT g.split, tot.n AS reviews, count(*) AS groups,
                   sum((c >= 2)::INT) groups_ge2, sum((c >= 3)::INT) groups_ge3,
                   sum((c >= 5)::INT) groups_ge5, sum((c >= 10)::INT) groups_ge10,
                   sum(CASE WHEN c >= 2 THEN c ELSE 0 END) * 1.0 / tot.n AS review_share_in_ge2,
                   sum(CASE WHEN c >= 5 THEN c ELSE 0 END) * 1.0 / tot.n AS review_share_in_ge5,
                   sum(CASE WHEN c >= 10 THEN c ELSE 0 END) * 1.0 / tot.n AS review_share_in_ge10
            FROM g JOIN tot USING (split) GROUP BY g.split, tot.n ORDER BY g.split"""
    ).to_dict("records")
audit["group_sizes_by_window_and_split_counts_only"] = group_sizes
dump(audit, OUT_AUDIT / "audit.json")

# ------------------------------------------------------------------ EDA (development rows only)
eda: dict = {"dev_rule": f"timestamp_ms < {val_end} (before sealed test start)"}

eda["dev_rows"] = int(q("SELECT count(*) n FROM dev").n[0])
eda["vote_by_year"] = q(
    """SELECT yr, count(*) n, avg((helpful_vote >= 1)::INT) share_ge1, avg((helpful_vote >= 3)::INT) share_ge3,
              avg(helpful_vote) mean_votes
       FROM dev GROUP BY yr ORDER BY yr"""
).to_dict("records")

ORDER_BUCKET = (
    "CASE WHEN rn = 1 THEN '1st' WHEN rn = 2 THEN '2nd' WHEN rn = 3 THEN '3rd' "
    "WHEN rn <= 5 THEN '4-5' WHEN rn <= 10 THEN '6-10' ELSE '11+' END"
)
eda["vote_by_order_within_product"] = q(
    f"""WITH o AS (SELECT helpful_vote, row_number() OVER (PARTITION BY parent_asin ORDER BY timestamp_ms, user_id) rn
                   FROM dev)
        SELECT {ORDER_BUCKET} AS bucket, count(*) n, avg((helpful_vote >= 1)::INT) share_ge1, avg(helpful_vote) mean_votes
        FROM o GROUP BY 1 ORDER BY min(rn)"""
).to_dict("records")

WORD_BUCKET = (
    "CASE WHEN words = 0 THEN '0' WHEN words <= 5 THEN '1-5' WHEN words <= 10 THEN '6-10' WHEN words <= 20 THEN '11-20' "
    "WHEN words <= 50 THEN '21-50' WHEN words <= 100 THEN '51-100' ELSE '101+' END"
)
eda["vote_by_words"] = q(
    f"""SELECT {WORD_BUCKET} AS bucket, count(*) n, avg((helpful_vote >= 1)::INT) share_ge1, avg(helpful_vote) mean_votes
        FROM dev GROUP BY 1 ORDER BY min(words)"""
).to_dict("records")
eda["vote_by_rating"] = q(
    """SELECT cast(rating AS INT) rating, count(*) n, avg((helpful_vote >= 1)::INT) share_ge1, avg(helpful_vote) mean_votes
       FROM dev GROUP BY 1 ORDER BY 1"""
).to_dict("records")
eda["vote_by_verified_and_image"] = q(
    """SELECT verified_purchase, has_image, count(*) n, avg((helpful_vote >= 1)::INT) share_ge1, avg(helpful_vote) mean_votes
       FROM dev GROUP BY 1, 2 ORDER BY 1, 2"""
).to_dict("records")
eda["vote_concentration"] = q(
    """WITH k AS (SELECT helpful_vote, row_number() OVER (ORDER BY helpful_vote DESC) rn, count(*) OVER () n FROM dev)
       SELECT (SELECT avg((helpful_vote = 0)::INT) FROM dev) AS share_zero_votes,
              sum(helpful_vote) * 1.0 / (SELECT sum(helpful_vote) FROM dev) AS top1pct_reviews_share_of_votes
       FROM k WHERE rn <= n * 0.01"""
).to_dict("records")[0]
eda["top_review_share_of_product_votes_products_ge5_with_votes"] = q(
    """WITH p AS (SELECT parent_asin, count(*) c, sum(helpful_vote) s, max(helpful_vote) m FROM dev GROUP BY parent_asin)
       SELECT count(*) products, quantile_cont(m * 1.0 / s, 0.5) median_top_share FROM p WHERE c >= 5 AND s > 0"""
).to_dict("records")[0]
dump(eda, OUT_EDA / "eda.json")

# ------------------------------------------------------------------ plots
OUT_EDA.mkdir(parents=True, exist_ok=True)


def bar(xs, ys, title, xlabel, ylabel, fname, pct=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([str(x) for x in xs], ys, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if pct:
        ax.set_ylim(0, max(ys) * 1.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(OUT_EDA / fname, dpi=130)
    plt.close(fig)


import matplotlib.ticker  # noqa: E402

rp = audit["rows_per_year"]
bar([d["yr"] for d in rp], [d["n"] for d in rp], "Reviews per year (all rows)", "Posting year", "Reviews", "01_reviews_per_year.png")
vy = eda["vote_by_year"]
bar([d["yr"] for d in vy], [d["share_ge1"] for d in vy], "Share of reviews with >=1 helpful vote, by posting year (dev rows)", "Posting year", "Share", "02_vote_share_by_year.png", pct=True)
vo = eda["vote_by_order_within_product"]
bar([d["bucket"] for d in vo], [d["share_ge1"] for d in vo], "Share with >=1 vote by order among the product's reviews (dev)", "Review order within product", "Share", "03_vote_share_by_order.png", pct=True)
vw = eda["vote_by_words"]
bar([d["bucket"] for d in vw], [d["share_ge1"] for d in vw], "Share with >=1 vote by review length (dev)", "Words", "Share", "04_vote_share_by_words.png", pct=True)
vr = eda["vote_by_rating"]
bar([d["rating"] for d in vr], [d["share_ge1"] for d in vr], "Share with >=1 vote by star rating (dev)", "Rating", "Share", "05_vote_share_by_rating.png", pct=True)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
names = list(group_sizes)
for i, split in enumerate(["validation", "test"]):
    vals = []
    for n in names:
        row = next(d for d in group_sizes[n] if d["split"] == split)
        vals.append(row["review_share_in_ge5"])
    ax.bar([x + i * 0.38 for x in range(len(names))], vals, width=0.38, label=split)
ax.set_xticks([x + 0.19 for x in range(len(names))])
ax.set_xticklabels(names)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Share of reviews in (product, window) groups with >=5 reviews\n(counts only, no votes)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_EDA / "06_group_coverage_ge5.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ follow-ups that drive decisions
follow: dict = {}

# (a) Are exact-key duplicates true copies? (dev rows only because votes are compared)
follow["exact_key_duplicates_dev"] = q(
    """WITH d AS (SELECT count(*) c, count(DISTINCT asin) asins, count(DISTINCT text) texts,
                        count(DISTINCT helpful_vote) votes
                 FROM dev GROUP BY user_id, parent_asin, timestamp_ms HAVING count(*) > 1)
       SELECT count(*) AS keys, avg((texts = 1)::INT) AS share_identical_text,
              avg((votes = 1)::INT) AS share_identical_votes, avg((asins = c)::INT) AS share_distinct_asin_variants
       FROM d"""
).to_dict("records")[0]

# (b) Text in validation/test that also appears in train (normalized, >=20 chars; no votes used).
follow["text_seen_in_train"] = q(
    r"""WITH n AS (SELECT split, lower(trim(regexp_replace(text, '\s+', ' ', 'g'))) AS nt FROM r
                   WHERE length(trim(text)) >= 20),
             tr AS (SELECT DISTINCT nt FROM n WHERE split = 'train')
        SELECT split, count(*) AS reviews, sum((nt IN (SELECT nt FROM tr))::INT) AS seen_in_train
        FROM n WHERE split IN ('validation', 'test') GROUP BY split ORDER BY split"""
).to_dict("records")

# (c) Candidate relevance grades by absolute votes: 0, 1, 2-4, 5+ (dev rows).
GRADES = (
    "avg((helpful_vote = 0)::INT) AS g0, avg((helpful_vote = 1)::INT) AS g1, "
    "avg((helpful_vote BETWEEN 2 AND 4)::INT) AS g2_4, avg((helpful_vote >= 5)::INT) AS g5plus"
)
follow["grade_distribution_dev_all_years"] = q(f"SELECT {GRADES} FROM dev").to_dict("records")[0]
follow["grade_distribution_dev_2015_plus"] = q(f"SELECT {GRADES} FROM dev WHERE yr >= 2015").to_dict("records")[0]

# (d) Among half-year product groups with >=5 reviews, how many have any vote-bearing review?
half = WINDOWS["half_year"]
follow["groups_ge5_with_votes_dev_by_window_year"] = q(
    f"""WITH g AS (SELECT {half} AS w, parent_asin, count(*) AS c, sum((helpful_vote >= 1)::INT) AS pos FROM dev GROUP BY ALL)
        SELECT year(w) AS yr, count(*) AS groups_ge5, avg((pos >= 1)::INT) AS share_with_1plus_voted,
               avg((pos >= 2)::INT) AS share_with_2plus_voted
        FROM g WHERE c >= 5 GROUP BY 1 ORDER BY 1"""
).to_dict("records")

# (e) Train-start candidates: rows and share with >=1 vote if training starts in 2015 vs earlier.
follow["rows_before_2015_in_train"] = int(q(f"SELECT count(*) n FROM r WHERE split = 'train' AND yr < 2015").n[0])
dump(follow, OUT_EDA / "followups.json")

print("audit ->", OUT_AUDIT / "audit.json")
print("eda   ->", OUT_EDA / "eda.json")
print("follow->", OUT_EDA / "followups.json")
