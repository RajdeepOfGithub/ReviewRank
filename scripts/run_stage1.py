"""Stage 1: acquire the pinned source file, verify it, convert to Parquet, write the manifest.

Run from the repo root:  python scripts/run_stage1.py
Idempotent: a verified cached download is reused; the manifest is rewritten each run.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewrank.acquire import fetch, sha256_file, write_manifest  # noqa: E402
from reviewrank.ingest import jsonl_to_parquet  # noqa: E402

REPO = "McAuley-Lab/Amazon-Reviews-2023"
REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"  # dataset repo commit, read 2026-09-30
PATH_IN_REPO = "raw/review_categories/Health_and_Personal_Care.jsonl"
# Size and sha256 published by the source itself (Hugging Face LFS metadata, read 2026-09-30).
EXPECTED_BYTES = 226990293
EXPECTED_SHA256 = "b532849dcf8e42910ce688ad423b3c1622ce860e7bf88a3641358bb89f345d02"

URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{PATH_IN_REPO}"
RAW = ROOT / "data" / "raw" / "Health_and_Personal_Care.jsonl"
PARQUET = ROOT / "data" / "processed" / "reviews.parquet"
MANIFEST = ROOT / "artifacts" / "manifests" / "source_manifest.json"


def main() -> None:
    status = fetch(URL, RAW, EXPECTED_BYTES, EXPECTED_SHA256)
    print("raw file:", status)

    parquet_info = jsonl_to_parquet(RAW, PARQUET)
    print("parquet rows:", parquet_info["rows"])

    manifest = {
        "dataset": "Amazon Reviews 2023 (McAuley Lab)",
        "category": "Health_and_Personal_Care",
        "hf_repo": REPO,
        "hf_revision": REVISION,
        "url": URL,
        "retrieval": status.get("retrieved_at_utc", "cached copy; see earlier manifest"),
        "manifest_written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_bytes": EXPECTED_BYTES,
        "raw_sha256": EXPECTED_SHA256,
        "raw_sha256_source": "Hugging Face LFS metadata; recomputed locally and matched",
        "rows": parquet_info["rows"],
        "parquet_columns": parquet_info["columns"],
        "parquet_sha256": sha256_file(PARQUET),
        "parquet_bytes": PARQUET.stat().st_size,
        "license": {
            "dataset_license_assigned": False,
            "note": (
                "No dataset license is stated on the project site, the Hugging Face card or its "
                "API metadata. The MIT license in the GitHub repo covers code, not data. Maintainers "
                "state they cannot assign a license and that academic, non-commercial use is permitted "
                "(HF discussion #1). Use: non-commercial portfolio/research; cite; do not redistribute "
                "raw data or review text."
            ),
            "citation": "Hou et al., Bridging Language and Items for Retrieval and Recommendation, arXiv:2403.03952 (2024)",
            "verified_on": "2026-09-30",
        },
        "snapshot_note": "Source states data spans May 1996 to Sep 2023; snapshot year 2023.",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": "none (repository has no commits yet at time of run)",
    }
    write_manifest(manifest, MANIFEST)
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
