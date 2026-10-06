"""Idempotent, integrity-checked download of the source file.

Design rules (docs/planning/02_ARCHITECTURE.md):
- The download is pinned to a dataset repo revision so it is reproducible.
- The file is verified against the checksum the source itself publishes.
- A cached file is reused only if its size and sha256 both match.
- Raw data stays outside Git (data/raw/ is ignored).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class IntegrityError(RuntimeError):
    """Downloaded or cached file does not match the expected size or checksum."""


def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()


def check_file(path: str | Path, expected_bytes: int, expected_sha256: str) -> str:
    """Return the file's sha256 if size and checksum match, else raise IntegrityError."""
    path = Path(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise IntegrityError(f"{path.name}: size {size} != expected {expected_bytes}")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise IntegrityError(f"{path.name}: sha256 {actual} != expected {expected_sha256}")
    return actual


def fetch(url: str, dest: str | Path, expected_bytes: int, expected_sha256: str) -> dict:
    """Download url to dest unless a verified copy exists. Returns a small status dict."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        try:
            check_file(dest, expected_bytes, expected_sha256)
            return {"status": "cached", "path": str(dest)}
        except IntegrityError:
            dest.unlink()  # corrupt or partial earlier copy: replace it

    part = dest.with_suffix(dest.suffix + ".part")
    started = datetime.now(timezone.utc)
    with urllib.request.urlopen(url) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    try:
        check_file(part, expected_bytes, expected_sha256)
    except IntegrityError:
        part.unlink(missing_ok=True)
        raise
    part.replace(dest)
    return {
        "status": "downloaded",
        "path": str(dest),
        "retrieved_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
