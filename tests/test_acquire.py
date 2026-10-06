import hashlib
import json

import pytest

from reviewrank.acquire import (
    IntegrityError,
    check_file,
    fetch,
    read_manifest,
    sha256_file,
    write_manifest,
)


def make_source(tmp_path, content=b"hello reviews\n"):
    src = tmp_path / "src.bin"
    src.write_bytes(content)
    return src, len(content), hashlib.sha256(content).hexdigest()


def test_sha256_matches_hashlib(tmp_path):
    src, _, digest = make_source(tmp_path)
    assert sha256_file(src) == digest


def test_check_file_accepts_matching_file(tmp_path):
    src, size, digest = make_source(tmp_path)
    assert check_file(src, size, digest) == digest


def test_check_file_rejects_wrong_size_and_wrong_hash(tmp_path):
    src, size, digest = make_source(tmp_path)
    with pytest.raises(IntegrityError):
        check_file(src, size + 1, digest)
    with pytest.raises(IntegrityError):
        check_file(src, size, "0" * 64)


def test_fetch_downloads_then_reuses_cache(tmp_path):
    src, size, digest = make_source(tmp_path)
    dest = tmp_path / "raw" / "file.bin"
    first = fetch(src.as_uri(), dest, size, digest)
    assert first["status"] == "downloaded"
    assert dest.read_bytes() == src.read_bytes()
    second = fetch(src.as_uri(), dest, size, digest)
    assert second["status"] == "cached"


def test_fetch_replaces_corrupt_cache(tmp_path):
    src, size, digest = make_source(tmp_path)
    dest = tmp_path / "raw" / "file.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"corrupt")
    assert fetch(src.as_uri(), dest, size, digest)["status"] == "downloaded"
    assert sha256_file(dest) == digest


def test_fetch_rejects_bad_download_and_leaves_no_files(tmp_path):
    src, size, _ = make_source(tmp_path)
    dest = tmp_path / "raw" / "file.bin"
    with pytest.raises(IntegrityError):
        fetch(src.as_uri(), dest, size, "f" * 64)
    assert not dest.exists()
    assert not list(dest.parent.glob("*.part"))


def test_manifest_round_trip(tmp_path):
    manifest = {"sha256": "abc", "rows": 3, "nested": {"a": 1}}
    path = tmp_path / "m" / "manifest.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
