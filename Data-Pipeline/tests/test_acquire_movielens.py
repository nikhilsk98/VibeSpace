"""Tests for scripts.acquire_movielens (no network: fake sessions only)."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest
import requests

from scripts import acquire_movielens as am
from scripts.exceptions import ChecksumMismatchError, CorruptArchiveError, DataAcquisitionError, SchemaError
from tests.helpers import ML_FIXTURES, FakeResponse, FakeSession, NoSleep


def make_zip(files: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def movielens_zip() -> bytes:
    """A zip shaped like ml-25m.zip built from the fixture CSVs."""
    files = {f"ml-25m/{p.name}": p.read_bytes() for p in ML_FIXTURES.glob("*.csv")}
    return make_zip(files)


def session_for(zip_bytes: bytes, md5: str | None = None) -> FakeSession:
    md5 = md5 or hashlib.md5(zip_bytes).hexdigest()  # noqa: S324

    def handler(url, **_):
        if url.endswith(".md5"):
            return FakeResponse(200, text=f"{md5}  ml-25m.zip\n")
        return FakeResponse(200, content=zip_bytes, headers={"Content-Length": str(len(zip_bytes))})

    return FakeSession(handler)


def test_parse_md5():
    assert am.parse_md5("6B51FB2759A8657D3BFCBFC42B592ADA  ml-25m.zip") == "6b51fb2759a8657d3bfcbfc42b592ada"
    with pytest.raises(DataAcquisitionError):
        am.parse_md5("no checksum here")


def test_acquire_end_to_end_and_skip(cfg):
    data = movielens_zip()
    session = session_for(data)
    manifest = am.acquire_movielens(cfg, session=session, sleep=NoSleep())
    assert manifest["skipped"] is False
    assert (cfg.movielens_csv_dir / "movies.csv").is_file()
    assert (cfg.movielens_dir / "manifest.json").is_file()
    calls = len(session.calls)
    again = am.acquire_movielens(cfg, session=session, sleep=NoSleep())
    assert again["skipped"] is True
    assert len(session.calls) == calls  # nothing downloaded the second time


def test_checksum_mismatch_deletes_zip(cfg):
    session = session_for(movielens_zip(), md5="0" * 32)
    with pytest.raises(ChecksumMismatchError):
        am.acquire_movielens(cfg, session=session, sleep=NoSleep())
    assert not (cfg.movielens_dir / am.ZIP_NAME).exists()


def test_download_retries_then_succeeds(tmp_path):
    content = b"x" * 3000
    session = FakeSession(responses=[requests.ConnectionError("boom"), FakeResponse(503),
                                     FakeResponse(200, content=content)])
    sleep = NoSleep()
    out = am.download_file(session, "http://example/f", tmp_path / "f.bin", retries=3, sleep=sleep)
    assert out.read_bytes() == content
    assert len(sleep.calls) == 2


def test_download_gives_up(tmp_path):
    session = FakeSession(responses=[FakeResponse(500)] * 3)
    with pytest.raises(DataAcquisitionError):
        am.download_file(session, "http://example/f", tmp_path / "f.bin", retries=3, sleep=NoSleep())
    assert not (tmp_path / "f.bin.part").exists()


def test_download_detects_truncation(tmp_path):
    resp = FakeResponse(200, content=b"abc", headers={"Content-Length": "10"})
    with pytest.raises(DataAcquisitionError):
        am.download_file(FakeSession(responses=[resp]), "http://e/f", tmp_path / "f", retries=1, sleep=NoSleep())


def test_fetch_md5_http_error():
    session = FakeSession(responses=[FakeResponse(404)])
    with pytest.raises(DataAcquisitionError):
        am.fetch_expected_md5(session, "http://e/x.md5", retries=2, sleep=NoSleep())


def test_fetch_md5_retries_network_errors():
    session = FakeSession(responses=[requests.Timeout("slow"), FakeResponse(200, text="a" * 32)])
    assert am.fetch_expected_md5(session, "http://e/x.md5", retries=2, sleep=NoSleep()) == "a" * 32


def test_corrupt_zip_rejected(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    with pytest.raises(CorruptArchiveError):
        am.safe_extract_zip(bad, tmp_path / "out")


@pytest.mark.parametrize("name", ["../evil.csv", "/abs/evil.csv", "ml-25m/../../evil.csv", "C:/evil.csv"])
def test_path_traversal_rejected(tmp_path, name):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(make_zip({name: b"x"}))
    with pytest.raises(CorruptArchiveError):
        am.safe_extract_zip(archive, tmp_path / "out")
    assert not (tmp_path / "evil.csv").exists()


def test_symlink_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("ml-25m/link")
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    archive = tmp_path / "link.zip"
    archive.write_bytes(buf.getvalue())
    with pytest.raises(CorruptArchiveError):
        am.safe_extract_zip(archive, tmp_path / "out")


def test_header_check_missing_file(tmp_path):
    with pytest.raises(SchemaError):
        am.check_csv_headers(tmp_path)


def test_header_check_missing_column(tmp_path):
    for name in am.EXPECTED_COLUMNS:
        (tmp_path / name).write_text(",".join(am.EXPECTED_COLUMNS[name]) + "\n")
    (tmp_path / "links.csv").write_text("movieId,imdbId\n")
    with pytest.raises(SchemaError, match="tmdbId"):
        am.check_csv_headers(tmp_path)


def test_is_already_acquired_detects_tampering(cfg):
    am.acquire_movielens(cfg, session=session_for(movielens_zip()), sleep=NoSleep())
    assert am.is_already_acquired(cfg.movielens_dir)
    (cfg.movielens_csv_dir / "movies.csv").write_text("movieId,title,genres\n")
    assert not am.is_already_acquired(cfg.movielens_dir)


def test_is_already_acquired_bad_manifest(tmp_path: Path):
    (tmp_path / am.MANIFEST_NAME).write_text("{not json")
    assert not am.is_already_acquired(tmp_path)
    assert not am.is_already_acquired(tmp_path / "missing")
