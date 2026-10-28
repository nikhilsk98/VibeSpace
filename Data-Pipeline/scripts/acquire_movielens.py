"""Download, verify and extract the MovieLens 25M dataset.

Steps: fetch the published MD5, stream the zip with retries, verify the MD5,
extract safely (corrupt archives and path-traversal entries are rejected),
check every CSV header and write a manifest. If the manifest shows the data is
already present and verified, nothing is downloaded again.
"""

from __future__ import annotations

import re
import shutil
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from scripts.config import PipelineConfig
from scripts.exceptions import ChecksumMismatchError, CorruptArchiveError, DataAcquisitionError, SchemaError
from scripts.io_utils import ensure_dir, file_md5, read_csv_header, read_json, write_json
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

ZIP_NAME = "ml-25m.zip"
MANIFEST_NAME = "manifest.json"
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "movies.csv": ["movieId", "title", "genres"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "genome-scores.csv": ["movieId", "tagId", "relevance"],
    "genome-tags.csv": ["tagId", "tag"],
}
_MD5_RE = re.compile(r"\b([0-9a-fA-F]{32})\b")
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def parse_md5(text: str) -> str:
    """Extract a 32-character hex MD5 from the content of a ``.md5`` file."""
    match = _MD5_RE.search(text or "")
    if not match:
        raise DataAcquisitionError("Checksum file does not contain a valid MD5 digest")
    return match.group(1).lower()


def _backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff delay in seconds for a 0-based ``attempt``."""
    return min(cap, base * (2**attempt))


def fetch_expected_md5(session: requests.Session, url: str, timeout: float = 30.0, retries: int = 3,
                       sleep: Any = time.sleep) -> str:
    """Download the published checksum file and return the MD5 digest."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return parse_md5(resp.text)
            last_error = DataAcquisitionError(f"HTTP {resp.status_code} for checksum {url}")
            if resp.status_code not in RETRYABLE_STATUS:
                break
        except requests.RequestException as exc:
            last_error = exc
        logger.warning("Checksum download attempt %d/%d failed: %s", attempt + 1, retries, last_error)
        sleep(_backoff(attempt))
    raise DataAcquisitionError(f"Could not fetch checksum from {url}: {last_error}")


def download_file(session: requests.Session, url: str, dest: Path, retries: int = 4, timeout: float = 60.0,
                  chunk_size: int = 1 << 20, sleep: Any = time.sleep) -> Path:
    """Stream ``url`` to ``dest`` with retries; a ``.part`` file avoids partial outputs."""
    dest = Path(dest)
    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    raise DataAcquisitionError(f"HTTP {resp.status_code} while downloading {url}")
                total = int(resp.headers.get("Content-Length") or 0)
                written = 0
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)
                if total and written != total:
                    raise DataAcquisitionError(f"Incomplete download: got {written} of {total} bytes")
            part.replace(dest)
            logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return dest
        except (requests.RequestException, DataAcquisitionError) as exc:
            last_error = exc
            logger.warning("Download attempt %d/%d for %s failed: %s", attempt + 1, retries, url, exc)
            if part.exists():
                part.unlink()
            if attempt + 1 < retries:
                sleep(_backoff(attempt))
    raise DataAcquisitionError(f"Giving up on {url} after {retries} attempts: {last_error}")


def verify_md5(path: Path, expected: str) -> str:
    """Raise :class:`ChecksumMismatchError` unless ``path`` has MD5 ``expected``."""
    actual = file_md5(path)
    if actual != expected.lower():
        raise ChecksumMismatchError(f"MD5 mismatch for {Path(path).name}: expected {expected}, got {actual}")
    return actual


def _is_unsafe_member(name: str) -> bool:
    """True for absolute paths or entries that escape the target via ``..``."""
    posix = PurePosixPath(name.replace("\\", "/"))
    return posix.is_absolute() or ".." in posix.parts or bool(re.match(r"^[A-Za-z]:", name))


def safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract ``zip_path`` into ``dest_dir`` after integrity and safety checks."""
    zip_path, dest_dir = Path(zip_path), Path(dest_dir)
    if not zipfile.is_zipfile(zip_path):
        raise CorruptArchiveError(f"{zip_path.name} is not a valid zip archive")
    dest_root = ensure_dir(dest_dir).resolve()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise CorruptArchiveError(f"Corrupt member {bad!r} in {zip_path.name}")
            members = zf.infolist()
            for info in members:
                target = (dest_root / info.filename).resolve()
                if _is_unsafe_member(info.filename) or not target.is_relative_to(dest_root):
                    raise CorruptArchiveError(f"Unsafe path in archive rejected: {info.filename!r}")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise CorruptArchiveError(f"Symlink in archive rejected: {info.filename!r}")
            extracted = []
            for info in members:
                zf.extract(info, dest_root)
                if not info.is_dir():
                    extracted.append(dest_root / info.filename)
    except zipfile.BadZipFile as exc:
        raise CorruptArchiveError(f"Cannot read {zip_path.name}: {exc}") from exc
    logger.info("Extracted %d files from %s", len(extracted), zip_path.name)
    return extracted


def check_csv_headers(csv_dir: Path, expected: dict[str, list[str]] = EXPECTED_COLUMNS) -> dict[str, list[str]]:
    """Verify each expected CSV exists and its header contains the expected columns."""
    headers: dict[str, list[str]] = {}
    for name, required in expected.items():
        path = Path(csv_dir) / name
        if not path.is_file():
            raise SchemaError(f"Missing MovieLens file: {name}")
        header = read_csv_header(path)
        missing = [c for c in required if c not in header]
        if missing:
            raise SchemaError(f"{name} header {header} is missing columns {missing}")
        headers[name] = header
    return headers


def build_manifest(zip_path: Path, md5: str, csv_dir: Path, url: str) -> dict[str, Any]:
    """Describe the acquired dataset (checksum, file sizes, time)."""
    files = {
        name: {"bytes": (Path(csv_dir) / name).stat().st_size}
        for name in EXPECTED_COLUMNS
    }
    return {
        "dataset": "MovieLens 25M",
        "source_url": url,
        "zip_file": Path(zip_path).name,
        "zip_md5": md5,
        "zip_bytes": Path(zip_path).stat().st_size,
        "files": files,
        "acquired_at": datetime.now(UTC).isoformat(),
    }


def is_already_acquired(movielens_dir: Path, verify_checksum: bool = True) -> bool:
    """True if a manifest exists and the zip and CSVs still match it."""
    movielens_dir = Path(movielens_dir)
    manifest_path = movielens_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = read_json(manifest_path)
        zip_path = movielens_dir / manifest["zip_file"]
        csv_dir = movielens_dir / "ml-25m"
        if not zip_path.is_file() or zip_path.stat().st_size != manifest["zip_bytes"]:
            return False
        for name, meta in manifest["files"].items():
            path = csv_dir / name
            if not path.is_file() or path.stat().st_size != meta["bytes"]:
                return False
        if verify_checksum and file_md5(zip_path) != manifest["zip_md5"]:
            return False
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Existing MovieLens manifest is unusable (%s); re-acquiring", exc)
        return False
    return True


def acquire_movielens(cfg: PipelineConfig, session: requests.Session | None = None,
                      sleep: Any = time.sleep) -> dict[str, Any]:
    """Make sure verified MovieLens 25M CSVs exist; return the manifest."""
    movielens_dir = ensure_dir(cfg.movielens_dir)
    if is_already_acquired(movielens_dir):
        logger.info("MovieLens 25M already present and verified; skipping download")
        manifest = read_json(movielens_dir / MANIFEST_NAME)
        manifest["skipped"] = True
        return manifest

    session = session or requests.Session()
    expected_md5 = fetch_expected_md5(session, cfg.movielens_md5_url, sleep=sleep)
    zip_path = movielens_dir / ZIP_NAME
    if zip_path.is_file() and file_md5(zip_path) == expected_md5:
        logger.info("Reusing previously downloaded %s (checksum ok)", ZIP_NAME)
    else:
        download_file(session, cfg.movielens_url, zip_path, retries=cfg.download_retries,
                      timeout=cfg.download_timeout, sleep=sleep)
    try:
        verify_md5(zip_path, expected_md5)
    except ChecksumMismatchError:
        zip_path.unlink(missing_ok=True)
        raise

    extract_root = movielens_dir / "_extract"
    shutil.rmtree(extract_root, ignore_errors=True)
    safe_extract_zip(zip_path, extract_root)
    extracted_csv_dir = extract_root / "ml-25m"
    check_csv_headers(extracted_csv_dir)
    shutil.rmtree(cfg.movielens_csv_dir, ignore_errors=True)
    extracted_csv_dir.replace(cfg.movielens_csv_dir)
    shutil.rmtree(extract_root, ignore_errors=True)

    manifest = build_manifest(zip_path, expected_md5, cfg.movielens_csv_dir, cfg.movielens_url)
    write_json(movielens_dir / MANIFEST_NAME, manifest)
    manifest["skipped"] = False
    logger.info("MovieLens 25M acquired and verified (md5 %s)", expected_md5)
    return manifest
