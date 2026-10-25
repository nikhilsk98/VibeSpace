"""File input/output helpers: atomic writes, checksums and checked CSV reads."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.exceptions import SchemaError


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


FILE_MODE = 0o644


def _atomic_target(path: Path) -> tuple[int, str]:
    """Open a temporary file next to ``path`` for an atomic replace."""
    ensure_dir(path.parent)
    return tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")


def write_bytes_atomic(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` so readers never see a partial file."""
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def _json_default(obj: Any) -> Any:
    """Make numpy / pandas / Path values JSON serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path | pd.Timestamp):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _sanitize(obj: Any) -> Any:
    """Replace NaN / infinity floats (anywhere in the structure) with ``None``."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize(v) for v in obj]
    return obj


def write_json(path: Path, payload: Any) -> Path:
    """Atomically write ``payload`` as pretty JSON (NaN becomes ``null``)."""
    text = json.dumps(_sanitize(payload), indent=2, sort_keys=True, default=_json_default, allow_nan=False)
    return write_bytes_atomic(Path(path), (text + "\n").encode("utf-8"))


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_text(path: Path, text: str) -> Path:
    """Atomically write a UTF-8 text file."""
    return write_bytes_atomic(Path(path), text.encode("utf-8"))


def write_parquet(path: Path, df: pd.DataFrame) -> Path:
    """Atomically write a DataFrame to Parquet (index dropped)."""
    path = Path(path)
    fd, tmp = _atomic_target(path)
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file, raising a clear error when it is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected input file is missing: {path}. Run the upstream step first.")
    return pd.read_parquet(path, columns=columns)


def save_numpy(path: Path, array: np.ndarray) -> Path:
    """Atomically save a numpy array as ``.npy``."""
    path = Path(path)
    fd, tmp = _atomic_target(path)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array)
        os.chmod(tmp, FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file, read in chunks."""
    digest = hashlib.md5()  # noqa: S324 - MovieLens publishes MD5 checksums
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha1(text: str) -> str:
    """Return a short stable hash of a text (used as a cache key)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324 - not security related


def read_csv_header(path: Path) -> list[str]:
    """Return the column names of a CSV file without loading it."""
    with open(path, encoding="utf-8") as fh:
        first = fh.readline().strip()
    return [c.strip().strip('"') for c in first.split(",")] if first else []


def check_columns(columns: Iterable[str], required: Iterable[str], source: str) -> None:
    """Raise :class:`SchemaError` if any ``required`` column is missing."""
    missing = sorted(set(required) - set(columns))
    if missing:
        raise SchemaError(f"{source} is missing required columns: {missing}")


def read_csv_checked(path: Path, required: Iterable[str], **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file and verify that it has the ``required`` columns."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Expected CSV is missing: {path}")
    df = pd.read_csv(path, **kwargs)
    check_columns(df.columns, required, str(path.name))
    return df
