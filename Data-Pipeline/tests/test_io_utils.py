"""Tests for scripts.io_utils."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from scripts import io_utils as io
from scripts.exceptions import SchemaError


def test_json_roundtrip_with_numpy(tmp_path):
    path = tmp_path / "a" / "b.json"
    io.write_json(path, {"i": np.int64(3), "f": np.float32(1.5), "nan": np.float64("nan"), "b": np.bool_(True),
                         "arr": np.arange(3), "p": tmp_path, "s": {2, 1}})
    data = io.read_json(path)
    assert data == {"i": 3, "f": 1.5, "nan": None, "b": True, "arr": [0, 1, 2], "p": str(tmp_path), "s": [1, 2]}
    if os.name != "nt":  # Windows has no Unix permission bits
        assert oct(os.stat(path).st_mode & 0o777) == "0o644"


def test_json_rejects_unknown_type(tmp_path):
    with pytest.raises(TypeError):
        io.write_json(tmp_path / "x.json", {"x": object()})
    assert not list(tmp_path.glob(".*tmp"))  # no temp file left behind


def test_parquet_roundtrip_and_missing(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [["x"], []]})
    io.write_parquet(tmp_path / "t.parquet", df)
    back = io.read_parquet(tmp_path / "t.parquet")
    assert back["a"].tolist() == [1, 2]
    with pytest.raises(FileNotFoundError, match="upstream"):
        io.read_parquet(tmp_path / "missing.parquet")


def test_numpy_and_text(tmp_path):
    io.save_numpy(tmp_path / "a.npy", np.ones((2, 3), dtype=np.float32))
    assert np.load(tmp_path / "a.npy").shape == (2, 3)
    io.write_text(tmp_path / "r.md", "héllo")
    assert (tmp_path / "r.md").read_text(encoding="utf-8") == "héllo"


def test_md5_and_sha1(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc" * 1000)
    assert io.file_md5(p, chunk_size=7) == hashlib.md5(b"abc" * 1000).hexdigest()  # noqa: S324
    assert io.text_sha1("x") == io.text_sha1("x") != io.text_sha1("y")


def test_csv_helpers(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text('"movieId",title\n1,A\n')
    assert io.read_csv_header(p) == ["movieId", "title"]
    assert len(io.read_csv_checked(p, ["movieId"])) == 1
    with pytest.raises(SchemaError):
        io.read_csv_checked(p, ["genres"])
    with pytest.raises(FileNotFoundError):
        io.read_csv_checked(tmp_path / "none.csv", [])
    empty = tmp_path / "e.csv"
    empty.write_text("")
    assert io.read_csv_header(empty) == []