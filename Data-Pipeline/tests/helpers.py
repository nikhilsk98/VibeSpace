"""Shared test helpers: fake HTTP, fake encoder, config and fixture installers.

Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import requests

from scripts.acquire_tmdb import NOT_FOUND_MARKER
from scripts.config import PIPELINE_ROOT, PipelineConfig

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ML_FIXTURES = FIXTURES / "movielens"
TMDB_FIXTURES = FIXTURES / "tmdb"
MISSING_TMDB_ID = 999999


class FakeResponse:
    """Just enough of ``requests.Response`` for the pipeline code."""

    def __init__(self, status_code: int = 200, json_data: Any = None, text: str = "",
                 headers: dict[str, str] | None = None, content: bytes = b"", bad_json: bool = False) -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}
        self.content = content
        self._bad_json = bad_json

    def json(self) -> Any:
        if self._bad_json:
            raise ValueError("bad json")
        return self._json

    def iter_content(self, chunk_size: int = 1024):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class FakeSession:
    """Session whose ``get`` / ``post`` answers come from a handler or a queue."""

    def __init__(self, handler: Callable[..., Any] | None = None, responses: list[Any] | None = None) -> None:
        self.handler = handler
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def _answer(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self.handler(url, **kwargs) if self.handler else self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._answer("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return self._answer("POST", url, **kwargs)


def tmdb_fixture_handler(url: str, **_: Any) -> FakeResponse:
    """Serve TMDB fixture JSON by id; unknown ids get HTTP 404."""
    tmdb_id = url.rstrip("/").rsplit("/", 1)[-1]
    path = TMDB_FIXTURES / f"{tmdb_id}.json"
    if path.is_file():
        return FakeResponse(200, json.loads(path.read_text(encoding="utf-8")))
    return FakeResponse(404, {"status_code": 34})


class NoSleep:
    """Records requested sleeps instead of sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class FakeEncoder:
    """Deterministic 384-dim 'embeddings' derived from a hash of the text."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.encoded: list[str] = []

    def encode(self, sentences: list[str], batch_size: int = 32, **_: Any) -> np.ndarray:
        self.encoded.extend(sentences)
        rows = []
        for text in sentences:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)  # noqa: S324
            vec = np.random.default_rng(seed).normal(size=self.dim)
            rows.append(vec / np.linalg.norm(vec))
        return np.asarray(rows, dtype=np.float32)


def make_config(tmp_path: Path, **overrides: Any) -> PipelineConfig:
    """A config whose data, logs, reports and GX folders live under ``tmp_path``."""
    base = {
        "root": tmp_path,
        "data_dir": tmp_path / "data",
        "logs_dir": tmp_path / "logs",
        "reports_dir": tmp_path / "reports",
        "gx_dir": tmp_path / "gx",
        "configs_dir": PIPELINE_ROOT / "configs",
        "tmdb_api_key": "test_key_0000000000000000000000",
        "tmdb_max_workers": 2,
        "tmdb_rate_limit": 10_000.0,
        "tmdb_backoff_base": 0.0,
        "min_match_rate": 0.70,
        "dvc_push": False,
    }
    base.update(overrides)
    return PipelineConfig(**base).validate()


def install_movielens(cfg: PipelineConfig) -> Path:
    """Copy the MovieLens fixture CSVs to where the config expects them."""
    shutil.copytree(ML_FIXTURES, cfg.movielens_csv_dir, dirs_exist_ok=True)
    return cfg.movielens_csv_dir


def seed_tmdb_cache(cfg: PipelineConfig) -> Path:
    """Fill the TMDB cache from fixtures, including a cached 404."""
    cache = cfg.tmdb_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    for path in TMDB_FIXTURES.glob("*.json"):
        shutil.copy(path, cache / path.name)
    (cache / f"{MISSING_TMDB_ID}.json").write_text(json.dumps({NOT_FOUND_MARKER: True, "id": MISSING_TMDB_ID}))
    return cache


def http_error(kind: str) -> Exception:
    """Network exceptions raised by fake sessions."""
    return {"timeout": requests.Timeout("timed out"), "conn": requests.ConnectionError("refused")}[kind]
