"""Pytest fixtures shared by all test modules."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest

# Airflow (imported by the DAG tests) must not touch a real AIRFLOW_HOME.
os.environ.setdefault("AIRFLOW_HOME", str(Path(tempfile.gettempdir()) / "vibespace_airflow_test"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
os.environ.setdefault("AIRFLOW__CORE__UNIT_TEST_MODE", "True")

from scripts.config import PipelineConfig
from tests.helpers import FakeEncoder, install_movielens, make_config, seed_tmdb_cache


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from real data, logs, secrets and alert channels."""
    monkeypatch.setenv("VIBESPACE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIBESPACE_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("VIBESPACE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("VIBESPACE_GX_DIR", str(tmp_path / "gx"))
    monkeypatch.setenv("TMDB_API_KEY", "test_key_0000000000000000000000")
    for name in ("SLACK_WEBHOOK_URL", "SMTP_HOST", "ALERT_EMAIL", "SMTP_USER", "SMTP_PASSWORD", "SAMPLE_SIZE"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("TMDB_STAGE", "genome")
    monkeypatch.setenv("GX_ANALYTICS_ENABLED", "false")
    yield
    logger = logging.getLogger("vibespace")
    handler = getattr(logger, "_vibespace_file_handler", None)
    if handler is not None:
        logger.removeHandler(handler)
        handler.close()
        delattr(logger, "_vibespace_file_handler")


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    """Config rooted in a temporary folder (no data installed)."""
    return make_config(tmp_path)


@pytest.fixture
def ml_cfg(cfg: PipelineConfig) -> PipelineConfig:
    """Config with the MovieLens fixture installed."""
    install_movielens(cfg)
    return cfg


@pytest.fixture
def cached_cfg(ml_cfg: PipelineConfig) -> PipelineConfig:
    """Config with MovieLens fixtures and a filled TMDB cache."""
    seed_tmdb_cache(ml_cfg)
    return ml_cfg


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    """Deterministic offline text encoder."""
    return FakeEncoder()
