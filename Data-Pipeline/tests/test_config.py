"""Tests for scripts.config."""

from __future__ import annotations

import pytest

from scripts.config import PipelineConfig, load_config
from scripts.exceptions import ConfigError


def test_load_config_reads_environment(monkeypatch):
    monkeypatch.setenv("TMDB_STAGE", "rest")
    monkeypatch.setenv("TMDB_MAX_WORKERS", "3")
    monkeypatch.setenv("SAMPLE_SIZE", "25")
    cfg = load_config()
    assert cfg.tmdb_stage == "rest"
    assert cfg.tmdb_max_workers == 3
    assert cfg.sample_size == 25


def test_overrides_win_and_none_is_ignored():
    cfg = load_config(tmdb_stage="all", tmdb_max_workers=None)
    assert cfg.tmdb_stage == "all"
    assert cfg.tmdb_max_workers == 8


def test_sample_size_zero_means_all():
    assert load_config(sample_size=0).sample_size is None


@pytest.mark.parametrize("field,value", [
    ("tmdb_stage", "everything"), ("tmdb_max_workers", 0), ("tmdb_rate_limit", 0),
    ("min_match_rate", 1.5), ("sample_size", -1), ("encoder_batch_size", 0), ("tmdb_max_failure_rate", 2),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ConfigError):
        PipelineConfig(**{field: value}).validate()


def test_bad_integer_env(monkeypatch):
    monkeypatch.setenv("TMDB_MAX_WORKERS", "many")
    with pytest.raises(ConfigError):
        load_config()


def test_bad_float_env(monkeypatch):
    monkeypatch.setenv("TMDB_RATE_LIMIT", "fast")
    with pytest.raises(ConfigError):
        load_config()


def test_bool_env(monkeypatch):
    monkeypatch.setenv("USE_CACHE", "no")
    assert load_config().use_cache is False


def test_secrets_are_masked():
    cfg = PipelineConfig(tmdb_api_key="supersecret", slack_webhook_url="https://hooks.example/x")
    safe = cfg.to_safe_dict()
    assert safe["tmdb_api_key"] == "***set***"
    assert "supersecret" not in repr(cfg)
    assert "hooks.example" not in str(safe)


def test_bearer_detection():
    assert PipelineConfig(tmdb_api_key="eyJhbGciOi.eyJzdWIiOi.sig").tmdb_uses_bearer
    assert not PipelineConfig(tmdb_api_key="0123456789abcdef0123456789abcdef").tmdb_uses_bearer


def test_derived_paths(tmp_path):
    cfg = PipelineConfig(data_dir=tmp_path)
    assert cfg.movielens_csv_dir == tmp_path / "raw" / "movielens" / "ml-25m"
    assert cfg.tmdb_cache_dir == tmp_path / "raw" / "tmdb" / "movies"
    assert cfg.dna_mapping_path.name == "movie_dna_mapping_v1.yaml"


def test_sample_runs_write_reports_to_sample_folder(tmp_path):
    full = PipelineConfig(reports_dir=tmp_path)
    assert full.run_reports_dir == tmp_path
    assert PipelineConfig(reports_dir=tmp_path, sample_size=200).run_reports_dir == tmp_path / "sample"
