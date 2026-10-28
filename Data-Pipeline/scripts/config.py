"""Central configuration for the VibeSpace data pipeline.

All settings come from environment variables (optionally loaded from
``Data-Pipeline/.env``) with safe defaults. Secrets are never logged: they are
excluded from ``repr`` and from :meth:`PipelineConfig.to_safe_dict`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from scripts.exceptions import ConfigError

PIPELINE_ROOT = Path(__file__).resolve().parents[1]

TMDB_STAGES = ("genome", "rest", "all")

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
MOVIELENS_MD5_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip.md5"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SECRET_FIELDS = frozenset({"tmdb_api_key", "slack_webhook_url", "smtp_password"})


def load_dotenv_file(root: Path = PIPELINE_ROOT) -> None:
    """Load ``root/.env`` into the environment without overriding set values."""
    env_path = root / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        return
    load_dotenv(env_path, override=False)


def _env_str(name: str, default: str = "") -> str:
    """Return a stripped string environment variable."""
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int | None) -> int | None:
    """Return an integer environment variable, or ``default`` when unset."""
    raw = _env_str(name)
    if raw == "" or raw.lower() == "none":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    """Return a float environment variable, or ``default`` when unset."""
    raw = _env_str(name)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable (1/true/yes/on are truthy)."""
    raw = _env_str(name).lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable pipeline settings. Use :func:`load_config` to build one."""

    root: Path = PIPELINE_ROOT
    data_dir: Path = PIPELINE_ROOT / "data"
    logs_dir: Path = PIPELINE_ROOT / "logs"
    reports_dir: Path = PIPELINE_ROOT / "reports"
    configs_dir: Path = PIPELINE_ROOT / "configs"
    gx_dir: Path = PIPELINE_ROOT / "gx"

    movielens_url: str = MOVIELENS_URL
    movielens_md5_url: str = MOVIELENS_MD5_URL
    download_retries: int = 4
    download_timeout: float = 60.0

    tmdb_api_key: str = field(default="", repr=False)
    tmdb_base_url: str = TMDB_BASE_URL
    tmdb_stage: str = "genome"
    tmdb_max_workers: int = 8
    tmdb_rate_limit: float = 20.0
    tmdb_timeout: float = 10.0
    tmdb_max_retries: int = 5
    tmdb_backoff_base: float = 1.0
    tmdb_max_failure_rate: float = 0.05

    sample_size: int | None = None
    random_seed: int = 42
    use_cache: bool = True

    min_match_rate: float = 0.90
    year_mismatch_tolerance: int = 2
    runtime_outlier_minutes: int = 400
    short_overview_chars: int = 40

    dna_mapping_file: str = "movie_dna_mapping_v1.yaml"
    encoder_model: str = ENCODER_MODEL
    encoder_batch_size: int = 64
    embedding_dim: int = 384

    psi_warning: float = 0.10
    psi_critical: float = 0.25
    null_rate_jump: float = 0.05
    row_count_drop: float = 0.20
    fail_on_critical_anomaly: bool = True

    slack_webhook_url: str = field(default="", repr=False)
    alert_email: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = field(default="", repr=False)
    smtp_sender: str = ""
    smtp_use_tls: bool = True

    dvc_push: bool = True

    # ----- derived locations -------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        """Raw, immutable inputs."""
        return self.data_dir / "raw"

    @property
    def movielens_dir(self) -> Path:
        """Folder holding the MovieLens zip, manifest and extracted CSVs."""
        return self.raw_dir / "movielens"

    @property
    def movielens_csv_dir(self) -> Path:
        """Folder with the extracted MovieLens 25M CSV files."""
        return self.movielens_dir / "ml-25m"

    @property
    def tmdb_dir(self) -> Path:
        """Root of the TMDB response cache."""
        return self.raw_dir / "tmdb"

    @property
    def tmdb_cache_dir(self) -> Path:
        """One cached JSON file per TMDB movie id."""
        return self.tmdb_dir / "movies"

    @property
    def interim_dir(self) -> Path:
        """Intermediate tables between steps."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Final, model-ready outputs."""
        return self.data_dir / "processed"

    @property
    def metrics_dir(self) -> Path:
        """Validation results, statistics, anomalies and bias metrics."""
        return self.data_dir / "metrics"

    @property
    def cache_dir(self) -> Path:
        """Reusable caches (e.g. plot embeddings)."""
        return self.data_dir / "cache"

    @property
    def run_reports_dir(self) -> Path:
        """Where this run writes its reports: sample runs use ``reports/sample/`` (git-ignored)."""
        return self.reports_dir / "sample" if self.sample_size else self.reports_dir

    @property
    def dna_mapping_path(self) -> Path:
        """Path to the Tag Genome to Movie DNA mapping."""
        return self.configs_dir / self.dna_mapping_file

    @property
    def tmdb_uses_bearer(self) -> bool:
        """True when the TMDB credential is a v4 read access token (JWT)."""
        key = self.tmdb_api_key
        return key.startswith("eyJ") and key.count(".") == 2

    def to_safe_dict(self) -> dict[str, Any]:
        """Return the config as a dict with secrets masked (safe to log)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in SECRET_FIELDS:
                value = "***set***" if value else ""
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out

    def validate(self) -> PipelineConfig:
        """Check value ranges and return ``self`` so calls can be chained."""
        if self.tmdb_stage not in TMDB_STAGES:
            raise ConfigError(f"TMDB_STAGE must be one of {TMDB_STAGES}, got {self.tmdb_stage!r}")
        if self.tmdb_max_workers < 1:
            raise ConfigError("TMDB_MAX_WORKERS must be >= 1")
        if self.tmdb_rate_limit <= 0:
            raise ConfigError("TMDB_RATE_LIMIT must be > 0")
        if not 0 <= self.tmdb_max_failure_rate <= 1:
            raise ConfigError("TMDB_MAX_FAILURE_RATE must be within [0, 1]")
        if not 0 <= self.min_match_rate <= 1:
            raise ConfigError("MIN_MATCH_RATE must be within [0, 1]")
        if self.sample_size is not None and self.sample_size < 1:
            raise ConfigError("SAMPLE_SIZE must be a positive integer when set")
        if self.encoder_batch_size < 1:
            raise ConfigError("ENCODER_BATCH_SIZE must be >= 1")
        return self


def load_config(**overrides: Any) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from the environment plus ``overrides``.

    Overrides with value ``None`` are ignored, except ``sample_size`` where
    ``None`` explicitly means "use every movie".
    """
    load_dotenv_file()
    data_dir = Path(_env_str("VIBESPACE_DATA_DIR") or PIPELINE_ROOT / "data")
    logs_dir = Path(_env_str("VIBESPACE_LOGS_DIR") or PIPELINE_ROOT / "logs")
    reports_dir = Path(_env_str("VIBESPACE_REPORTS_DIR") or PIPELINE_ROOT / "reports")
    gx_dir = Path(_env_str("VIBESPACE_GX_DIR") or PIPELINE_ROOT / "gx")
    tmdb_key = _env_str("TMDB_API_KEY") or _env_str("TMDB_READ_ACCESS_TOKEN")
    cfg = PipelineConfig(
        data_dir=data_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        gx_dir=gx_dir,
        tmdb_api_key=tmdb_key,
        tmdb_stage=_env_str("TMDB_STAGE", "genome") or "genome",
        tmdb_max_workers=_env_int("TMDB_MAX_WORKERS", 8) or 8,
        tmdb_rate_limit=_env_float("TMDB_RATE_LIMIT", 20.0),
        tmdb_timeout=_env_float("TMDB_TIMEOUT", 10.0),
        tmdb_max_retries=_env_int("TMDB_MAX_RETRIES", 5) or 5,
        tmdb_max_failure_rate=_env_float("TMDB_MAX_FAILURE_RATE", 0.05),
        sample_size=_env_int("SAMPLE_SIZE", None),
        random_seed=_env_int("RANDOM_SEED", 42) or 42,
        use_cache=_env_bool("USE_CACHE", True),
        min_match_rate=_env_float("MIN_MATCH_RATE", 0.90),
        encoder_batch_size=_env_int("ENCODER_BATCH_SIZE", 64) or 64,
        slack_webhook_url=_env_str("SLACK_WEBHOOK_URL"),
        alert_email=_env_str("ALERT_EMAIL"),
        smtp_host=_env_str("SMTP_HOST"),
        smtp_port=_env_int("SMTP_PORT", 587) or 587,
        smtp_user=_env_str("SMTP_USER"),
        smtp_password=_env_str("SMTP_PASSWORD"),
        smtp_sender=_env_str("SMTP_SENDER"),
        smtp_use_tls=_env_bool("SMTP_USE_TLS", True),
        dvc_push=_env_bool("DVC_PUSH", True),
    )
    clean_overrides = {k: v for k, v in overrides.items() if v is not None or k == "sample_size"}
    if "sample_size" in clean_overrides and clean_overrides["sample_size"] in (0, "0", ""):
        clean_overrides["sample_size"] = None
    return replace(cfg, **clean_overrides).validate()
