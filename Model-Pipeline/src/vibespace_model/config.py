"""Configuration: YAML files plus environment overrides.

Relative paths in the YAML are resolved against the Model-Pipeline folder (the parent of ``configs/``).
Environment variables override selected values:

- ``RANDOM_SEED``: global seed.
- ``VIBESPACE_MODEL_PROCESSED_DIR``: folder with the four pipeline outputs.
- ``VIBESPACE_MODEL_DATA_PIPELINE_DIR``: folder holding ``dvc.lock``.
- ``VIBESPACE_MODEL_REPORTS_DIR``, ``VIBESPACE_MODEL_ARTIFACTS_DIR``, ``VIBESPACE_MODEL_MLFLOW_DIR``: outputs.
- ``VIBESPACE_MODEL_OVERRIDES``: JSON deep-merged into the config, e.g. ``{"gate": {"min_r2": 0.99}}``;
  the keys ``tuning`` and ``bias`` go into ``tuning.yaml`` / ``bias.yaml`` (used to test CI failures).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vibespace_model.errors import ConfigError

DNA_DIMENSIONS: tuple[str, ...] = (
    "energy",
    "intensity",
    "pace",
    "humor",
    "darkness",
    "hopefulness",
    "emotional_intensity",
    "romance",
    "suspense",
    "complexity",
    "realism",
    "inspiration",
)

_PATH_ENV = {
    "processed_dir": "VIBESPACE_MODEL_PROCESSED_DIR",
    "data_pipeline_dir": "VIBESPACE_MODEL_DATA_PIPELINE_DIR",
    "reports_dir": "VIBESPACE_MODEL_REPORTS_DIR",
    "artifacts_dir": "VIBESPACE_MODEL_ARTIFACTS_DIR",
    "mlflow_dir": "VIBESPACE_MODEL_MLFLOW_DIR",
}


@dataclass
class Paths:
    """Absolute locations used by a run."""

    data_pipeline_dir: Path
    processed_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    mlflow_dir: Path

    @property
    def dvc_lock(self) -> Path:
        """``dvc.lock`` of the data pipeline."""
        return self.data_pipeline_dir / "dvc.lock"


@dataclass
class ModelConfig:
    """Everything a run needs, after overrides and path resolution."""

    root: Path
    random_seed: int
    experiment_name: str
    registered_model_name: str
    paths: Paths
    data: dict[str, Any]
    targets: list[str]
    features: dict[str, Any]
    split: dict[str, Any]
    slices: list[str]
    models: dict[str, dict[str, Any]]
    gate: dict[str, float]
    explain: dict[str, Any] = field(default_factory=dict)
    registry: dict[str, Any] = field(default_factory=dict)
    ci: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)
    bias: dict[str, Any] = field(default_factory=dict)
    sample_size: int | None = None

    def model_params(self, name: str) -> dict[str, Any]:
        """Copy of the default parameters of one candidate model."""
        if name not in self.models:
            raise ConfigError(f"no parameters for model '{name}' in the config")
        return copy.deepcopy(self.models[name])

    def reports_subdir(self, part: str) -> Path:
        """Report folder of one part; sample runs write to ``reports/sample/`` instead."""
        base = self.paths.reports_dir / "sample" if self.sample_size else self.paths.reports_dir
        path = base / part
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts_subdir(self, part: str) -> Path:
        """Artifact folder of one part (models, split, studies)."""
        base = self.paths.artifacts_dir / "sample" if self.sample_size else self.paths.artifacts_dir
        path = base / part
        path.mkdir(parents=True, exist_ok=True)
        return path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ConfigError(f"config file must contain a mapping: {path}")
    return content


def _resolve(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _require(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw:
        raise ConfigError(f"missing key '{key}' in {path}")
    return raw[key]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge (override wins; nested dicts are merged, other values replaced)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _env_overrides() -> dict[str, Any]:
    text = os.environ.get("VIBESPACE_MODEL_OVERRIDES", "").strip()
    if not text:
        return {}
    try:
        overrides = json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"VIBESPACE_MODEL_OVERRIDES is not valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise ConfigError("VIBESPACE_MODEL_OVERRIDES must be a JSON object")
    return overrides


def _validate(cfg: ModelConfig) -> None:
    ratios = [float(cfg.split[k]) for k in ("train", "val", "test")]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ConfigError(f"split ratios must be positive and sum to 1, got {ratios}")
    if list(cfg.targets) != list(DNA_DIMENSIONS):
        raise ConfigError(f"targets must be the 12 DNA dimensions in order: {list(DNA_DIMENSIONS)}")
    for key in ("min_mae_improvement", "min_r2"):
        if key not in cfg.gate:
            raise ConfigError(f"gate.{key} is required")
    unknown = set(cfg.models) - {"baseline", "ridge", "lightgbm", "mlp"}
    if unknown:
        raise ConfigError(f"unknown candidate models: {sorted(unknown)}")


def load_config(
    path: str | os.PathLike[str],
    tuning_path: str | os.PathLike[str] | None = None,
    bias_path: str | os.PathLike[str] | None = None,
    sample_size: int | None = None,
) -> ModelConfig:
    """Load ``model_config.yaml`` (+ tuning and bias configs next to it) and apply overrides."""
    path = Path(path).expanduser().resolve()
    overrides = _env_overrides()
    tuning_override, bias_override = overrides.pop("tuning", {}), overrides.pop("bias", {})
    raw = deep_merge(_read_yaml(path), overrides)
    root = path.parent.parent
    raw_paths = _require(raw, "paths", path)

    resolved: dict[str, Path] = {}
    for key, env in _PATH_ENV.items():
        value = os.environ.get(env) or _require(raw_paths, key, path)
        resolved[key] = _resolve(root, value)

    seed = int(os.environ.get("RANDOM_SEED", raw.get("random_seed", 42)))
    tuning_file = Path(tuning_path) if tuning_path else path.parent / "tuning.yaml"
    bias_file = Path(bias_path) if bias_path else path.parent / "bias.yaml"

    cfg = ModelConfig(
        root=root,
        random_seed=seed,
        experiment_name=str(raw.get("experiment_name", "vibespace-movie-dna")),
        registered_model_name=str(raw.get("registered_model_name", "vibespace-movie-dna")),
        paths=Paths(**resolved),
        data=dict(_require(raw, "data", path)),
        targets=list(_require(raw, "targets", path)),
        features=dict(_require(raw, "features", path)),
        split=dict(_require(raw, "split", path)),
        slices=list(raw.get("slices", [])),
        models={k: dict(v or {}) for k, v in _require(raw, "models", path).items()},
        gate={k: float(v) for k, v in _require(raw, "gate", path).items()},
        explain=dict(raw.get("explain") or {}),
        registry=dict(raw.get("registry") or {}),
        ci=dict(raw.get("ci") or {}),
        tuning=deep_merge(_read_yaml(tuning_file) if tuning_file.is_file() else {}, tuning_override),
        bias=deep_merge(_read_yaml(bias_file) if bias_file.is_file() else {}, bias_override),
        sample_size=sample_size,
    )
    _validate(cfg)
    return cfg
