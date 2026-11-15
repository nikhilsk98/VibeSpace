"""Local MLflow tracking (SQLite backend + local artifact folder) and the local Model Registry.

One parent run per session with nested runs per candidate (and per Optuna trial in Part 2). Selected
models are logged as MLflow pyfunc models that wrap the saved bundle (model + preprocessing + card) and
registered as versions of ``vibespace-movie-dna`` with an alias (``candidate``, ``champion-candidate``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import contextlib  # noqa: E402
import logging  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from vibespace_model.config import ModelConfig  # noqa: E402

LOGGER = logging.getLogger(__name__)
PACKAGE_DIR = Path(__file__).resolve().parent
_MAX_PARAM_LEN = 500


def tracking_uri(mlflow_dir: Path) -> str:
    """SQLite URI of the local store."""
    return f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve()}"


def setup_tracking(cfg: ModelConfig) -> str:
    """Point MLflow at ``<mlflow_dir>/mlflow.db`` and make sure the experiment exists; return its id."""
    mlflow_dir = cfg.paths.mlflow_dir
    (mlflow_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri(mlflow_dir))
    mlflow.set_registry_uri(tracking_uri(mlflow_dir))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            cfg.experiment_name, artifact_location=(mlflow_dir / "artifacts").resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@contextlib.contextmanager
def run(name: str, nested: bool = False, tags: dict[str, Any] | None = None) -> Iterator[mlflow.ActiveRun]:
    """Start a (nested) MLflow run with string tags."""
    with mlflow.start_run(run_name=name, nested=nested) as active:
        if tags:
            mlflow.set_tags({k: str(v) for k, v in tags.items()})
        yield active


def session_tags(
    cfg: ModelConfig, data_version: dict[str, Any], split_hash: str, part: str
) -> dict[str, Any]:
    """Tags every run of a session carries: data version, split hash, Git commit."""
    return {
        "part": part,
        "split_hash": split_hash,
        "data_version_id": data_version.get("data_version_id"),
        "mapping_version": data_version.get("mapping_version"),
        "hash_check": data_version.get("hash_check"),
        "git_commit": data_version.get("git_commit"),
        "seed": cfg.random_seed,
        "sample_size": cfg.sample_size or "full",
    }


def log_params(params: dict[str, Any], prefix: str = "") -> None:
    """Log parameters as strings (MLflow limits their length)."""
    clean = {f"{prefix}{k}": str(v)[:_MAX_PARAM_LEN] for k, v in params.items()}
    if clean:
        mlflow.log_params(clean)


def flatten_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    """``{"overall": {...}, "per_dimension": {...}}`` -> ``{prefix_mae: ..., prefix_mae_energy: ...}``."""
    flat = {f"{prefix}_{k}": float(v) for k, v in metrics["overall"].items() if k != "n"}
    for dim, values in metrics.get("per_dimension", {}).items():
        flat.update({f"{prefix}_{k}_{dim}": float(v) for k, v in values.items()})
    return flat


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log finite numeric metrics."""
    clean = {k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(float(v))}
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_artifacts(paths: list[Path], artifact_path: str | None = None) -> None:
    """Log existing files / folders as artifacts of the active run."""
    for path in paths:
        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        elif path.is_file():
            mlflow.log_artifact(str(path), artifact_path=artifact_path)


class DnaPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper: catalog columns + ``emb_000..emb_383`` in, 12 DNA scores out."""

    def load_context(self, context: Any) -> None:
        """Load the saved bundle shipped with the MLflow model."""
        from vibespace_model.select import load_bundle

        self.model, self.builder, self.card = load_bundle(Path(context.artifacts["bundle"]))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Build features with the saved preprocessing, then predict."""
        emb_cols = [f"emb_{i:03d}" for i in range(self.builder.embedding_dim)]
        embeddings = model_input[emb_cols].to_numpy(dtype=np.float32)
        catalog = model_input.drop(columns=emb_cols)
        scores = self.model.predict(self.builder.transform(catalog, embeddings))
        return pd.DataFrame(
            scores, columns=self.card.get("targets", [f"dim_{i}" for i in range(scores.shape[1])])
        )


def register_bundle(cfg: ModelConfig, bundle_dir: Path, alias: str, description: str) -> dict[str, Any]:
    """Log the bundle as a pyfunc model in the active run, register a new version and set ``alias``."""
    requirements = cfg.root / "requirements.txt"
    info = mlflow.pyfunc.log_model(
        name="movie_dna_model",
        python_model=DnaPyfunc(),
        artifacts={"bundle": str(bundle_dir)},
        code_paths=[str(PACKAGE_DIR)],
        pip_requirements=str(requirements) if requirements.is_file() else None,
    )
    version = mlflow.register_model(info.model_uri, cfg.registered_model_name)
    client = MlflowClient()
    client.set_registered_model_alias(cfg.registered_model_name, alias, version.version)
    client.update_model_version(cfg.registered_model_name, version.version, description=description)
    LOGGER.info("Registered %s version %s with alias '%s'", cfg.registered_model_name, version.version, alias)
    return {
        "name": cfg.registered_model_name,
        "version": str(version.version),
        "alias": alias,
        "model_uri": info.model_uri,
    }


def registry_summary(cfg: ModelConfig) -> list[dict[str, Any]]:
    """All registered versions with their aliases (empty if nothing is registered yet)."""
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
        aliases = client.get_registered_model(cfg.registered_model_name).aliases
    except mlflow.exceptions.MlflowException:
        return []
    by_version: dict[str, list[str]] = {}
    for alias_name, version in aliases.items():
        by_version.setdefault(str(version), []).append(alias_name)
    return sorted(
        (
            {
                "version": str(v.version),
                "aliases": by_version.get(str(v.version), []),
                "run_id": v.run_id,
                "description": v.description,
            }
            for v in versions
        ),
        key=lambda item: int(item["version"]),
    )
