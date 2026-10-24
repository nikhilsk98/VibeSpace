"""Thin task layer: one function per pipeline step.

Each function takes the Airflow context (``**context``), builds the config from
the environment plus DAG params / trigger conf, runs the pure step function and
returns a small JSON-serialisable summary (stored as an XCom). The same
functions are used by ``scripts.run_pipeline`` to run the pipeline without
Airflow.
"""

from __future__ import annotations

import os
from typing import Any

from scripts.config import PIPELINE_ROOT, PipelineConfig, load_config
from scripts.logging_utils import get_logger, log_step, setup_logging

logger = get_logger(__name__)

# DAG param name -> PipelineConfig field
PARAM_FIELDS = {
    "tmdb_stage": "tmdb_stage",
    "sample_size": "sample_size",
    "use_cache": "use_cache",
    "tmdb_max_workers": "tmdb_max_workers",
    "tmdb_rate_limit": "tmdb_rate_limit",
    "encoder_batch_size": "encoder_batch_size",
}


def _params(context: dict[str, Any]) -> dict[str, Any]:
    """DAG params overlaid with the trigger conf of the current DAG run."""
    params = dict(context.get("params") or {})
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}
    params.update(conf)
    return params


def build_config(context: dict[str, Any] | None = None) -> PipelineConfig:
    """Config for a task: environment first, then DAG params / conf overrides."""
    context = context or {}
    params = _params(context)
    overrides: dict[str, Any] = {}
    for param, field_name in PARAM_FIELDS.items():
        if param in params and (params[param] is not None or param == "sample_size"):
            overrides[field_name] = params[param]
    if "sample_size" in overrides and overrides["sample_size"] is not None:
        overrides["sample_size"] = int(overrides["sample_size"]) or None
    cfg = load_config(**overrides)
    os.environ.setdefault("HF_HOME", str(PIPELINE_ROOT / ".cache" / "huggingface"))
    setup_logging(cfg.logs_dir)
    return cfg


def _run(step: str, context: dict[str, Any], fn: Any, *args: Any) -> dict[str, Any]:
    """Build the config, log the step and run ``fn(cfg, *args)``."""
    cfg = build_config(context)
    logger.info("Step %s | stage=%s sample_size=%s use_cache=%s run_id=%s", step, cfg.tmdb_stage,
                cfg.sample_size, cfg.use_cache, context.get("run_id"))
    with log_step(logger, step):
        return fn(cfg, *args)


def download_movielens(**context: Any) -> dict[str, Any]:
    """Download and verify MovieLens 25M."""
    from scripts.acquire_movielens import acquire_movielens

    manifest = _run("download_movielens", context, acquire_movielens)
    return {"zip_md5": manifest["zip_md5"], "skipped": manifest.get("skipped", False)}


def fetch_tmdb(**context: Any) -> dict[str, Any]:
    """Fetch TMDB details for the selected stage into the cache."""
    from scripts.acquire_tmdb import run_fetch_tmdb

    return _run("fetch_tmdb", context, run_fetch_tmdb)


def build_tmdb_table(**context: Any) -> dict[str, Any]:
    """Flatten cached TMDB JSON into a table."""
    from scripts.build_tmdb_table import run_build_tmdb_table

    return _run("build_tmdb_table", context, run_build_tmdb_table)


def validate_raw(**context: Any) -> dict[str, Any]:
    """Great Expectations checks on raw inputs."""
    from scripts.validate import run_validate_raw

    return _run("validate_raw", context, run_validate_raw)


def join_ids(**context: Any) -> dict[str, Any]:
    """Join MovieLens and TMDB."""
    from scripts.join_ids import run_join_ids

    return _run("join_ids", context, run_join_ids)


def clean(**context: Any) -> dict[str, Any]:
    """Clean the joined catalog and derive slice columns."""
    from scripts.clean import run_clean

    return _run("clean", context, run_clean)


def build_dna_labels(**context: Any) -> dict[str, Any]:
    """Build Movie DNA labels from the Tag Genome."""
    from scripts.build_dna_labels import run_build_dna_labels

    return _run("build_dna_labels", context, run_build_dna_labels)


def encode_plots(**context: Any) -> dict[str, Any]:
    """Encode plot overviews into embeddings."""
    from scripts.encode_plots import run_encode_plots

    return _run("encode_plots", context, run_encode_plots)


def validate_processed(**context: Any) -> dict[str, Any]:
    """Great Expectations checks on processed outputs."""
    from scripts.validate import run_validate_processed

    return _run("validate_processed", context, run_validate_processed)


def generate_stats(**context: Any) -> dict[str, Any]:
    """Schema + statistics and drift versus the previous run."""
    from scripts.generate_stats import run_generate_stats

    return _run("generate_stats", context, run_generate_stats, context.get("run_id"))


def detect_anomalies(**context: Any) -> dict[str, Any]:
    """Detect anomalies and alert."""
    from scripts.anomalies import run_detect_anomalies

    return _run("detect_anomalies", context, run_detect_anomalies, context.get("run_id"))


def bias_slicing(**context: Any) -> dict[str, Any]:
    """Slice-based bias analysis and mitigation."""
    from scripts.bias_slicing import run_bias_slicing

    return _run("bias_slicing", context, run_bias_slicing)


def dvc_version(**context: Any) -> dict[str, Any]:
    """Commit and push outputs with DVC."""
    from scripts.dvc_versioning import version_outputs

    return _run("dvc_version", context, version_outputs)


# Execution order used outside Airflow (a valid topological order of the DAG).
PIPELINE_STEPS = (
    ("download_movielens", download_movielens),
    ("fetch_tmdb", fetch_tmdb),
    ("build_tmdb_table", build_tmdb_table),
    ("validate_raw", validate_raw),
    ("join_ids", join_ids),
    ("clean", clean),
    ("build_dna_labels", build_dna_labels),
    ("encode_plots", encode_plots),
    ("validate_processed", validate_processed),
    ("generate_stats", generate_stats),
    ("detect_anomalies", detect_anomalies),
    ("bias_slicing", bias_slicing),
    ("dvc_version", dvc_version),
)
