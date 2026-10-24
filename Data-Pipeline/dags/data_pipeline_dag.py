"""Airflow DAGs for the VibeSpace data pipeline.

Two DAGs are built from one factory:

* ``vibespace_data_pipeline`` - the optimised pipeline (parallel branches,
  concurrent TMDB requests, large encoding batches, caching).
* ``vibespace_data_pipeline_baseline`` - the same tasks chained one after the
  other with one TMDB worker, small batches and no cache. It exists only to
  measure the "before" Gantt chart (see ``docs/optimization.md``).

Every task wraps a function from ``scripts.tasks``. Tasks retry with
exponential backoff, except for errors that a retry cannot fix (bad TMDB key,
invalid DNA mapping, failed validation), and every failure triggers an alert.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from scripts import tasks
from scripts.alerts import airflow_failure_callback
from scripts.exceptions import AnomalyError, ConfigError, MappingError, TMDBAuthError, ValidationFailedError

NON_RETRYABLE = (TMDBAuthError, MappingError, ValidationFailedError, ConfigError, AnomalyError)

DEFAULT_ARGS = {
    "owner": "vibespace-team",
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=10),
    "on_failure_callback": airflow_failure_callback,
    "execution_timeout": timedelta(hours=3),
}

TASK_DOCS = {
    "download_movielens": "Download MovieLens 25M with retries, verify the MD5, extract safely, check CSV "
                          "headers and write a manifest. Skipped when the verified data is already present.",
    "fetch_tmdb": "Fetch TMDB details (+keywords, videos) for the selected stage (`tmdb_stage` param) into a "
                  "per-movie JSON cache. Thread pool + shared rate limiter, retries with backoff, 429 "
                  "Retry-After, 404 cached, 401 stops at once, fails above 5% failures.",
    "build_tmdb_table": "Flatten cached TMDB JSON into `tmdb_movies.parquet`; unreadable files are quarantined.",
    "validate_raw": "Great Expectations suites for raw MovieLens files and the TMDB table. Failure stops "
                    "downstream tasks.",
    "join_ids": "Join MovieLens to TMDB via links.csv. Unmatched movies are quarantined with a reason; year "
                "mismatches are flagged; fails below the minimum match rate.",
    "clean": "Unify genres, normalise languages, derive slice columns (language_group, era, "
             "popularity_bucket), flag data problems and quarantine duplicates / missing titles / adult titles.",
    "build_dna_labels": "Validate the Tag Genome mapping and build the 12 Movie DNA percentile labels plus the "
                        "DNA sanity report. Depends only on MovieLens, so it runs in parallel with TMDB work.",
    "encode_plots": "Encode overviews with all-MiniLM-L6-v2 (384-dim), batched and cached.",
    "validate_processed": "Great Expectations suites for the catalog, DNA labels, embeddings and match rate.",
    "generate_stats": "Schema + statistics for this run and drift versus the previous comparable run (PSI).",
    "detect_anomalies": "Missing values, outliers, invalid formats, schema and drift checks; alerts on warnings.",
    "bias_slicing": "Fairlearn slice analysis of a diagnostic model, mitigation (re-weighting / re-sampling) "
                    "and `reports/bias_report.md`.",
    "dvc_version": "`dvc commit` + `dvc push` of all pipeline outputs, then `dvc status`.",
}

DAG_DOC = """
### VibeSpace data pipeline

MovieLens 25M + TMDB -> clean catalog, 12 Movie DNA labels (from the Tag Genome) and plot embeddings,
with Great Expectations validation, statistics, anomaly alerts, Fairlearn bias analysis and DVC versioning.

**Params:** `tmdb_stage` (genome / rest / all), `sample_size` (run on N movies; empty = all),
`use_cache`, `tmdb_max_workers`, `tmdb_rate_limit`, `encoder_batch_size`.
"""


def no_retry_on_permanent_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Turn errors that retries cannot fix into ``AirflowFailException`` (fail immediately)."""

    @functools.wraps(fn)
    def wrapper(**context: Any) -> Any:
        try:
            return fn(**context)
        except NON_RETRYABLE as exc:
            raise AirflowFailException(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def pipeline_params(max_workers: int, batch_size: int, use_cache: bool, rate_limit: float) -> dict[str, Param]:
    """DAG params (can be overridden in the trigger form / conf)."""
    return {
        "tmdb_stage": Param(os.environ.get("TMDB_STAGE", "genome") or "genome", type="string",
                            enum=["genome", "rest", "all"], description="Which movies to fetch from TMDB"),
        "sample_size": Param(None, type=["null", "integer"], minimum=1,
                             description="Run on N movies only (empty = every movie of the stage)"),
        "use_cache": Param(use_cache, type="boolean", description="Reuse cached TMDB responses and embeddings"),
        "tmdb_max_workers": Param(max_workers, type="integer", minimum=1, maximum=64),
        "tmdb_rate_limit": Param(rate_limit, type="number", exclusiveMinimum=0, description="TMDB requests per second"),
        "encoder_batch_size": Param(batch_size, type="integer", minimum=1, maximum=1024),
    }


def build_dag(dag_id: str, layout: str, schedule: str | None, max_workers: int, batch_size: int,
              use_cache: bool, rate_limit: float, description: str) -> DAG:
    """Create the pipeline DAG with a ``parallel`` or ``sequential`` layout."""
    dag = DAG(
        dag_id=dag_id,
        description=description,
        default_args=DEFAULT_ARGS,
        start_date=datetime(2024, 1, 1),
        schedule=schedule,
        catchup=False,
        max_active_runs=1,
        params=pipeline_params(max_workers, batch_size, use_cache, rate_limit),
        tags=["vibespace", "data-pipeline", layout],
        doc_md=DAG_DOC,
    )
    with dag:
        ops = {
            name: PythonOperator(task_id=name, python_callable=no_retry_on_permanent_errors(fn),
                                 doc_md=TASK_DOCS[name])
            for name, fn in tasks.PIPELINE_STEPS
        }
        if layout == "sequential":
            names = [name for name, _ in tasks.PIPELINE_STEPS]
            for upstream, downstream in zip(names, names[1:]):
                ops[upstream] >> ops[downstream]
        else:
            ops["download_movielens"] >> [ops["fetch_tmdb"], ops["build_dna_labels"]]
            ops["fetch_tmdb"] >> ops["build_tmdb_table"] >> ops["validate_raw"] >> ops["join_ids"]
            ops["join_ids"] >> ops["clean"] >> ops["encode_plots"]
            [ops["build_dna_labels"], ops["encode_plots"]] >> ops["validate_processed"]
            ops["validate_processed"] >> [ops["generate_stats"], ops["bias_slicing"]]
            ops["generate_stats"] >> ops["detect_anomalies"]
            [ops["detect_anomalies"], ops["bias_slicing"]] >> ops["dvc_version"]
    return dag


optimized_dag = build_dag(
    dag_id="vibespace_data_pipeline",
    layout="parallel",
    schedule=os.environ.get("VIBESPACE_SCHEDULE") or None,
    max_workers=8,
    batch_size=128,
    use_cache=True,
    rate_limit=40.0,  # TMDB allows roughly 50 requests/second per IP; stay below it
    description="VibeSpace data pipeline (optimised: parallel branches, concurrency, caching)",
)

baseline_dag = build_dag(
    dag_id="vibespace_data_pipeline_baseline",
    layout="sequential",
    schedule=None,
    max_workers=1,
    batch_size=8,
    use_cache=False,
    rate_limit=20.0,
    description="VibeSpace data pipeline baseline (sequential, for the Gantt before/after comparison)",
)
