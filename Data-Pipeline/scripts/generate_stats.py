"""Per-run schema and statistics, compared with the previous comparable run.

Each run writes ``data/metrics/stats/<profile>/stats_<run_id>.json`` with the
schema, row counts, per-column statistics (null rates, numeric distributions,
top categories) and slice counts, then compares it with the previous run of the
same profile (TMDB stage + sample size) to record drift:

* schema changes (added / removed columns, dtype changes)
* null-rate jumps per column
* row-count changes
* distribution shift of slice columns, measured with PSI
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_dna_labels import DNA_DIMENSIONS, DNA_LABELS_FILE
from scripts.clean import CLEAN_FILE
from scripts.config import PipelineConfig
from scripts.encode_plots import EMBEDDINGS_INDEX_FILE
from scripts.io_utils import ensure_dir, read_json, read_parquet, write_json
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

SLICE_COLUMNS = ("language_group", "era", "primary_genre", "popularity_bucket")
LATEST_FILE = "latest.json"
PSI_EPS = 1e-4


def run_profile(stage: str, sample_size: int | None) -> str:
    """Name of the comparison group, e.g. ``genome_all`` or ``genome_n500``."""
    return f"{stage}_{'all' if not sample_size else f'n{sample_size}'}"


def _is_listlike(series: pd.Series) -> bool:
    """True when the first non-null value is a list/array (e.g. genres)."""
    non_null = series.dropna()
    return bool(len(non_null)) and isinstance(non_null.iloc[0], list | tuple | np.ndarray)


def column_stats(series: pd.Series) -> dict[str, Any]:
    """Statistics for one column (numeric, boolean, categorical or list)."""
    n = len(series)
    nulls = int(series.isna().sum())
    out: dict[str, Any] = {"dtype": str(series.dtype), "count": n, "null_count": nulls,
                           "null_rate": round(nulls / n, 6) if n else 0.0}
    if _is_listlike(series):
        lengths = series.dropna().map(len)
        empty = int((lengths == 0).sum())
        out.update({"kind": "list", "empty_count": empty, "mean_length": round(float(lengths.mean()), 4)})
        return out
    if pd.api.types.is_bool_dtype(series):
        out.update({"kind": "boolean", "true_count": int(series.fillna(False).sum())})
        return out
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna().astype("float64")
        out["kind"] = "numeric"
        if len(values):
            q = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
            out.update({"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
                        "std": float(values.std()) if len(values) > 1 else 0.0,
                        "p01": float(q[0.01]), "p25": float(q[0.25]), "p50": float(q[0.5]),
                        "p75": float(q[0.75]), "p99": float(q[0.99]), "zero_count": int((values == 0).sum())})
        return out
    text = series.dropna().astype(str)
    top = text.value_counts().head(10)
    out.update({"kind": "categorical", "n_unique": int(text.nunique()),
                "top_values": {str(k): int(v) for k, v in top.items()}})
    return out


def dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Schema plus per-column statistics for one table."""
    return {
        "rows": int(len(df)),
        "schema": {c: str(t) for c, t in df.dtypes.items()},
        "columns": {c: column_stats(df[c]) for c in df.columns},
    }


def distribution(series: pd.Series) -> dict[str, float]:
    """Normalised value distribution of a categorical column."""
    counts = series.fillna("missing").astype(str).value_counts()
    total = counts.sum()
    return {str(k): float(v / total) for k, v in counts.items()} if total else {}


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = PSI_EPS) -> float:
    """Population stability index between two categorical distributions."""
    keys = set(expected) | set(actual)
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), eps)
        a = max(actual.get(k, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return round(total, 6)


def compare_runs(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    """Drift report between two stats documents (empty when there is no previous run)."""
    if not previous:
        return {"previous_run_id": None, "note": "no previous run for this profile"}
    drift: dict[str, Any] = {"previous_run_id": previous.get("run_id"), "datasets": {}, "psi": {}}
    for name, cur in current["datasets"].items():
        prev = previous.get("datasets", {}).get(name)
        if prev is None:
            drift["datasets"][name] = {"note": "dataset not present in previous run"}
            continue
        prev_schema, cur_schema = prev["schema"], cur["schema"]
        null_changes = {}
        for col, stats in cur["columns"].items():
            if col in prev["columns"]:
                delta = stats["null_rate"] - prev["columns"][col]["null_rate"]
                if abs(delta) > 1e-9:
                    null_changes[col] = round(delta, 6)
        drift["datasets"][name] = {
            "rows_previous": prev["rows"],
            "rows_current": cur["rows"],
            "row_change_pct": round((cur["rows"] - prev["rows"]) / prev["rows"], 6) if prev["rows"] else None,
            "added_columns": sorted(set(cur_schema) - set(prev_schema)),
            "removed_columns": sorted(set(prev_schema) - set(cur_schema)),
            "dtype_changes": {c: [prev_schema[c], cur_schema[c]] for c in set(cur_schema) & set(prev_schema)
                              if prev_schema[c] != cur_schema[c]},
            "null_rate_changes": null_changes,
        }
    for col, dist in current.get("slice_distributions", {}).items():
        prev_dist = previous.get("slice_distributions", {}).get(col)
        if prev_dist:
            drift["psi"][col] = psi(prev_dist, dist)
    return drift


def build_stats(catalog: pd.DataFrame, labels: pd.DataFrame, emb_index: pd.DataFrame, run_id: str,
                profile: str) -> dict[str, Any]:
    """Assemble the statistics document for one run."""
    labelled = catalog["movieId"].isin(set(labels["movieId"]))
    embedded = catalog["movieId"].isin(set(emb_index["movieId"]))
    return {
        "run_id": run_id,
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat(),
        "datasets": {
            "catalog": dataset_stats(catalog),
            "dna_labels": dataset_stats(labels[["movieId", *DNA_DIMENSIONS]]),
            "embeddings_index": dataset_stats(emb_index[["movieId"]]),
        },
        "coverage": {
            "catalog_movies": int(len(catalog)),
            "with_dna_labels": int(labelled.sum()),
            "with_embeddings": int(embedded.sum()),
            "label_coverage": round(float(labelled.mean()), 6) if len(catalog) else 0.0,
            "embedding_coverage": round(float(embedded.mean()), 6) if len(catalog) else 0.0,
        },
        "slice_counts": {c: {str(k): int(v) for k, v in catalog[c].value_counts().items()}
                         for c in SLICE_COLUMNS if c in catalog},
        "slice_distributions": {c: distribution(catalog[c]) for c in SLICE_COLUMNS if c in catalog},
    }


def stats_dir(cfg: PipelineConfig, profile: str) -> Path:
    """Folder that holds the stats history of a profile."""
    return cfg.metrics_dir / "stats" / profile


def load_previous(cfg: PipelineConfig, profile: str) -> dict[str, Any] | None:
    """Most recent stats document of ``profile``, if any."""
    latest = stats_dir(cfg, profile) / LATEST_FILE
    return read_json(latest) if latest.is_file() else None


def run_generate_stats(cfg: PipelineConfig, run_id: str | None = None) -> dict[str, Any]:
    """Compute stats for the current outputs, record drift and update ``latest.json``."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_run_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_id)
    profile = run_profile(cfg.tmdb_stage, cfg.sample_size)
    catalog = read_parquet(cfg.processed_dir / CLEAN_FILE)
    labels = read_parquet(cfg.processed_dir / DNA_LABELS_FILE)
    emb_index = read_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE)

    stats = build_stats(catalog, labels, emb_index, safe_run_id, profile)
    previous = load_previous(cfg, profile)
    stats["drift"] = compare_runs(previous, stats)
    out_dir = ensure_dir(stats_dir(cfg, profile))
    write_json(out_dir / f"stats_{safe_run_id}.json", stats)
    write_json(out_dir / LATEST_FILE, stats)
    write_json(cfg.metrics_dir / "stats_current.json", stats)
    logger.info("Stats for %s (%s): catalog rows=%d, coverage=%s, psi=%s", safe_run_id, profile,
                stats["datasets"]["catalog"]["rows"], stats["coverage"], stats["drift"].get("psi"))
    return {"run_id": safe_run_id, "profile": profile, "coverage": stats["coverage"],
            "psi": stats["drift"].get("psi", {}), "previous_run_id": stats["drift"].get("previous_run_id")}
