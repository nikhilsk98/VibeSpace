"""Run the Great Expectations suites and stop the pipeline on failure.

``validate_raw`` checks the MovieLens inputs, the TMDB table and the join
match rate; ``validate_processed`` checks the catalog, DNA labels and plot
embeddings. Results are saved as JSON (``data/metrics``), the suites are
exported to ``gx/expectations`` and any failed suite raises
:class:`ValidationFailedError`, which stops downstream Airflow tasks.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.build_dna_labels import DNA_DIMENSIONS, DNA_LABELS_FILE
from scripts.build_tmdb_table import TMDB_TABLE_FILE
from scripts.clean import CANONICAL_GENRES, CLEAN_FILE
from scripts.config import PipelineConfig
from scripts.encode_plots import EMBEDDINGS_FILE, EMBEDDINGS_INDEX_FILE
from scripts.exceptions import ValidationFailedError
from scripts.gx_suites import ExpectationSpec, processed_suites, raw_suites
from scripts.io_utils import ensure_dir, read_csv_checked, read_json, read_parquet, write_json
from scripts.join_ids import JOIN_STATS_FILE
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

VALIDATION_RAW_FILE = "validation_raw.json"
VALIDATION_PROCESSED_FILE = "validation_processed.json"


def _gx() -> Any:
    """Import GX lazily with anonymous usage statistics disabled."""
    os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")
    import great_expectations as gx

    return gx


def _summarise_result(result: Any) -> dict[str, Any]:
    """Keep the useful parts of one GX expectation result."""
    cfg = result.expectation_config
    observed = result.result or {}
    return {
        "expectation": cfg.type,
        "kwargs": {k: v for k, v in cfg.kwargs.items() if k != "batch_id"},
        "success": bool(result.success),
        "unexpected_count": observed.get("unexpected_count"),
        "unexpected_percent": observed.get("unexpected_percent"),
        "observed_value": observed.get("observed_value"),
        "partial_unexpected_list": [str(v) for v in (observed.get("partial_unexpected_list") or [])[:10]],
    }


def run_suites(frames: dict[str, pd.DataFrame], suites: dict[str, list[ExpectationSpec]]
               ) -> dict[str, Any]:
    """Validate each frame with the suite of the same name; return a JSON-ready report."""
    gx = _gx()
    from great_expectations.data_context.types.base import ProgressBarsConfig

    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False, metric_calculations=False)
    source = context.data_sources.add_pandas("vibespace")
    report: dict[str, Any] = {"validated_at": datetime.now(UTC).isoformat(), "suites": {}}
    for name, specs in suites.items():
        if name not in frames:
            raise KeyError(f"No data frame provided for suite {name!r}")
        suite = context.suites.add(gx.ExpectationSuite(name=name))
        for spec in specs:
            suite.add_expectation(getattr(gx.expectations, spec.expectation)(**spec.kwargs))
        batch = source.add_dataframe_asset(name=name).add_batch_definition_whole_dataframe("all")
        validation = context.validation_definitions.add(gx.ValidationDefinition(name=name, data=batch, suite=suite))
        result = validation.run(batch_parameters={"dataframe": frames[name]})
        details = [_summarise_result(r) for r in result.results]
        failed = [d for d in details if not d["success"]]
        report["suites"][name] = {
            "success": bool(result.success),
            "rows": int(len(frames[name])),
            "evaluated": len(details),
            "failed": len(failed),
            "failures": failed,
            "results": details,
        }
        level = logger.info if result.success else logger.error
        level("GX suite %s: %s (%d/%d expectations passed)", name,
              "PASSED" if result.success else "FAILED", len(details) - len(failed), len(details))
    report["success"] = all(s["success"] for s in report["suites"].values())
    return report


def export_suites(suites: dict[str, list[ExpectationSpec]], out_dir: Any) -> None:
    """Write each suite as a GX-style JSON file (deterministic, no random ids)."""
    out = ensure_dir(out_dir)
    for name, specs in suites.items():
        gx = _gx()
        expectations = []
        for spec in specs:
            exp = getattr(gx.expectations, spec.expectation)(**spec.kwargs)
            expectations.append({"type": exp.expectation_type, "kwargs": spec.kwargs})
        write_json(out / f"{name}.json", {"name": name, "expectations": expectations})


def raise_on_failure(report: dict[str, Any], stage: str) -> None:
    """Raise :class:`ValidationFailedError` listing every failed expectation."""
    if report["success"]:
        return
    problems = []
    for name, suite in report["suites"].items():
        for f in suite["failures"]:
            col = f["kwargs"].get("column", "<table>")
            problems.append(f"{name}.{col}: {f['expectation']} (unexpected={f['unexpected_count']})")
    raise ValidationFailedError(f"{stage} validation failed: " + "; ".join(problems))


LIST_COLUMNS = ("genres", "tmdb_genres", "keywords", "production_countries", "spoken_languages")


def _genre_list(value: Any) -> list[str]:
    """Genres of one catalog row as a list (lists / arrays as stored; missing -> empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        return [g for g in value.split("|") if g]
    return [str(g) for g in value]


def catalog_frame(catalog: pd.DataFrame) -> pd.DataFrame:
    """The clean catalog as GX checks it: list columns replaced by facts GX can validate.

    ``genre_count`` (number of genres) and ``genres_known`` (every genre in the canonical vocabulary)
    stand in for the list-valued ``genres`` column the model uses as a feature.
    """
    frame = catalog.drop(columns=[c for c in LIST_COLUMNS if c in catalog.columns])
    if "genres" in catalog.columns:
        vocab = set(CANONICAL_GENRES)
        genres = catalog["genres"].map(_genre_list)
        frame["genre_count"] = genres.map(len).astype("int64")
        frame["genres_known"] = genres.map(lambda gs: all(g in vocab for g in gs))
    return frame


def embedding_frame(embeddings: np.ndarray, index: pd.DataFrame, catalog_ids: set[int]) -> pd.DataFrame:
    """Per-row embedding facts that GX can check (dim, finiteness, norm, catalog link, row order, encoder)."""
    n = len(index)
    if embeddings.shape[0] != n:
        raise ValidationFailedError(f"Embeddings have {embeddings.shape[0]} rows but the index has {n}")
    dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    rows = index["row"].to_numpy() if "row" in index.columns else np.full(n, -1)
    return pd.DataFrame({
        "movieId": index["movieId"].astype("int64").to_numpy(),
        "dim": np.full(n, dim, dtype=np.int64),
        "is_finite": np.isfinite(embeddings).all(axis=1) if n else np.array([], dtype=bool),
        "norm": np.linalg.norm(embeddings, axis=1) if n else np.array([], dtype=float),
        "in_catalog": index["movieId"].isin(catalog_ids).to_numpy(),
        # matrix row i must be index row i (contiguous 0..n-1), the order the model relies on
        "row_in_order": rows == np.arange(n),
        "encoder": index["encoder"].astype(str).to_numpy() if "encoder" in index.columns else [None] * n,
    })


def run_validate_raw(cfg: PipelineConfig) -> dict[str, Any]:
    """Validate the raw MovieLens files and the flattened TMDB table."""
    csv_dir = cfg.movielens_csv_dir
    frames = {
        "raw_movielens_movies": read_csv_checked(csv_dir / "movies.csv", ["movieId", "title", "genres"]),
        "raw_movielens_links": read_csv_checked(csv_dir / "links.csv", ["movieId", "tmdbId"]),
        "raw_genome_tags": read_csv_checked(csv_dir / "genome-tags.csv", ["tagId", "tag"]),
        "raw_tmdb_movies": read_parquet(cfg.interim_dir / TMDB_TABLE_FILE).drop(
            columns=["spoken_languages", "tmdb_genres", "keywords", "production_countries"]),
    }
    suites = {k: v for k, v in raw_suites(cfg).items() if k != "join_quality"}
    export_suites(suites, cfg.gx_dir / "expectations")
    report = run_suites(frames, suites)
    write_json(cfg.metrics_dir / VALIDATION_RAW_FILE, report)
    raise_on_failure(report, "Raw data")
    return {"success": report["success"], "suites": {k: v["success"] for k, v in report["suites"].items()}}


def run_validate_processed(cfg: PipelineConfig) -> dict[str, Any]:
    """Validate the clean catalog, DNA labels, embeddings and the join match rate."""
    catalog = catalog_frame(read_parquet(cfg.processed_dir / CLEAN_FILE))
    labels = read_parquet(cfg.processed_dir / DNA_LABELS_FILE, columns=["movieId", "mapping_version",
                                                                         *DNA_DIMENSIONS])
    embeddings = np.load(cfg.processed_dir / EMBEDDINGS_FILE)
    index = read_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE)
    stats = read_json(cfg.interim_dir / JOIN_STATS_FILE)
    frames = {
        "processed_catalog": catalog,
        "processed_dna_labels": labels,
        "processed_embeddings": embedding_frame(embeddings, index, set(catalog["movieId"].astype(int))),
        "join_quality": pd.DataFrame([{"match_rate": stats["match_rate"]}]),
    }
    suites = processed_suites(cfg) | {"join_quality": raw_suites(cfg)["join_quality"]}
    export_suites(suites, cfg.gx_dir / "expectations")
    report = run_suites(frames, suites)
    write_json(cfg.metrics_dir / VALIDATION_PROCESSED_FILE, report)
    raise_on_failure(report, "Processed data")
    return {"success": report["success"], "suites": {k: v["success"] for k, v in report["suites"].items()}}
