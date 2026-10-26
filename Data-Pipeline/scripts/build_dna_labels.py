"""Build the 12 Movie DNA labels from the MovieLens Tag Genome.

The mapping in ``configs/movie_dna_mapping_v1.yaml`` lists positive and inverse
genome tags per dimension. For each movie:

    raw score = mean(relevance of positive tags + (1 - relevance of inverse tags))

and the raw scores are then turned into percentiles across all Tag Genome
movies (0 = lowest, 1 = highest). Only the needed tags are loaded from the
~15M-row ``genome-scores.csv`` (streamed in batches with pyarrow).

The Tag Genome is used only to build labels, never as a model input feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import yaml

from scripts.config import PipelineConfig
from scripts.exceptions import MappingError
from scripts.io_utils import read_csv_checked, write_json, write_parquet
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

DNA_DIMENSIONS = (
    "energy", "intensity", "pace", "humor", "darkness", "hopefulness",
    "emotional_intensity", "romance", "suspense", "complexity", "realism", "inspiration",
)
DNA_LABELS_FILE = "movie_dna_labels.parquet"
DNA_SUMMARY_FILE = "dna_labels_summary.json"
MIN_POSITIVE_TAGS = 3
MAX_POSITIVE_TAGS = 6


@dataclass(frozen=True)
class DnaMapping:
    """Validated mapping from genome tags to DNA dimensions."""

    version: str
    status: str
    positive: dict[str, list[str]]
    inverse: dict[str, list[str]]

    def all_tags(self) -> set[str]:
        """Every tag used by any dimension."""
        tags: set[str] = set()
        for dim in DNA_DIMENSIONS:
            tags.update(self.positive[dim])
            tags.update(self.inverse.get(dim, []))
        return tags


def load_mapping(path: Path) -> DnaMapping:
    """Read and structurally check the mapping YAML (tags are checked separately)."""
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"DNA mapping file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    dims = raw.get("dimensions")
    if not isinstance(dims, dict):
        raise MappingError("Mapping must contain a 'dimensions' section")
    missing = [d for d in DNA_DIMENSIONS if d not in dims]
    extra = [d for d in dims if d not in DNA_DIMENSIONS]
    if missing or extra:
        raise MappingError(f"Mapping dimensions mismatch: missing={missing}, unexpected={extra}")

    positive: dict[str, list[str]] = {}
    inverse: dict[str, list[str]] = {}
    for dim in DNA_DIMENSIONS:
        spec = dims[dim] or {}
        pos = [str(t) for t in spec.get("positive") or []]
        inv = [str(t) for t in spec.get("inverse") or []]
        if not MIN_POSITIVE_TAGS <= len(pos) <= MAX_POSITIVE_TAGS:
            raise MappingError(
                f"Dimension {dim!r} needs {MIN_POSITIVE_TAGS}-{MAX_POSITIVE_TAGS} positive tags, has {len(pos)}"
            )
        if set(pos) & set(inv):
            raise MappingError(f"Dimension {dim!r} uses {sorted(set(pos) & set(inv))} as positive and inverse")
        positive[dim], inverse[dim] = pos, inv

    _check_shared_tags(positive, inverse, raw.get("allowed_shared_tags") or {})
    return DnaMapping(str(raw.get("version", "unknown")), str(raw.get("status", "")), positive, inverse)


def _check_shared_tags(positive: dict[str, list[str]], inverse: dict[str, list[str]],
                       allowed: dict[str, Any]) -> None:
    """Reject tags used by two dimensions unless explicitly allowed with a reason."""
    owners: dict[str, list[str]] = {}
    for dim in DNA_DIMENSIONS:
        for tag in positive[dim] + inverse[dim]:
            owners.setdefault(tag, []).append(dim)
    shared = {t: d for t, d in owners.items() if len(set(d)) > 1}
    unjustified = {t: d for t, d in shared.items() if not str(allowed.get(t) or "").strip()}
    if unjustified:
        raise MappingError(f"Tags reused across dimensions without justification: {unjustified}")


def validate_mapping_tags(mapping: DnaMapping, genome_tags: pd.DataFrame) -> dict[str, int]:
    """Check every mapping tag exists exactly in ``genome-tags.csv``; return tag -> tagId."""
    lookup = dict(zip(genome_tags["tag"].astype(str), genome_tags["tagId"].astype(int)))
    unknown = sorted(t for t in mapping.all_tags() if t not in lookup)
    if unknown:
        raise MappingError(
            f"{len(unknown)} mapping tag(s) do not exist in genome-tags.csv: {unknown}. "
            "Fix configs/movie_dna_mapping_v1.yaml (tags are case-sensitive)."
        )
    return {t: lookup[t] for t in mapping.all_tags()}


def load_genome_relevance(genome_scores_csv: Path, tag_ids: list[int],
                          block_size: int = 16 << 20) -> pd.DataFrame:
    """Stream ``genome-scores.csv`` and return a movieId x tagId relevance matrix.

    Only rows for ``tag_ids`` are kept, so memory stays small even though the
    file has ~15M rows.
    """
    wanted = pa.array(sorted(set(tag_ids)), type=pa.int64())
    reader = pacsv.open_csv(
        genome_scores_csv,
        read_options=pacsv.ReadOptions(block_size=block_size),
        convert_options=pacsv.ConvertOptions(
            column_types={"movieId": pa.int64(), "tagId": pa.int64(), "relevance": pa.float64()},
            include_columns=["movieId", "tagId", "relevance"],
        ),
    )
    parts: list[pa.Table] = []
    for batch in reader:
        table = pa.Table.from_batches([batch])
        parts.append(table.filter(pc.is_in(table["tagId"], value_set=wanted)))
    if not parts:
        return pd.DataFrame()
    long = pa.concat_tables(parts).to_pandas()
    if long.empty:
        raise MappingError("None of the mapped tags have relevance scores in genome-scores.csv")
    wide = long.pivot_table(index="movieId", columns="tagId", values="relevance", aggfunc="mean")
    return wide.sort_index()


def compute_raw_scores(relevance: pd.DataFrame, mapping: DnaMapping, tag_ids: dict[str, int]) -> pd.DataFrame:
    """Raw DNA score per movie and dimension (before percentiles)."""
    out = pd.DataFrame(index=relevance.index)
    for dim in DNA_DIMENSIONS:
        terms = [relevance[tag_ids[t]] for t in mapping.positive[dim] if tag_ids[t] in relevance]
        terms += [1.0 - relevance[tag_ids[t]] for t in mapping.inverse[dim] if tag_ids[t] in relevance]
        expected = len(mapping.positive[dim]) + len(mapping.inverse[dim])
        if len(terms) != expected:
            raise MappingError(f"Dimension {dim!r}: only {len(terms)} of {expected} tags have scores")
        out[dim] = pd.concat(terms, axis=1).mean(axis=1, skipna=True)
    return out


def to_percentiles(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert each column to a percentile in [0, 1] (ties share the average rank)."""
    n = len(raw)
    if n == 0:
        return raw.copy()
    if n == 1:
        return pd.DataFrame(0.5, index=raw.index, columns=raw.columns)
    ranks = raw.rank(method="average")
    return ((ranks - 1.0) / (n - 1.0)).clip(0.0, 1.0)


def build_labels(relevance: pd.DataFrame, mapping: DnaMapping, tag_ids: dict[str, int]) -> pd.DataFrame:
    """Return the label table: movieId, 12 percentile scores, 12 raw scores, version."""
    raw = compute_raw_scores(relevance, mapping, tag_ids)
    pct = to_percentiles(raw)
    labels = pct.copy()
    for dim in DNA_DIMENSIONS:
        labels[f"{dim}_raw"] = raw[dim]
    labels["mapping_version"] = mapping.version
    labels = labels.reset_index().rename(columns={"index": "movieId"})
    labels["movieId"] = labels["movieId"].astype("int64")
    if not np.isfinite(labels[list(DNA_DIMENSIONS)].to_numpy()).all():
        raise MappingError("DNA labels contain NaN or infinite values")
    return labels


def run_build_dna_labels(cfg: PipelineConfig) -> dict[str, Any]:
    """Validate the mapping, build labels for all Tag Genome movies and write the report."""
    from scripts.dna_sanity_report import write_sanity_report

    csv_dir = cfg.movielens_csv_dir
    mapping = load_mapping(cfg.dna_mapping_path)
    genome_tags = read_csv_checked(csv_dir / "genome-tags.csv", ["tagId", "tag"])
    tag_ids = validate_mapping_tags(mapping, genome_tags)
    logger.info("Mapping %s valid: %d tags across %d dimensions", mapping.version, len(tag_ids), len(DNA_DIMENSIONS))

    relevance = load_genome_relevance(csv_dir / "genome-scores.csv", list(tag_ids.values()))
    relevance.index.name = "movieId"
    labels = build_labels(relevance, mapping, tag_ids)
    write_parquet(cfg.processed_dir / DNA_LABELS_FILE, labels)

    movies = read_csv_checked(csv_dir / "movies.csv", ["movieId", "title", "genres"])
    flagged = write_sanity_report(labels, movies, mapping, cfg.reports_dir / "dna_sanity_report.md")
    summary = {
        "mapping_version": mapping.version,
        "movies_labelled": int(len(labels)),
        "tags_used": len(tag_ids),
        "flagged_dimensions": flagged,
        "raw_score_means": {d: round(float(labels[f"{d}_raw"].mean()), 4) for d in DNA_DIMENSIONS},
    }
    write_json(cfg.processed_dir / DNA_SUMMARY_FILE, summary)
    logger.info("DNA labels built for %d movies; flagged dimensions: %s", len(labels), flagged or "none")
    return summary
