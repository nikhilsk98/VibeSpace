"""Great Expectations (GX Core 1.x) suite definitions for raw and processed data.

Each suite is declared as a list of :class:`ExpectationSpec` so the rules are
easy to read, test and export. ``scripts.validate`` turns them into GX
``ExpectationSuite`` objects and runs them. Thresholds come from the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from scripts.build_dna_labels import DNA_DIMENSIONS
from scripts.clean import CANONICAL_GENRES, ERAS, LANGUAGE_GROUPS, MIN_VALID_YEAR, POPULARITY_BUCKETS
from scripts.config import PipelineConfig

# Flag columns the model pipeline uses as features (Model-Pipeline/configs/model_config.yaml, features.flag_columns).
MODEL_FLAG_COLUMNS = (
    "flag_short_overview",
    "flag_missing_overview",
    "flag_missing_runtime",
    "flag_runtime_outlier",
    "flag_year_mismatch",
    "flag_unknown_language",
)


@dataclass(frozen=True)
class ExpectationSpec:
    """One expectation: the GX class name plus its keyword arguments."""

    expectation: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def _not_null(col: str, mostly: float = 1.0) -> ExpectationSpec:
    kwargs: dict[str, Any] = {"column": col}
    if mostly < 1.0:
        kwargs["mostly"] = mostly
    return ExpectationSpec("ExpectColumnValuesToNotBeNull", kwargs)


def _unique(col: str) -> ExpectationSpec:
    return ExpectationSpec("ExpectColumnValuesToBeUnique", {"column": col})


def _exists(col: str) -> ExpectationSpec:
    return ExpectationSpec("ExpectColumnToExist", {"column": col})


def _between(col: str, lo: float | None, hi: float | None, mostly: float = 1.0) -> ExpectationSpec:
    kwargs: dict[str, Any] = {"column": col, "min_value": lo, "max_value": hi}
    if mostly < 1.0:
        kwargs["mostly"] = mostly
    return ExpectationSpec("ExpectColumnValuesToBeBetween", kwargs)


def _in_set(col: str, values: list[Any]) -> ExpectationSpec:
    return ExpectationSpec("ExpectColumnValuesToBeInSet", {"column": col, "value_set": values})


def _regex(col: str, pattern: str, mostly: float = 1.0) -> ExpectationSpec:
    kwargs: dict[str, Any] = {"column": col, "regex": pattern}
    if mostly < 1.0:
        kwargs["mostly"] = mostly
    return ExpectationSpec("ExpectColumnValuesToMatchRegex", kwargs)


def _min_rows(n: int = 1) -> ExpectationSpec:
    return ExpectationSpec("ExpectTableRowCountToBeBetween", {"min_value": n})


def raw_suites(cfg: PipelineConfig) -> dict[str, list[ExpectationSpec]]:
    """Suites for raw inputs (MovieLens files, TMDB table) and the join match rate.

    ``join_quality`` runs with the processed suites because the join happens
    after raw validation.
    """
    return {
        "raw_movielens_movies": [
            _min_rows(), _exists("movieId"), _exists("title"), _exists("genres"),
            _not_null("movieId"), _unique("movieId"), _not_null("title"),
        ],
        "raw_movielens_links": [
            _min_rows(), _not_null("movieId"), _unique("movieId"), _not_null("tmdbId", mostly=0.98),
            _between("tmdbId", 1, None),
        ],
        "raw_genome_tags": [
            _min_rows(), _not_null("tagId"), _unique("tagId"), _not_null("tag"), _unique("tag"),
        ],
        "raw_tmdb_movies": [
            _min_rows(), _not_null("tmdb_id"), _unique("tmdb_id"),
            _between("runtime", 0, 1000, mostly=0.99),
            _between("vote_count", 0, None), _between("vote_average", 0, 10),
            _regex("original_language", r"^[a-z]{2}$", mostly=0.99),
            _regex("release_date", r"^\d{4}-\d{2}-\d{2}$", mostly=0.95),
        ],
        "join_quality": [
            _between("match_rate", cfg.min_match_rate, 1.0),
        ],
    }


def processed_suites(cfg: PipelineConfig) -> dict[str, list[ExpectationSpec]]:
    """Suites for the processed catalog, DNA labels and plot embeddings."""
    max_year = datetime.now(UTC).year + 1
    dna = [_min_rows(), _not_null("movieId"), _unique("movieId"), _not_null("mapping_version")]
    for dim in DNA_DIMENSIONS:
        dna += [_not_null(dim), _between(dim, 0.0, 1.0)]
    return {
        "processed_catalog": [
            _min_rows(), _not_null("movieId"), _unique("movieId"), _not_null("tmdbId"), _unique("tmdbId"),
            _not_null("title"),
            _between("runtime", 1, cfg.runtime_outlier_minutes, mostly=0.95),
            _between("year", MIN_VALID_YEAR, max_year),
            _regex("language", r"^(?:[a-z]{2}|unknown)$"),
            _in_set("language_group", list(LANGUAGE_GROUPS)),
            _in_set("era", list(ERAS)),
            _in_set("popularity_bucket", list(POPULARITY_BUCKETS)),
            _in_set("primary_genre", [*CANONICAL_GENRES, "unknown"]),
            # inputs of the model pipeline's features (genres, popularity group, flags)
            _between("genre_count", 1, None),
            _in_set("genres_known", [True]),
            _not_null("popularity"), _between("popularity", 0, 10_000),
            _not_null("vote_count"), _between("vote_count", 0, 10_000_000),
            _not_null("vote_average"), _between("vote_average", 0, 10),
            *[spec for flag in MODEL_FLAG_COLUMNS
              for spec in (_exists(flag), _not_null(flag), _in_set(flag, [False, True]))],
        ],
        "processed_dna_labels": dna,
        "processed_embeddings": [
            _min_rows(), _not_null("movieId"), _unique("movieId"),
            _in_set("dim", [cfg.embedding_dim]),
            _in_set("is_finite", [True]),
            _between("norm", 0.99, 1.01),
            _in_set("in_catalog", [True]),
            _in_set("row_in_order", [True]),
            _not_null("encoder"), _in_set("encoder", [cfg.encoder_model]),
        ],
    }
