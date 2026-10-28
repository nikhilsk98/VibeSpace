"""Clean the joined catalog and derive the slice columns used for bias analysis.

* genres from MovieLens and TMDB are mapped to one canonical vocabulary;
* language codes are normalised to ISO 639-1 (or ``unknown``);
* slice columns ``language_group``, ``era`` and ``popularity_bucket`` are added;
* data problems are *flagged* (``flag_*`` columns), never silently fixed;
* duplicates, rows without any title and adult titles are quarantined.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.config import PipelineConfig
from scripts.io_utils import read_parquet, write_json, write_parquet
from scripts.join_ids import JOINED_FILE
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

CLEAN_FILE = "movies_clean.parquet"
CLEAN_QUARANTINE_FILE = "clean_quarantine.parquet"
CLEAN_SUMMARY_FILE = "clean_summary.json"

INDIAN_REGIONAL = frozenset({"ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "or", "as"})
LANGUAGE_GROUPS = ("english", "hindi", "indian_regional", "other_international", "unknown")
ERAS = ("pre_1980", "1980_1999", "2000_plus", "unknown")
POPULARITY_BUCKETS = ("low", "mid", "high")
MIN_VALID_YEAR = 1874  # earliest surviving motion picture
_LANG_RE = re.compile(r"^[a-z]{2}$")
# TMDB uses a few non-standard codes.
LANGUAGE_ALIASES = {"cn": "zh", "iw": "he", "in": "id", "ji": "yi", "mo": "ro", "sh": "sr"}

GENRE_MAP: dict[str, str | None] = {
    # MovieLens vocabulary
    "action": "action", "adventure": "adventure", "animation": "animation", "children": "family",
    "comedy": "comedy", "crime": "crime", "documentary": "documentary", "drama": "drama",
    "fantasy": "fantasy", "film-noir": "film_noir", "horror": "horror", "imax": None,
    "musical": "music", "mystery": "mystery", "romance": "romance", "sci-fi": "science_fiction",
    "thriller": "thriller", "war": "war", "western": "western", "(no genres listed)": None,
    # TMDB vocabulary
    "family": "family", "history": "history", "music": "music", "science fiction": "science_fiction",
    "tv movie": "tv_movie",
}
CANONICAL_GENRES = tuple(sorted({g for g in GENRE_MAP.values() if g}))


def map_genre(name: Any) -> str | None:
    """Map a MovieLens or TMDB genre name to the canonical vocabulary."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower()
    if key in GENRE_MAP:
        return GENRE_MAP[key]
    logger.debug("Unknown genre %r ignored", name)
    return None


def _as_list(value: Any) -> list[str]:
    """Return list-like values (lists, numpy arrays) as a Python list."""
    if value is None:
        return []
    if isinstance(value, list | tuple | np.ndarray):
        return [str(v) for v in value]
    return []


def unify_genres(ml_genres: Any, tmdb_genres: Any) -> tuple[list[str], str]:
    """Return (sorted canonical genres, primary genre).

    The primary genre is the first TMDB genre (TMDB orders by relevance), then
    the first MovieLens genre, else ``"unknown"``.
    """
    ml_list = ml_genres.split("|") if isinstance(ml_genres, str) else []
    tmdb_mapped = [g for g in (map_genre(x) for x in _as_list(tmdb_genres)) if g]
    ml_mapped = [g for g in (map_genre(x) for x in ml_list) if g]
    genres = sorted(set(tmdb_mapped) | set(ml_mapped))
    primary = (tmdb_mapped or ml_mapped or ["unknown"])[0]
    return genres, primary


def normalize_language(code: Any) -> str:
    """Normalise a language code to lowercase ISO 639-1, or ``"unknown"``."""
    if not isinstance(code, str):
        return "unknown"
    value = code.strip().lower().replace("_", "-").split("-")[0]
    value = LANGUAGE_ALIASES.get(value, value)
    if value in {"", "xx", "zxx", "und"} or not _LANG_RE.match(value):
        return "unknown"
    return value


def language_group(code: str) -> str:
    """Map a normalised language code to its slice group."""
    if code == "en":
        return "english"
    if code == "hi":
        return "hindi"
    if code in INDIAN_REGIONAL:
        return "indian_regional"
    if code == "unknown":
        return "unknown"
    return "other_international"


def era_of(year: Any) -> str:
    """Return the era slice for a release year."""
    if year is None or pd.isna(year):
        return "unknown"
    if year < 1980:
        return "pre_1980"
    if year < 2000:
        return "1980_1999"
    return "2000_plus"


def popularity_buckets(vote_count: pd.Series) -> pd.Series:
    """Bucket by vote count rank: top 20% ``high``, bottom 40% ``low``, else ``mid``."""
    if vote_count.empty:
        return pd.Series([], dtype="object", index=vote_count.index)
    pct = vote_count.fillna(0).rank(method="average", pct=True)
    bucket = np.where(pct > 0.8, "high", np.where(pct <= 0.4, "low", "mid"))
    return pd.Series(bucket, index=vote_count.index, dtype="object")


def _valid_year(year: pd.Series, max_year: int) -> pd.Series:
    """Boolean mask of plausible release years."""
    return year.between(MIN_VALID_YEAR, max_year)


def clean_catalog(joined: pd.DataFrame, runtime_outlier: int = 400, short_overview_chars: int = 40,
                  max_year: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (clean catalog, quarantine, summary) for a joined table."""
    max_year = max_year or datetime.now(UTC).year + 1
    df = joined.copy()

    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    reasons[df["movieId"].duplicated(keep="first")] = "duplicate_movie_id"
    title = df["tmdb_title"].where(df["tmdb_title"].notna() & (df["tmdb_title"].astype(str).str.strip() != ""),
                                   df["ml_title"])
    df["title"] = title
    no_title = title.isna() | (title.astype(str).str.strip() == "")
    reasons[no_title & reasons.isna()] = "missing_title"
    adult = df["adult"].fillna(False).astype(bool)
    reasons[adult & reasons.isna()] = "adult_content"

    quarantine = df.loc[reasons.notna(), ["movieId", "tmdbId", "title"]].copy()
    quarantine["reason"] = reasons[reasons.notna()].values
    df = df[reasons.isna()].copy()

    unified = [unify_genres(m, t) for m, t in zip(df["ml_genres"], df["tmdb_genres"])]
    df["genres"] = [u[0] for u in unified]
    df["primary_genre"] = [u[1] for u in unified]

    df["language"] = df["original_language"].map(normalize_language)
    df["language_group"] = df["language"].map(language_group)

    raw_year = df["tmdb_year"].where(df["tmdb_year"].notna(), df["ml_year"])
    df["year_raw"] = raw_year
    valid_year = _valid_year(raw_year, max_year)
    df["year"] = raw_year.where(valid_year).astype("Int64")
    df["era"] = df["year"].map(era_of)

    df["vote_count"] = df["vote_count"].astype("float64")
    df["popularity_bucket"] = popularity_buckets(df["vote_count"])

    overview = df["overview"].fillna("").astype(str).str.strip()
    runtime = df["runtime"].astype("float64")
    df["flag_missing_overview"] = overview.eq("")
    df["flag_short_overview"] = overview.ne("") & overview.str.len().lt(short_overview_chars)
    df["flag_missing_runtime"] = runtime.isna() | runtime.le(0)
    df["flag_runtime_outlier"] = runtime.gt(runtime_outlier)
    df["flag_invalid_year"] = raw_year.notna() & ~valid_year
    df["flag_missing_year"] = raw_year.isna()
    df["flag_unknown_language"] = df["language"].eq("unknown")
    df["flag_year_mismatch"] = df["year_mismatch"].astype(bool)

    columns = [
        "movieId", "tmdbId", "imdb_id", "title", "ml_title", "original_title", "year", "year_raw", "ml_year",
        "tmdb_year", "release_date", "overview", "runtime", "language", "original_language", "language_group",
        "era", "genres", "primary_genre", "ml_genres", "tmdb_genres", "keywords", "production_countries",
        "spoken_languages", "popularity", "vote_average", "vote_count", "popularity_bucket", "poster_path",
        "trailer_youtube_key",
    ] + [c for c in df.columns if c.startswith("flag_")]
    clean = df[columns].sort_values("movieId").reset_index(drop=True)
    flags = {c: int(clean[c].sum()) for c in clean.columns if c.startswith("flag_")}
    summary = {
        "input_rows": int(len(joined)),
        "clean_rows": int(len(clean)),
        "quarantined": {str(k): int(v) for k, v in quarantine["reason"].value_counts().items()},
        "flags": flags,
        "slices": {c: {str(k): int(v) for k, v in clean[c].value_counts().items()}
                   for c in ("language_group", "era", "primary_genre", "popularity_bucket")},
    }
    return clean, quarantine.reset_index(drop=True), summary


def run_clean(cfg: PipelineConfig) -> dict[str, Any]:
    """Clean ``joined.parquet`` and write the catalog, quarantine and summary."""
    joined = read_parquet(cfg.interim_dir / JOINED_FILE)
    clean, quarantine, summary = clean_catalog(joined, cfg.runtime_outlier_minutes, cfg.short_overview_chars)
    write_parquet(cfg.processed_dir / CLEAN_FILE, clean)
    write_parquet(cfg.processed_dir / CLEAN_QUARANTINE_FILE, quarantine)
    write_json(cfg.processed_dir / CLEAN_SUMMARY_FILE, summary)
    logger.info("Clean: %d rows kept, quarantined=%s, flags=%s",
                summary["clean_rows"], summary["quarantined"], summary["flags"])
    return {k: summary[k] for k in ("input_rows", "clean_rows", "quarantined", "flags")}
