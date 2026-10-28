"""Join MovieLens movies to TMDB records through ``links.csv``.

Every movie that cannot be matched goes to a quarantine table with a reason:

* ``no_tmdb_id``        - ``links.csv`` has no TMDB id for the movie
* ``duplicate_tmdb_id`` - several movies share one TMDB id (lowest movieId kept)
* ``tmdb_missing``      - TMDB has no usable record (404, failed fetch, quarantined)

Release-year disagreements larger than the tolerance are flagged, not dropped.
The step fails when the match rate is below the configured minimum.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from scripts.acquire_tmdb import SELECTION_FILE
from scripts.build_tmdb_table import TMDB_TABLE_FILE
from scripts.config import PipelineConfig
from scripts.exceptions import MatchRateError
from scripts.io_utils import read_csv_checked, read_parquet, write_json, write_parquet
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

JOINED_FILE = "joined.parquet"
JOIN_QUARANTINE_FILE = "join_quarantine.parquet"
JOIN_STATS_FILE = "join_stats.json"
_YEAR_RE = re.compile(r"\((\d{4})(?:[-–]\d{0,4})?\)\s*$")


def parse_title_year(title: Any) -> tuple[str | None, float]:
    """Split ``"Heat (1995)"`` into ``("Heat", 1995.0)``; the year is NaN if absent."""
    if not isinstance(title, str) or not title.strip():
        return None, float("nan")
    text = title.strip()
    match = _YEAR_RE.search(text)
    if not match:
        return text, float("nan")
    clean = text[: match.start()].strip() or None
    return clean, float(match.group(1))


@dataclass
class JoinStats:
    """Summary numbers for the join."""

    total_movies: int
    matched: int
    match_rate: float
    no_tmdb_id: int
    duplicate_tmdb_id: int
    tmdb_missing: int
    year_mismatches: int

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output."""
        return asdict(self)


def join_movielens_tmdb(movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame,
                        movie_ids: list[int] | None = None, year_tolerance: int = 2
                        ) -> tuple[pd.DataFrame, pd.DataFrame, JoinStats]:
    """Return (joined, quarantine, stats) for the given MovieLens ``movie_ids``."""
    ml = movies[["movieId", "title", "genres"]].copy()
    if movie_ids is not None:
        ml = ml[ml["movieId"].isin(set(movie_ids))]
    parsed = ml["title"].map(parse_title_year)
    ml["ml_title"] = parsed.map(lambda t: t[0])
    ml["ml_year"] = parsed.map(lambda t: t[1]).astype("float64")
    ml = ml.rename(columns={"genres": "ml_genres", "title": "ml_title_raw"})

    link = links[["movieId", "tmdbId"]].copy()
    link["tmdbId"] = pd.to_numeric(link["tmdbId"], errors="coerce").astype("Int64")
    df = ml.merge(link, on="movieId", how="left").sort_values("movieId").reset_index(drop=True)

    reasons = pd.Series(pd.NA, index=df.index, dtype="object")
    no_id = df["tmdbId"].isna()
    reasons[no_id] = "no_tmdb_id"

    has_id = ~no_id
    # ``df`` is sorted by movieId, so ``keep="first"`` keeps the lowest movieId.
    dup = has_id & df["tmdbId"].duplicated(keep="first")
    reasons[dup] = "duplicate_tmdb_id"

    tmdb_ids = set(tmdb["tmdb_id"].astype(int))
    missing = has_id & reasons.isna() & ~df["tmdbId"].fillna(-1).astype(int).isin(tmdb_ids)
    reasons[missing] = "tmdb_missing"

    quarantine = df[reasons.notna()][["movieId", "ml_title_raw", "tmdbId"]].copy()
    quarantine["reason"] = reasons[reasons.notna()].values
    keep = df[reasons.isna()].copy()
    keep["tmdbId"] = keep["tmdbId"].astype("int64")

    joined = keep.merge(tmdb, left_on="tmdbId", right_on="tmdb_id", how="inner").drop(columns=["tmdb_id"])
    tmdb_year = pd.to_datetime(joined["release_date"], errors="coerce").dt.year.astype("float64")
    joined["tmdb_year"] = tmdb_year
    diff = (joined["tmdb_year"] - joined["ml_year"]).abs()
    joined["year_mismatch"] = diff.gt(year_tolerance).fillna(False).astype(bool)

    total = int(len(df))
    stats = JoinStats(
        total_movies=total,
        matched=int(len(joined)),
        match_rate=round(len(joined) / total, 6) if total else 0.0,
        no_tmdb_id=int((quarantine["reason"] == "no_tmdb_id").sum()),
        duplicate_tmdb_id=int((quarantine["reason"] == "duplicate_tmdb_id").sum()),
        tmdb_missing=int((quarantine["reason"] == "tmdb_missing").sum()),
        year_mismatches=int(joined["year_mismatch"].sum()),
    )
    return joined.reset_index(drop=True), quarantine.reset_index(drop=True), stats


def check_match_rate(stats: JoinStats, min_rate: float) -> None:
    """Raise :class:`MatchRateError` when the match rate is below ``min_rate``."""
    if stats.total_movies == 0:
        raise MatchRateError("No movies to join; the selection is empty")
    if stats.match_rate < min_rate:
        raise MatchRateError(
            f"Match rate {stats.match_rate:.2%} is below the minimum {min_rate:.0%} "
            f"(no_tmdb_id={stats.no_tmdb_id}, duplicate_tmdb_id={stats.duplicate_tmdb_id}, "
            f"tmdb_missing={stats.tmdb_missing})"
        )


def run_join_ids(cfg: PipelineConfig) -> dict[str, Any]:
    """Join the selected MovieLens movies with the TMDB table and write the outputs."""
    csv_dir = cfg.movielens_csv_dir
    movies = read_csv_checked(csv_dir / "movies.csv", ["movieId", "title", "genres"])
    links = read_csv_checked(csv_dir / "links.csv", ["movieId", "tmdbId"])
    selection = read_parquet(cfg.interim_dir / SELECTION_FILE)
    tmdb = read_parquet(cfg.interim_dir / TMDB_TABLE_FILE)
    joined, quarantine, stats = join_movielens_tmdb(
        movies, links, tmdb, selection["movieId"].tolist(), cfg.year_mismatch_tolerance
    )
    write_parquet(cfg.interim_dir / JOINED_FILE, joined)
    write_parquet(cfg.interim_dir / JOIN_QUARANTINE_FILE, quarantine)
    result = stats.to_dict() | {"min_match_rate": cfg.min_match_rate}
    write_json(cfg.interim_dir / JOIN_STATS_FILE, result)
    logger.info("Join: %s", result)
    check_match_rate(stats, cfg.min_match_rate)
    return result
