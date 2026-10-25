"""Flatten cached TMDB JSON responses into one table.

Only the movies selected for the current run are read. Unreadable or invalid
cache files are moved to a quarantine folder (so the next fetch requests them
again) and listed in a quarantine table with the reason.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.acquire_tmdb import NOT_FOUND_MARKER, SELECTION_FILE, cache_path
from scripts.config import PipelineConfig
from scripts.io_utils import ensure_dir, read_parquet, write_json, write_parquet
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

TMDB_TABLE_FILE = "tmdb_movies.parquet"
TMDB_QUARANTINE_FILE = "tmdb_quarantine.parquet"
TMDB_QUARANTINE_DIR = "tmdb_quarantine_files"
TMDB_SUMMARY_FILE = "build_tmdb_table_summary.json"

TMDB_COLUMNS = [
    "tmdb_id", "imdb_id", "tmdb_title", "original_title", "overview", "runtime", "original_language",
    "spoken_languages", "tmdb_genres", "keywords", "production_countries", "release_date", "popularity",
    "vote_average", "vote_count", "poster_path", "adult", "trailer_youtube_key",
]


def _names(items: Any, key: str = "name") -> list[str]:
    """Collect ``key`` from a list of dicts, skipping blanks."""
    if not isinstance(items, list):
        return []
    return [str(i[key]).strip() for i in items if isinstance(i, dict) and i.get(key)]


def best_trailer_key(videos: Any) -> str | None:
    """Choose the best YouTube trailer: official trailers first, then newest, then largest."""
    results = videos.get("results", []) if isinstance(videos, dict) else []
    candidates = [v for v in results if isinstance(v, dict) and v.get("site") == "YouTube" and v.get("key")]
    if not candidates:
        return None

    def rank(video: dict[str, Any]) -> tuple[int, int, str, int]:
        return (
            1 if video.get("type") == "Trailer" else 0,
            1 if video.get("official") else 0,
            str(video.get("published_at") or ""),
            int(video.get("size") or 0),
        )

    return str(max(candidates, key=rank)["key"])


def flatten_movie(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn one TMDB movie JSON into a flat record (see ``TMDB_COLUMNS``)."""
    keywords = payload.get("keywords") or {}
    kw_list = keywords.get("keywords", keywords.get("results", [])) if isinstance(keywords, dict) else []
    runtime = payload.get("runtime")
    return {
        "tmdb_id": int(payload["id"]),
        "imdb_id": payload.get("imdb_id") or None,
        "tmdb_title": (payload.get("title") or "").strip() or None,
        "original_title": (payload.get("original_title") or "").strip() or None,
        "overview": (payload.get("overview") or "").strip() or None,
        "runtime": float(runtime) if isinstance(runtime, int | float) else None,
        "original_language": (payload.get("original_language") or "").strip() or None,
        "spoken_languages": _names(payload.get("spoken_languages"), "iso_639_1"),
        "tmdb_genres": _names(payload.get("genres")),
        "keywords": _names(kw_list),
        "production_countries": _names(payload.get("production_countries"), "iso_3166_1"),
        "release_date": (payload.get("release_date") or "").strip() or None,
        "popularity": _to_float(payload.get("popularity")),
        "vote_average": _to_float(payload.get("vote_average")),
        "vote_count": _to_float(payload.get("vote_count")),
        "poster_path": payload.get("poster_path") or None,
        "adult": bool(payload.get("adult", False)),
        "trailer_youtube_key": best_trailer_key(payload.get("videos")),
    }


def _to_float(value: Any) -> float | None:
    """Convert numbers to float; anything else becomes ``None``."""
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _quarantine_file(path: Path, quarantine_dir: Path) -> None:
    """Move an unusable cache file out of the cache so it is fetched again later."""
    ensure_dir(quarantine_dir)
    shutil.move(str(path), str(quarantine_dir / path.name))


def build_tmdb_table(cache_dir: Path, tmdb_ids: list[int], quarantine_dir: Path | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Read cached JSON for ``tmdb_ids`` and return (table, quarantine, counts)."""
    rows: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    counts = {"requested": 0, "ok": 0, "not_found": 0, "missing_cache": 0, "quarantined": 0}
    for tmdb_id in sorted({int(i) for i in tmdb_ids}):
        counts["requested"] += 1
        path = cache_path(cache_dir, tmdb_id)
        if not path.is_file():
            counts["missing_cache"] += 1
            continue
        reason = None
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            payload, reason = None, f"unreadable_json: {type(exc).__name__}"
        if payload is not None:
            if isinstance(payload, dict) and payload.get(NOT_FOUND_MARKER):
                counts["not_found"] += 1
                continue
            if not isinstance(payload, dict) or "id" not in payload:
                reason = "invalid_payload: no id"
            elif int(payload["id"]) != tmdb_id:
                reason = f"invalid_payload: id {payload['id']} != {tmdb_id}"
            else:
                try:
                    rows.append(flatten_movie(payload))
                    counts["ok"] += 1
                    continue
                except (KeyError, TypeError, ValueError) as exc:
                    reason = f"invalid_payload: {type(exc).__name__}: {exc}"
        quarantine.append({"tmdb_id": tmdb_id, "file": path.name, "reason": reason})
        counts["quarantined"] += 1
        logger.warning("Quarantined TMDB cache file %s (%s)", path.name, reason)
        if quarantine_dir is not None:
            _quarantine_file(path, quarantine_dir)

    table = pd.DataFrame(rows, columns=TMDB_COLUMNS)
    table["tmdb_id"] = table["tmdb_id"].astype("int64")
    quarantine_df = pd.DataFrame(quarantine, columns=["tmdb_id", "file", "reason"])
    return table, quarantine_df, counts


def run_build_tmdb_table(cfg: PipelineConfig) -> dict[str, Any]:
    """Build ``tmdb_movies.parquet`` for the movies selected in the fetch step."""
    selection = read_parquet(cfg.interim_dir / SELECTION_FILE)
    tmdb_ids = selection["tmdbId"].dropna().astype(int).tolist()
    quarantine_dir = ensure_dir(cfg.interim_dir / TMDB_QUARANTINE_DIR)  # always present for DVC
    table, quarantine, counts = build_tmdb_table(cfg.tmdb_cache_dir, tmdb_ids, quarantine_dir=quarantine_dir)
    write_parquet(cfg.interim_dir / TMDB_TABLE_FILE, table)
    write_parquet(cfg.interim_dir / TMDB_QUARANTINE_FILE, quarantine)
    write_json(cfg.interim_dir / TMDB_SUMMARY_FILE, counts)
    logger.info("TMDB table: %s", counts)
    return counts
