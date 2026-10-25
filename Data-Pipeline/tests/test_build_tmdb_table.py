"""Tests for scripts.build_tmdb_table."""

from __future__ import annotations

import json

import pandas as pd

from scripts import build_tmdb_table as bt
from scripts.acquire_tmdb import SELECTION_FILE
from scripts.io_utils import write_parquet
from tests.helpers import MISSING_TMDB_ID, TMDB_FIXTURES


def fixture_payload(tmdb_id: int) -> dict:
    return json.loads((TMDB_FIXTURES / f"{tmdb_id}.json").read_text())


def test_flatten_movie_fields():
    row = bt.flatten_movie(fixture_payload(862))
    assert row["tmdb_id"] == 862
    assert row["tmdb_title"] == "Toy Story"
    assert row["tmdb_genres"] == ["Animation", "Adventure", "Family", "Comedy"]
    assert row["keywords"] == ["fixture"]
    assert row["production_countries"] == ["US"]
    assert row["trailer_youtube_key"] == "trailer862"  # official trailer beats a newer teaser
    assert row["adult"] is False
    assert set(row) == set(bt.TMDB_COLUMNS)


def test_flatten_handles_missing_fields():
    row = bt.flatten_movie({"id": 5, "runtime": None, "overview": "  ", "vote_count": "n/a"})
    assert row["overview"] is None and row["runtime"] is None and row["vote_count"] is None
    assert row["tmdb_genres"] == [] and row["trailer_youtube_key"] is None


def test_best_trailer_key_rules():
    videos = {"results": [
        {"site": "Vimeo", "type": "Trailer", "key": "vimeo"},
        {"site": "YouTube", "type": "Clip", "key": "clip", "official": True},
        {"site": "YouTube", "type": "Trailer", "key": "fan", "official": False, "published_at": "2024"},
        {"site": "YouTube", "type": "Trailer", "key": "old", "official": True, "published_at": "2001"},
        {"site": "YouTube", "type": "Trailer", "key": "new", "official": True, "published_at": "2010"},
    ]}
    assert bt.best_trailer_key(videos) == "new"
    assert bt.best_trailer_key({"results": []}) is None
    assert bt.best_trailer_key(None) is None


def test_keywords_results_shape():
    row = bt.flatten_movie({"id": 1, "keywords": {"results": [{"name": "space"}]}})
    assert row["keywords"] == ["space"]


def test_build_table_counts_and_quarantine(cached_cfg):
    cache = cached_cfg.tmdb_cache_dir
    (cache / "777.json").write_text("{not json")
    (cache / "778.json").write_text(json.dumps({"title": "no id"}))
    (cache / "779.json").write_text(json.dumps({"id": 1234}))
    qdir = cached_cfg.interim_dir / "q"
    table, quarantine, counts = bt.build_tmdb_table(cache, [862, 949, MISSING_TMDB_ID, 777, 778, 779, 4242], qdir)
    assert counts == {"requested": 7, "ok": 2, "not_found": 1, "missing_cache": 1, "quarantined": 3}
    assert sorted(table["tmdb_id"]) == [862, 949]
    assert set(quarantine["tmdb_id"]) == {777, 778, 779}
    assert quarantine.loc[quarantine["tmdb_id"] == 777, "reason"].iloc[0].startswith("unreadable_json")
    assert not (cache / "777.json").exists() and (qdir / "777.json").exists()


def test_build_table_empty():
    table, quarantine, counts = bt.build_tmdb_table("/nonexistent", [])
    assert table.empty and list(table.columns) == bt.TMDB_COLUMNS and counts["requested"] == 0


def test_run_build_tmdb_table(cached_cfg):
    selection = pd.DataFrame({"movieId": [1, 2, 8], "tmdbId": pd.array([862, 8844, None], dtype="Int64")})
    write_parquet(cached_cfg.interim_dir / SELECTION_FILE, selection)
    counts = bt.run_build_tmdb_table(cached_cfg)
    assert counts["ok"] == 2
    assert len(pd.read_parquet(cached_cfg.interim_dir / bt.TMDB_TABLE_FILE)) == 2
