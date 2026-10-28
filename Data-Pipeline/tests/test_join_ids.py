"""Tests for scripts.join_ids."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts import join_ids as ji
from scripts.acquire_tmdb import SELECTION_FILE
from scripts.build_tmdb_table import TMDB_COLUMNS, TMDB_TABLE_FILE
from scripts.exceptions import MatchRateError
from scripts.io_utils import write_parquet


def tmdb_rows(ids_years: dict[int, str]) -> pd.DataFrame:
    rows = []
    for tmdb_id, date in ids_years.items():
        row = {c: None for c in TMDB_COLUMNS}
        row.update({"tmdb_id": tmdb_id, "tmdb_title": f"T{tmdb_id}", "release_date": date})
        rows.append(row)
    return pd.DataFrame(rows, columns=TMDB_COLUMNS)


@pytest.fixture
def frames():
    movies = pd.DataFrame({
        "movieId": [1, 2, 3, 4, 5, 6],
        "title": ["A (1995)", "B (1999)", "C (2000)", "D (2001)", "E (2002)", "F"],
        "genres": ["Comedy"] * 6,
    })
    links = pd.DataFrame({"movieId": [1, 2, 3, 4, 5, 6], "imdbId": range(6),
                          "tmdbId": [10, None, 10, 40, 50, 60]})
    tmdb = tmdb_rows({10: "1995-05-01", 50: "2010-01-01", 60: None})
    return movies, links, tmdb


@pytest.mark.parametrize("title,expected", [
    ("Heat (1995)", ("Heat", 1995.0)),
    ("Les Misérables (2012) ", ("Les Misérables", 2012.0)),
    ("Babylon 5", ("Babylon 5", math.nan)),
    ("Fawlty Towers (1975-1979)", ("Fawlty Towers", 1975.0)),
    ("(1995)", (None, 1995.0)),
    ("", (None, math.nan)),
    (None, (None, math.nan)),
])
def test_parse_title_year(title, expected):
    name, year = ji.parse_title_year(title)
    assert name == expected[0]
    assert (math.isnan(year) and math.isnan(expected[1])) or year == expected[1]


def test_quarantine_reasons(frames):
    movies, links, tmdb = frames
    joined, quarantine, stats = ji.join_movielens_tmdb(movies, links, tmdb)
    reasons = dict(zip(quarantine["movieId"], quarantine["reason"]))
    assert reasons == {2: "no_tmdb_id", 3: "duplicate_tmdb_id", 4: "tmdb_missing"}
    assert sorted(joined["movieId"]) == [1, 5, 6]
    assert stats.matched == 3 and stats.total_movies == 6 and stats.match_rate == 0.5


def test_duplicate_keeps_lowest_movie_id(frames):
    movies, links, tmdb = frames
    shuffled = links.iloc[::-1].reset_index(drop=True)
    joined, quarantine, _ = ji.join_movielens_tmdb(movies, shuffled, tmdb)
    assert 1 in set(joined["movieId"]) and 3 in set(quarantine["movieId"])


def test_year_mismatch_flag(frames):
    movies, links, tmdb = frames
    joined, _, stats = ji.join_movielens_tmdb(movies, links, tmdb, year_tolerance=2)
    flags = dict(zip(joined["movieId"], joined["year_mismatch"]))
    assert flags == {1: False, 5: True, 6: False}  # 6 has no years at all -> not flagged
    assert stats.year_mismatches == 1


def test_movie_id_filter(frames):
    movies, links, tmdb = frames
    joined, quarantine, stats = ji.join_movielens_tmdb(movies, links, tmdb, movie_ids=[1, 2])
    assert stats.total_movies == 2 and sorted(joined["movieId"]) == [1]


def test_empty_inputs(frames):
    movies, links, tmdb = frames
    joined, quarantine, stats = ji.join_movielens_tmdb(movies, links, tmdb.iloc[0:0], movie_ids=[])
    assert stats.total_movies == 0 and joined.empty
    with pytest.raises(MatchRateError, match="empty"):
        ji.check_match_rate(stats, 0.9)


def test_match_rate_threshold(frames):
    movies, links, tmdb = frames
    _, _, stats = ji.join_movielens_tmdb(movies, links, tmdb)
    with pytest.raises(MatchRateError, match="50.00%"):
        ji.check_match_rate(stats, 0.9)
    ji.check_match_rate(stats, 0.5)


def test_run_join_ids(ml_cfg):
    selection = pd.DataFrame({"movieId": [1, 2, 8, 9, 10], "tmdbId": pd.array([862, 8844, None, 862, 999999],
                                                                                 dtype="Int64")})
    write_parquet(ml_cfg.interim_dir / SELECTION_FILE, selection)
    write_parquet(ml_cfg.interim_dir / TMDB_TABLE_FILE, tmdb_rows({862: "1995-10-30", 8844: "1995-12-15"}))
    with pytest.raises(MatchRateError):
        ji.run_join_ids(ml_cfg)  # 2 of 5 matched < 70%
    assert (ml_cfg.interim_dir / ji.JOIN_STATS_FILE).is_file()
    quarantine = pd.read_parquet(ml_cfg.interim_dir / ji.JOIN_QUARANTINE_FILE)
    assert set(quarantine["reason"]) == {"no_tmdb_id", "duplicate_tmdb_id", "tmdb_missing"}
