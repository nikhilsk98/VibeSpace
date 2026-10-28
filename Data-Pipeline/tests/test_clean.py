"""Tests for scripts.clean."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import clean as cl
from scripts.io_utils import write_parquet
from scripts.join_ids import JOINED_FILE


def joined_row(movie_id: int, **kw) -> dict:
    row = {
        "movieId": movie_id, "ml_title_raw": f"M{movie_id} (2000)", "ml_title": f"M{movie_id}", "ml_year": 2000.0,
        "ml_genres": "Comedy", "tmdbId": 1000 + movie_id, "imdb_id": None, "tmdb_title": f"Movie {movie_id}",
        "original_title": None, "overview": "A long enough overview about a person who does many things.",
        "runtime": 100.0, "original_language": "en", "spoken_languages": [], "tmdb_genres": ["Comedy"],
        "keywords": [], "production_countries": [], "release_date": "2000-01-01", "popularity": 1.0,
        "vote_average": 6.0, "vote_count": float(movie_id), "poster_path": None, "adult": False,
        "trailer_youtube_key": None, "tmdb_year": 2000.0, "year_mismatch": False,
    }
    row.update(kw)
    return row


@pytest.mark.parametrize("code,expected", [
    ("en", "en"), (" FR ", "fr"), ("pt-BR", "pt"), ("cn", "zh"), ("xx", "unknown"), ("", "unknown"),
    (None, "unknown"), ("eng", "unknown"), ("1a", "unknown"), (5, "unknown"),
])
def test_normalize_language(code, expected):
    assert cl.normalize_language(code) == expected


@pytest.mark.parametrize("code,group", [
    ("en", "english"), ("hi", "hindi"), ("ta", "indian_regional"), ("te", "indian_regional"),
    ("as", "indian_regional"), ("fr", "other_international"), ("unknown", "unknown"),
])
def test_language_group(code, group):
    assert cl.language_group(code) == group


@pytest.mark.parametrize("year,era", [(1979, "pre_1980"), (1980, "1980_1999"), (1999, "1980_1999"),
                                      (2000, "2000_plus"), (None, "unknown"), (np.nan, "unknown"),
                                      (pd.NA, "unknown")])
def test_era(year, era):
    assert cl.era_of(year) == era


def test_genre_unification():
    genres, primary = cl.unify_genres("Sci-Fi|Children|IMAX", np.array(["Science Fiction", "Family"]))
    assert genres == ["family", "science_fiction"] and primary == "science_fiction"
    assert cl.unify_genres("(no genres listed)", []) == ([], "unknown")
    assert cl.unify_genres("Film-Noir", None) == (["film_noir"], "film_noir")
    assert cl.unify_genres(None, ["TV Movie", "Made Up Genre"]) == (["tv_movie"], "tv_movie")
    assert cl.map_genre(None) is None


def test_popularity_buckets():
    votes = pd.Series([float(v) for v in range(1, 11)])
    buckets = cl.popularity_buckets(votes).tolist()
    assert buckets.count("high") == 2 and buckets.count("low") == 4 and buckets.count("mid") == 4
    assert cl.popularity_buckets(pd.Series([], dtype=float)).empty
    with_missing = pd.Series([np.nan] + [float(v) for v in range(1, 10)])
    assert cl.popularity_buckets(with_missing)[0] == "low"  # missing vote count ranks as 0 votes


def test_clean_flags_and_quarantine():
    rows = [
        joined_row(1),
        joined_row(1),                                           # duplicate movieId
        joined_row(2, tmdb_title=None, ml_title=None),           # no title at all
        joined_row(3, adult=True),                               # adult
        joined_row(4, overview=None, runtime=0.0),               # missing plot, runtime 0
        joined_row(5, overview="Too short.", runtime=450.0),     # short plot, outlier runtime
        joined_row(6, original_language="xx", tmdb_year=1700.0, ml_year=1700.0),  # unknown lang, bad year
        joined_row(7, tmdb_title="", ml_title="Fallback Title", tmdb_year=np.nan, ml_year=np.nan,
                   year_mismatch=True),
        joined_row(8, original_language="hi", tmdb_genres=[], ml_genres="(no genres listed)"),
    ]
    clean, quarantine, summary = cl.clean_catalog(pd.DataFrame(rows), runtime_outlier=400, max_year=2026)
    assert dict(zip(quarantine["movieId"], quarantine["reason"])) == {1: "duplicate_movie_id", 2: "missing_title",
                                                                      3: "adult_content"}
    c = clean.set_index("movieId")
    assert c.loc[4, "flag_missing_overview"] and c.loc[4, "flag_missing_runtime"]
    assert c.loc[5, "flag_short_overview"] and c.loc[5, "flag_runtime_outlier"]
    assert c.loc[6, "flag_unknown_language"] and c.loc[6, "flag_invalid_year"]
    assert pd.isna(c.loc[6, "year"]) and c.loc[6, "year_raw"] == 1700 and c.loc[6, "era"] == "unknown"
    assert c.loc[7, "title"] == "Fallback Title" and c.loc[7, "flag_missing_year"] and c.loc[7, "flag_year_mismatch"]
    assert c.loc[8, "language_group"] == "hindi" and c.loc[8, "primary_genre"] == "unknown"
    assert c.loc[1, "runtime"] == 100.0  # flagged values are kept, never silently fixed
    assert summary["clean_rows"] == 6 and summary["quarantined"]["adult_content"] == 1
    assert summary["flags"]["flag_runtime_outlier"] == 1


def test_clean_empty_input():
    clean, quarantine, summary = cl.clean_catalog(pd.DataFrame([joined_row(1)]).iloc[0:0])
    assert clean.empty and quarantine.empty and summary["clean_rows"] == 0


def test_run_clean(cfg):
    write_parquet(cfg.interim_dir / JOINED_FILE, pd.DataFrame([joined_row(i) for i in range(1, 6)]))
    out = cl.run_clean(cfg)
    assert out["clean_rows"] == 5
    assert (cfg.processed_dir / cl.CLEAN_FILE).is_file()
