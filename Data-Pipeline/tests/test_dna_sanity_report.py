"""Tests for scripts.dna_sanity_report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import dna_sanity_report as sr
from scripts.build_dna_labels import DNA_DIMENSIONS, DnaMapping


def make_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    genres = rng.choice(["Comedy", "Drama", "Horror", "Romance", "Action"], size=n)
    labels = pd.DataFrame({"movieId": np.arange(n)})
    for dim in DNA_DIMENSIONS:
        labels[dim] = rng.uniform(size=n)
        labels[f"{dim}_raw"] = rng.uniform(size=n)
    movies = pd.DataFrame({"movieId": np.arange(n), "title": [f"T{i}" for i in range(n)], "genres": genres})
    return labels, movies


def mapping():
    return DnaMapping("v1", "pending team review", {d: ["a", "b", "c"] for d in DNA_DIMENSIONS},
                      {d: [] for d in DNA_DIMENSIONS})


def test_genre_lift_detects_signal():
    labels, movies = make_data()
    labels["humor"] = (movies["genres"] == "Comedy").astype(float) + 0.01 * labels["humor"]
    assert sr.genre_lift(labels, movies, "humor") > 2
    assert np.isnan(sr.genre_lift(labels, movies.assign(genres="Western"), "humor"))


def test_flags_weak_flat_and_correlated_dimensions():
    labels, movies = make_data()
    labels["humor_raw"] = 0.5          # flat raw scores
    labels["romance"] = labels["darkness"]   # redundant pair
    flags = sr.find_flags(labels, movies, DNA_DIMENSIONS)
    assert any("barely vary" in r for r in flags["humor"])
    assert any("correlated" in r for r in flags["darkness"])
    assert any("weak genre" in r for r in flags["humor"])


def test_report_lists_top_and_bottom(tmp_path):
    labels, movies = make_data(50)
    flagged = sr.write_sanity_report(labels, movies, mapping(), tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    top = movies.set_index("movieId").loc[labels.sort_values("energy").movieId.iloc[-1], "title"]
    assert "**v1**" in text and f"{top}" in text and "Top 10" in text and "Bottom 10" in text
    assert isinstance(flagged, list)
