"""Tests for scripts.generate_stats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import generate_stats as gs
from scripts.build_dna_labels import DNA_DIMENSIONS, DNA_LABELS_FILE
from scripts.clean import CLEAN_FILE
from scripts.encode_plots import EMBEDDINGS_INDEX_FILE
from scripts.io_utils import read_json, write_parquet


def catalog(langs):
    n = len(langs)
    return pd.DataFrame({
        "movieId": range(n), "language_group": langs, "era": ["2000_plus"] * n, "primary_genre": ["drama"] * n,
        "popularity_bucket": ["mid"] * n, "runtime": np.linspace(80, 120, n), "overview": ["x"] * (n - 1) + [None],
        "genres": [["drama"]] * n, "flag": [True, False] * (n // 2) + [False] * (n % 2),
    })


def test_column_stats_kinds():
    assert gs.column_stats(pd.Series([1.0, 2.0, None]))["kind"] == "numeric"
    num = gs.column_stats(pd.Series([1.0, 2.0, None]))
    assert num["null_count"] == 1 and num["max"] == 2.0
    assert gs.column_stats(pd.Series(["a", "a", "b"]))["top_values"] == {"a": 2, "b": 1}
    assert gs.column_stats(pd.Series([True, False]))["true_count"] == 1
    assert gs.column_stats(pd.Series([["a"], []]))["empty_count"] == 1
    assert gs.column_stats(pd.Series([], dtype=float))["null_rate"] == 0.0
    assert "min" not in gs.column_stats(pd.Series([None, None], dtype=float))


def test_psi_properties():
    a = {"x": 0.5, "y": 0.5}
    assert gs.psi(a, a) == 0
    assert gs.psi(a, {"x": 0.9, "y": 0.1}) > 0.25
    assert gs.psi({"x": 1.0}, {"y": 1.0}) > 1  # disjoint categories handled via epsilon


def test_compare_runs_detects_changes():
    prev = {"run_id": "r1", "datasets": {"catalog": gs.dataset_stats(catalog(["english"] * 10))},
            "slice_distributions": {"language_group": {"english": 1.0}}}
    cur_df = catalog(["english"] * 5 + ["hindi"] * 3).drop(columns=["flag"]).assign(extra=1)
    cur_df.loc[0:3, "overview"] = None
    cur = {"datasets": {"catalog": gs.dataset_stats(cur_df), "new": gs.dataset_stats(cur_df)},
           "slice_distributions": {"language_group": gs.distribution(cur_df["language_group"])}}
    drift = gs.compare_runs(prev, cur)
    d = drift["datasets"]["catalog"]
    assert drift["previous_run_id"] == "r1"
    assert d["added_columns"] == ["extra"] and d["removed_columns"] == ["flag"]
    assert d["null_rate_changes"]["overview"] > 0.3
    assert d["row_change_pct"] == pytest.approx(-0.2)
    assert drift["psi"]["language_group"] > 0.25
    assert "note" in drift["datasets"]["new"]
    assert gs.compare_runs(None, cur)["previous_run_id"] is None


def test_run_profile():
    assert gs.run_profile("genome", None) == "genome_all"
    assert gs.run_profile("rest", 500) == "rest_n500"


def write_outputs(cfg, langs):
    cat = catalog(langs)
    write_parquet(cfg.processed_dir / CLEAN_FILE, cat)
    labels = pd.DataFrame({"movieId": cat["movieId"][:-1]})
    for dim in DNA_DIMENSIONS:
        labels[dim] = 0.5
    write_parquet(cfg.processed_dir / DNA_LABELS_FILE, labels)
    write_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE, pd.DataFrame({"row": [0], "movieId": [0]}))


def test_run_generate_stats_history(cfg):
    write_outputs(cfg, ["english"] * 10)
    first = gs.run_generate_stats(cfg, run_id="manual__2024-01-01T00:00:00+00:00")
    assert first["previous_run_id"] is None and first["run_id"] == "manual__2024-01-01T00_00_00_00_00"
    assert first["coverage"]["label_coverage"] == pytest.approx(0.9)
    write_outputs(cfg, ["english"] * 5 + ["hindi"] * 5)
    second = gs.run_generate_stats(cfg, run_id="r2")
    assert second["previous_run_id"] == first["run_id"] and second["psi"]["language_group"] > 0.25
    history = list((cfg.metrics_dir / "stats" / "genome_all").glob("stats_*.json"))
    assert len(history) == 2
    assert read_json(cfg.metrics_dir / "stats_current.json")["run_id"] == "r2"


def test_run_generate_stats_default_run_id(cfg):
    write_outputs(cfg, ["english"] * 4)
    assert gs.run_generate_stats(cfg)["run_id"]
