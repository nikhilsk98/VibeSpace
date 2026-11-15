"""Stratified split by movie: determinism, disjointness, ratios, strata, persistence, test guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibespace_model.errors import SplitError, TestSplitAccessError
from vibespace_model.splits import (
    SPLITS,
    assert_no_test_rows,
    load_split,
    make_split,
    merge_small_strata,
    save_split,
    slice_counts,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


@pytest.fixture
def population(rng):
    ids = rng.permutation(np.arange(1000, 3000))[:1500]
    strata = pd.Series(
        rng.choice(["english", "other", "hindi", "regional"], size=1500, p=[0.8, 0.17, 0.025, 0.005])
    )
    return ids, strata


def test_deterministic_and_order_independent(population) -> None:
    ids, strata = population
    a = make_split(ids, strata, RATIOS, seed=42)
    b = make_split(ids, strata, RATIOS, seed=42)
    order = np.random.default_rng(1).permutation(len(ids))
    c = make_split(ids[order], strata.iloc[order].reset_index(drop=True), RATIOS, seed=42)
    assert a.split_hash == b.split_hash == c.split_hash
    assert make_split(ids, strata, RATIOS, seed=43).split_hash != a.split_hash


def test_no_overlap_and_complete(population) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42)
    sets = [set(split.ids(name)) for name in SPLITS]
    assert sum(len(s) for s in sets) == len(ids)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(ids)


def test_ratios_within_tolerance(population) -> None:
    ids, strata = population
    counts = make_split(ids, strata, RATIOS, seed=42).meta["counts"]
    for name, ratio in RATIOS.items():
        assert abs(counts[name] / len(ids) - ratio) < 0.01


def test_every_stratum_in_every_split(population) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42, min_stratum_size=20)
    frame = split.assignment.merge(pd.DataFrame({"movieId": ids, "stratum": strata}), on="movieId")
    for stratum in ("english", "other", "hindi"):
        present = set(frame.loc[frame["stratum"] == stratum, "split"])
        assert present == set(SPLITS)


def test_tiny_strata_are_merged() -> None:
    merged = merge_small_strata(pd.Series(["a"] * 30 + ["b"] * 3 + ["c"] * 2 + [None]), 20)
    assert set(merged) == {"a", "_small"}
    split = make_split(np.arange(36), pd.Series(["a"] * 30 + ["b"] * 3 + ["c"] * 3), RATIOS, seed=1)
    assert split.meta["n_movies"] == 36
    # very small populations never fail
    assert make_split(np.arange(2), pd.Series(["x", "y"]), RATIOS, seed=1).meta["n_movies"] == 2


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(SplitError, match="unique"):
        make_split(np.array([1, 1, 2]), pd.Series(["a", "a", "a"]), RATIOS, seed=1)


def test_saved_assignment_reloads_identically(population, tmp_path) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42)
    save_split(split, tmp_path / "split")
    again = load_split(tmp_path / "split")
    assert again.split_hash == split.split_hash
    pd.testing.assert_frame_equal(again.assignment, split.assignment)
    assert b"\r\n" not in (tmp_path / "split" / "split_assignment.csv").read_bytes()


def test_tampered_or_missing_split(population, tmp_path) -> None:
    ids, strata = population
    save_split(make_split(ids, strata, RATIOS, seed=42), tmp_path / "split")
    csv = tmp_path / "split" / "split_assignment.csv"
    csv.write_text(csv.read_text().replace(",train", ",test", 1))
    with pytest.raises(SplitError, match="modified"):
        load_split(tmp_path / "split")
    with pytest.raises(SplitError, match="no saved split"):
        load_split(tmp_path / "missing")


def test_masks_and_unknown_movies(population) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42)
    masks = split.masks(ids)
    assert sum(int(m.sum()) for m in masks.values()) == len(ids)
    with pytest.raises(SplitError, match="not in the saved split"):
        split.masks(np.array([ids[0], -5]))


def test_test_split_guard(population) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42)
    assert_no_test_rows(np.concatenate([split.ids("train"), split.ids("val")]), split, "tuning")
    with pytest.raises(TestSplitAccessError, match="tuning must not use test movies"):
        assert_no_test_rows(split.ids("test")[:3], split, "tuning")


def test_slice_counts_table(population) -> None:
    ids, strata = population
    split = make_split(ids, strata, RATIOS, seed=42)
    catalog = pd.DataFrame({"movieId": ids, "language_group": strata})
    table = slice_counts(catalog, split.masks(ids), ["language_group"])
    assert set(table.columns) == {"slice_column", "slice", "train", "val", "test"}
    assert int(table[list(SPLITS)].to_numpy().sum()) == len(ids)
