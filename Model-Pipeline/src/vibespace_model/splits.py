"""Deterministic 70/15/15 split by movie, stratified on ``language_group``.

Strata smaller than ``min_stratum_size`` are merged into one ``_small`` stratum, so the split never
fails on tiny groups (e.g. the 9 Indian regional films). The movieId -> split assignment is saved with a
SHA-256 hash; Parts 2 and 3 load and reuse exactly this assignment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vibespace_model.errors import SplitError, TestSplitAccessError

SPLITS: tuple[str, ...] = ("train", "val", "test")
ASSIGNMENT_FILE = "split_assignment.csv"
META_FILE = "split_meta.json"


@dataclass
class Split:
    """movieId -> split assignment plus its fingerprint."""

    assignment: pd.DataFrame  # columns: movieId, split
    split_hash: str
    meta: dict[str, Any]

    def ids(self, name: str) -> np.ndarray:
        """movieIds of one split."""
        return self.assignment.loc[self.assignment["split"] == name, "movieId"].to_numpy()

    def masks(self, movie_ids: np.ndarray) -> dict[str, np.ndarray]:
        """Boolean masks over ``movie_ids`` for train / val / test; every movie must be assigned."""
        lookup = dict(zip(self.assignment["movieId"], self.assignment["split"], strict=True))
        missing = [m for m in movie_ids if m not in lookup]
        if missing:
            raise SplitError(f"{len(missing)} movies are not in the saved split (e.g. {missing[:5]})")
        labels = np.array([lookup[m] for m in movie_ids])
        return {name: labels == name for name in SPLITS}


def merge_small_strata(strata: pd.Series, min_size: int) -> pd.Series:
    """Replace strata with fewer than ``min_size`` members by a single ``_small`` stratum."""
    values = strata.fillna("unknown").astype(str)
    counts = values.value_counts()
    small = counts[counts < min_size].index
    return values.where(~values.isin(small), "_small")


def assignment_hash(assignment: pd.DataFrame) -> str:
    """SHA-256 of the sorted ``movieId,split`` lines (order independent)."""
    ordered = assignment.sort_values("movieId")
    text = "\n".join(f"{m},{s}" for m, s in zip(ordered["movieId"], ordered["split"], strict=True))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_split(
    movie_ids: np.ndarray,
    strata: pd.Series,
    ratios: dict[str, float],
    seed: int,
    min_stratum_size: int = 20,
) -> Split:
    """Shuffle each stratum with a fixed seed and cut it at the train / val ratios."""
    frame = pd.DataFrame(
        {
            "movieId": np.asarray(movie_ids),
            "stratum": merge_small_strata(
                pd.Series(np.asarray(strata), dtype="object"), min_stratum_size
            ).to_numpy(),
        }
    )
    if frame["movieId"].duplicated().any():
        raise SplitError("movie ids must be unique to split by movie")
    frame = frame.sort_values("movieId").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for stratum in sorted(frame["stratum"].unique()):
        members = frame.loc[frame["stratum"] == stratum, "movieId"].to_numpy()
        members = members[rng.permutation(len(members))]
        n_train = int(round(len(members) * ratios["train"]))
        n_val = int(round(len(members) * ratios["val"]))
        if len(members) >= 3:  # every split gets at least one movie of the stratum
            n_train = min(max(n_train, 1), len(members) - 2)
            n_val = min(max(n_val, 1), len(members) - n_train - 1)
        labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (len(members) - n_train - n_val)
        parts.append(pd.DataFrame({"movieId": members, "split": labels}))
    assignment = pd.concat(parts).sort_values("movieId").reset_index(drop=True)
    meta = {
        "seed": seed,
        "ratios": ratios,
        "min_stratum_size": min_stratum_size,
        "n_movies": len(assignment),
        "counts": assignment["split"].value_counts().reindex(list(SPLITS), fill_value=0).to_dict(),
    }
    return Split(assignment=assignment, split_hash=assignment_hash(assignment), meta=meta)


def save_split(split: Split, directory: Path) -> None:
    """Write the assignment CSV and a JSON file with its hash and settings."""
    directory.mkdir(parents=True, exist_ok=True)
    split.assignment.to_csv(directory / ASSIGNMENT_FILE, index=False, lineterminator="\n")
    meta = {**split.meta, "split_hash": split.split_hash}
    (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_split(directory: Path) -> Split:
    """Reload a saved split and verify its hash."""
    csv, meta_path = directory / ASSIGNMENT_FILE, directory / META_FILE
    if not csv.is_file() or not meta_path.is_file():
        raise SplitError(f"no saved split in {directory}; run the `train` command (Part 1) first")
    assignment = pd.read_csv(csv)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual = assignment_hash(assignment)
    if actual != meta.get("split_hash"):
        raise SplitError(
            f"saved split in {directory} was modified: hash {actual} != {meta.get('split_hash')}"
        )
    return Split(assignment=assignment, split_hash=actual, meta=meta)


def slice_counts(catalog: pd.DataFrame, masks: dict[str, np.ndarray], columns: list[str]) -> pd.DataFrame:
    """Movies per slice value and split, e.g. how many Hindi films are in validation."""
    rows = []
    for column in columns:
        values = catalog[column].fillna("unknown").astype(str).to_numpy()
        for value in sorted(set(values)):
            row = {"slice_column": column, "slice": value}
            for name, mask in masks.items():
                row[name] = int(((values == value) & mask).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def assert_no_test_rows(used_ids: np.ndarray, split: Split, step: str) -> None:
    """Guard for tuning / selection: raise if any test movie is about to be used."""
    overlap = np.intersect1d(np.asarray(used_ids), split.ids("test"))
    if overlap.size:
        raise TestSplitAccessError(
            f"{step} must not use test movies; got {overlap.size} (e.g. {overlap[:5]})"
        )
