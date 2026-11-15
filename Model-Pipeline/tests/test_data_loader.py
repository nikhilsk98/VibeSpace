"""Loading, DVC hash verification, validation and joining of the pipeline outputs."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.build_fixtures import write_lock
from vibespace_model.data_loader import (
    dir_md5,
    file_md5,
    git_commit,
    load_dataset,
    output_md5,
    read_lock_hashes,
    sample_rows,
    verify_hashes,
)
from vibespace_model.errors import DataValidationError, DataVersionError, LeakageError


def _rewrite(tree, name: str, frame: pd.DataFrame, relock: bool = True) -> None:
    frame.to_parquet(tree["processed"] / name, index=False)
    if relock:
        write_lock(tree["data_pipeline"])


def test_happy_path_hash_check_and_join_counts(cfg) -> None:
    data = load_dataset(cfg)
    report = data.join_report
    assert report["labels_total"] == 240
    assert report["dropped_no_catalog_row"] == 5
    assert report["dropped_no_embedding"] == 3
    assert report["after_embedding_join"] == len(data) == 232
    assert report["catalog_rows_without_labels"] == 2
    assert data.labels.shape == (232, 12)
    assert data.embeddings.shape == (232, 16)
    assert data.embeddings.dtype == np.float32
    assert data.data_version["hash_check"] == "passed"
    assert data.data_version["mapping_version"] == "v1"
    assert set(data.data_version["dvc_lock_hashes"]) == {
        f"data/processed/{f}" for f in cfg.data["hash_checked_files"]
    }
    assert len(data.data_version["data_version_id"]) == 16
    assert (np.diff(data.movie_ids) > 0).all()


def test_rows_are_aligned_after_join(cfg, tree) -> None:
    data = load_dataset(cfg)
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet").set_index("movieId")
    index = pd.read_parquet(tree["processed"] / "plot_embeddings_index.parquet").set_index("movieId")
    raw_emb = np.load(tree["processed"] / "plot_embeddings.npy")
    for pos in (0, 17, len(data) - 1):
        movie = data.movie_ids[pos]
        np.testing.assert_allclose(data.labels[pos], labels.loc[movie, cfg.targets].to_numpy(dtype=float))
        np.testing.assert_array_equal(data.embeddings[pos], raw_emb[index.loc[movie, "row"]])
        assert data.catalog.loc[pos, "movieId"] == movie


def test_missing_file_is_reported(cfg, tree) -> None:
    (tree["processed"] / "plot_embeddings.npy").unlink()
    with pytest.raises(DataVersionError, match="missing pipeline output"):
        load_dataset(cfg)
    with pytest.raises(DataValidationError, match="missing embeddings file"):
        load_dataset(cfg, skip_hash_check=True)


def test_hash_mismatch_is_reported(cfg, tree) -> None:
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet")
    labels.loc[0, "energy"] = 0.5
    _rewrite(tree, "movie_dna_labels.parquet", labels, relock=False)
    with pytest.raises(DataVersionError, match="hash mismatch for data/processed/movie_dna_labels.parquet"):
        load_dataset(cfg)
    assert load_dataset(cfg, skip_hash_check=True).data_version["hash_check"] == "skipped"


def test_lock_problems(cfg, tree) -> None:
    lock = tree["lock"]
    text = lock.read_text().replace("data/processed/plot_embeddings.npy", "data/processed/other.npy")
    lock.write_text(text)
    with pytest.raises(DataVersionError, match="not an output"):
        verify_hashes(cfg)
    lock.write_text("stages: {}\n")
    with pytest.raises(DataVersionError, match="no outputs"):
        read_lock_hashes(lock)
    lock.unlink()
    with pytest.raises(DataVersionError, match="not found"):
        read_lock_hashes(lock)


def test_processed_dir_outside_pipeline_dir(cfg, tmp_path) -> None:
    cfg.paths.data_pipeline_dir = tmp_path / "somewhere-else"
    (cfg.paths.data_pipeline_dir).mkdir()
    (cfg.paths.data_pipeline_dir / "dvc.lock").write_text("stages: {a: {outs: [{path: x, md5: y}]}}\n")
    with pytest.raises(DataVersionError, match="not inside"):
        verify_hashes(cfg)


@pytest.mark.parametrize("which", ["labels", "catalog"])
def test_duplicate_movie_ids(cfg, tree, which) -> None:
    name = "movie_dna_labels.parquet" if which == "labels" else "movies_clean.parquet"
    frame = pd.read_parquet(tree["processed"] / name)
    frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    _rewrite(tree, name, frame)
    with pytest.raises(DataValidationError, match=f"duplicate movieIds in {which}"):
        load_dataset(cfg)


@pytest.mark.parametrize(
    ("value", "message"), [(1.2, "outside"), (-0.1, "outside"), (np.nan, "missing values")]
)
def test_label_problems(cfg, tree, value, message) -> None:
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet")
    labels.loc[3, "humor"] = value
    _rewrite(tree, "movie_dna_labels.parquet", labels)
    with pytest.raises(DataValidationError, match=message):
        load_dataset(cfg)


def test_missing_label_and_catalog_columns(cfg, tree) -> None:
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet").drop(columns=["romance"])
    _rewrite(tree, "movie_dna_labels.parquet", labels)
    with pytest.raises(DataValidationError, match="missing columns"):
        load_dataset(cfg)


def test_catalog_missing_feature_column(cfg, tree) -> None:
    catalog = pd.read_parquet(tree["processed"] / "movies_clean.parquet").drop(columns=["runtime"])
    _rewrite(tree, "movies_clean.parquet", catalog)
    with pytest.raises(DataValidationError, match="catalog is missing columns"):
        load_dataset(cfg)


def test_wrong_embedding_dimension(cfg, tree) -> None:
    emb = np.load(tree["processed"] / "plot_embeddings.npy")
    np.save(tree["processed"] / "plot_embeddings.npy", emb[:, :10])
    write_lock(tree["data_pipeline"])
    with pytest.raises(DataValidationError, match="shape"):
        load_dataset(cfg)


def test_index_order_mismatch(cfg, tree) -> None:
    index = pd.read_parquet(tree["processed"] / "plot_embeddings_index.parquet")
    index["row"] = index["row"].to_numpy()[::-1]
    _rewrite(tree, "plot_embeddings_index.parquet", index)
    with pytest.raises(DataValidationError, match="not 0..n-1"):
        load_dataset(cfg)


def test_index_length_and_duplicates(cfg, tree) -> None:
    index = pd.read_parquet(tree["processed"] / "plot_embeddings_index.parquet")
    _rewrite(tree, "plot_embeddings_index.parquet", index.iloc[:-1])
    with pytest.raises(DataValidationError, match="rows but the matrix"):
        load_dataset(cfg)
    index.loc[1, "movieId"] = index.loc[0, "movieId"]
    _rewrite(tree, "plot_embeddings_index.parquet", index)
    with pytest.raises(DataValidationError, match="duplicate movieIds in the embedding index"):
        load_dataset(cfg)


def test_non_finite_embeddings(cfg, tree) -> None:
    emb = np.load(tree["processed"] / "plot_embeddings.npy")
    emb[4, 2] = np.nan
    np.save(tree["processed"] / "plot_embeddings.npy", emb)
    write_lock(tree["data_pipeline"])
    with pytest.raises(DataValidationError, match="NaN"):
        load_dataset(cfg)


def test_leakage_guard_rejects_genome_columns_in_catalog(cfg, tree) -> None:
    catalog = pd.read_parquet(tree["processed"] / "movies_clean.parquet")
    catalog["genome_relevance_0042"] = 0.3
    _rewrite(tree, "movies_clean.parquet", catalog)
    with pytest.raises(LeakageError, match="genome_relevance_0042"):
        load_dataset(cfg)


def test_no_overlap_at_all(cfg, tree) -> None:
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet")
    labels["movieId"] = labels["movieId"] + 1_000_000
    _rewrite(tree, "movie_dna_labels.parquet", labels)
    with pytest.raises(DataValidationError, match="no movie has labels"):
        load_dataset(cfg)


def test_sample_size_is_deterministic(tree) -> None:
    from vibespace_model.config import load_config

    a = load_dataset(load_config(tree["config"], sample_size=50))
    b = load_dataset(load_config(tree["config"], sample_size=50))
    assert len(a) == 50
    assert a.join_report["sample_size"] == 50
    np.testing.assert_array_equal(a.movie_ids, b.movie_ids)
    np.testing.assert_array_equal(sample_rows(10, None, 1), np.arange(10))
    np.testing.assert_array_equal(sample_rows(10, 20, 1), np.arange(10))


def test_dir_md5_matches_dvc_format(tmp_path) -> None:
    folder = tmp_path / "out"
    (folder / "sub").mkdir(parents=True)
    (folder / "b.txt").write_bytes(b"bee\n")
    (folder / "sub" / "a.txt").write_bytes(b"ay\n")
    entries = [
        {"md5": hashlib.md5(b"bee\n").hexdigest(), "relpath": "b.txt"},  # noqa: S324
        {"md5": hashlib.md5(b"ay\n").hexdigest(), "relpath": "sub/a.txt"},  # noqa: S324
    ]
    expected = hashlib.md5(json.dumps(entries, sort_keys=True).encode()).hexdigest() + ".dir"  # noqa: S324
    assert dir_md5(folder) == expected == output_md5(folder)
    assert output_md5(folder / "b.txt") == file_md5(folder / "b.txt")


def test_git_commit_env_override_and_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GIT_COMMIT", "abc123")
    assert git_commit(tmp_path) == {"git_commit": "abc123", "git_dirty": None}
    monkeypatch.delenv("GIT_COMMIT")
    assert git_commit(tmp_path)["git_commit"] == "unknown"  # not a Git repository


def test_subset_by_index_and_mask(cfg) -> None:
    data = load_dataset(cfg)
    by_index = data.subset(np.array([0, 2, 4]))
    mask = np.zeros(len(data), dtype=bool)
    mask[[0, 2, 4]] = True
    by_mask = data.subset(mask)
    np.testing.assert_array_equal(by_index.movie_ids, by_mask.movie_ids)
    assert len(by_index) == 3
