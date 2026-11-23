"""Tests for scripts.validate and scripts.gx_suites (Great Expectations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import gx_suites as gs
from scripts import validate as va
from scripts.build_dna_labels import DNA_DIMENSIONS
from scripts.exceptions import ValidationFailedError


def good_catalog(n: int = 5) -> pd.DataFrame:
    """A clean catalog as written by the clean step (``genres`` is a list per movie)."""
    df = pd.DataFrame({
        "movieId": range(1, n + 1), "tmdbId": range(101, 101 + n), "title": [f"T{i}" for i in range(n)],
        "runtime": [100.0] * n, "year": pd.array([2000] * n, dtype="Int64"), "language": ["en"] * n,
        "language_group": ["english"] * n, "era": ["2000_plus"] * n, "popularity_bucket": ["mid"] * n,
        "primary_genre": ["comedy"] * n, "genres": [np.array(["comedy", "drama"])] * n,
        "popularity": [12.5] * n, "vote_count": [300.0] * n, "vote_average": [6.8] * n,
    })
    for flag in gs.MODEL_FLAG_COLUMNS:
        df[flag] = False
    return df


def catalog_report(cfg, catalog: pd.DataFrame) -> dict:
    return va.run_suites({"processed_catalog": va.catalog_frame(catalog)},
                         {"processed_catalog": gs.processed_suites(cfg)["processed_catalog"]})


def failed_expectations(report: dict, suite: str = "processed_catalog") -> set[tuple[str, str]]:
    return {(f["expectation"], f["kwargs"].get("column")) for f in report["suites"][suite]["failures"]}


def test_suite_passes_on_good_data(cfg):
    report = catalog_report(cfg, good_catalog())
    assert report["success"] and report["suites"]["processed_catalog"]["failed"] == 0


def test_catalog_frame_replaces_genre_lists():
    df = good_catalog(2)
    df.at[1, "genres"] = np.array([], dtype=object)
    frame = va.catalog_frame(df)
    assert "genres" not in frame.columns
    assert frame["genre_count"].tolist() == [2, 0] and frame["genres_known"].tolist() == [True, True]


@pytest.mark.parametrize("change,expected", [
    (lambda df: df.__setitem__("genres", [np.array([], dtype=object)] + [np.array(["comedy"])] * 4),
     ("expect_column_values_to_be_between", "genre_count")),
    (lambda df: df.__setitem__("genres", [np.array(["comedy", "space_opera"])] + [np.array(["comedy"])] * 4),
     ("expect_column_values_to_be_in_set", "genres_known")),
    (lambda df: df.__setitem__("popularity", [None, 1.0, 2.0, 3.0, 4.0]),
     ("expect_column_values_to_not_be_null", "popularity")),
    (lambda df: df.__setitem__("popularity", [-1.0, 1.0, 2.0, 3.0, 4.0]),
     ("expect_column_values_to_be_between", "popularity")),
    (lambda df: df.__setitem__("vote_count", [-5.0, 1.0, 2.0, 3.0, 4.0]),
     ("expect_column_values_to_be_between", "vote_count")),
    (lambda df: df.__setitem__("vote_average", [11.0, 1.0, 2.0, 3.0, 4.0]),
     ("expect_column_values_to_be_between", "vote_average")),
    (lambda df: df.__setitem__("vote_average", [None, 1.0, 2.0, 3.0, 4.0]),
     ("expect_column_values_to_not_be_null", "vote_average")),
    (lambda df: df.drop(columns=["flag_year_mismatch"], inplace=True),
     ("expect_column_to_exist", "flag_year_mismatch")),
    (lambda df: df.__setitem__("flag_short_overview", [None, False, False, False, False]),
     ("expect_column_values_to_not_be_null", "flag_short_overview")),
])
def test_model_inputs_are_validated(cfg, change, expected):
    df = good_catalog()
    change(df)
    report = catalog_report(cfg, df)
    assert not report["success"]
    assert expected in failed_expectations(report)


@pytest.mark.parametrize("column,value,expectation", [
    ("movieId", 1, "expect_column_values_to_be_unique"),
    ("language", "ENG", "expect_column_values_to_match_regex"),
    ("language_group", "martian", "expect_column_values_to_be_in_set"),
    ("year", 1700, "expect_column_values_to_be_between"),
    ("title", None, "expect_column_values_to_not_be_null"),
])
def test_suite_failures_detected(cfg, column, value, expectation):
    df = good_catalog()
    df.loc[1, column] = value
    report = catalog_report(cfg, df)
    assert not report["success"]
    failed = {f["expectation"] for f in report["suites"]["processed_catalog"]["failures"]}
    assert expectation in failed
    with pytest.raises(ValidationFailedError, match=column):
        va.raise_on_failure(report, "Processed")


def test_dna_scores_out_of_range(cfg):
    labels = pd.DataFrame({"movieId": [1, 2], "mapping_version": ["v1", "v1"]})
    for dim in DNA_DIMENSIONS:
        labels[dim] = [0.2, 0.8]
    labels.loc[0, "humor"] = 1.5
    report = va.run_suites({"processed_dna_labels": labels},
                           {"processed_dna_labels": gs.processed_suites(cfg)["processed_dna_labels"]})
    assert not report["success"]


def test_embedding_frame_and_dim_check(cfg):
    emb = np.eye(3, 384, dtype=np.float32)
    index = pd.DataFrame({"row": [0, 1, 2], "movieId": [1, 2, 3], "encoder": [cfg.encoder_model] * 3})
    frame = va.embedding_frame(emb, index, {1, 2, 3})
    assert frame["dim"].unique().tolist() == [384] and frame["is_finite"].all()
    suite = {"processed_embeddings": gs.processed_suites(cfg)["processed_embeddings"]}
    assert va.run_suites({"processed_embeddings": frame}, suite)["success"]
    bad = va.embedding_frame(np.eye(3, 100, dtype=np.float32), index, {1, 2})
    report = va.run_suites({"processed_embeddings": bad}, suite)
    failed = {f["kwargs"].get("column") for f in report["suites"]["processed_embeddings"]["failures"]}
    assert {"dim", "in_catalog"} <= failed
    with pytest.raises(ValidationFailedError, match="rows"):
        va.embedding_frame(emb, index.iloc[:2], {1})


def test_missing_frame_raises(cfg):
    with pytest.raises(KeyError):
        va.run_suites({}, {"raw_genome_tags": gs.raw_suites(cfg)["raw_genome_tags"]})


def test_export_suites(cfg, tmp_path):
    va.export_suites(gs.raw_suites(cfg), tmp_path / "exp")
    files = sorted(p.name for p in (tmp_path / "exp").glob("*.json"))
    assert "raw_tmdb_movies.json" in files and "join_quality.json" in files


def test_match_rate_suite(cfg):
    suite = {"join_quality": gs.raw_suites(cfg)["join_quality"]}
    assert va.run_suites({"join_quality": pd.DataFrame([{"match_rate": 0.95}])}, suite)["success"]
    assert not va.run_suites({"join_quality": pd.DataFrame([{"match_rate": 0.5}])}, suite)["success"]


def test_empty_table_fails_row_count(cfg):
    suite = {"raw_genome_tags": gs.raw_suites(cfg)["raw_genome_tags"]}
    report = va.run_suites({"raw_genome_tags": pd.DataFrame({"tagId": [], "tag": []})}, suite)
    assert not report["success"]


@pytest.mark.parametrize("change,column", [
    (lambda ix: ix.__setitem__("row", [0, 2, 1]), "row_in_order"),
    (lambda ix: ix.__setitem__("row", [1, 2, 3]), "row_in_order"),
    (lambda ix: ix.__setitem__("movieId", [1, 1, 3]), "movieId"),
    (lambda ix: ix.__setitem__("encoder", ["other-encoder"] * 3), "encoder"),
    (lambda ix: ix.drop(columns=["encoder"], inplace=True), "encoder"),
])
def test_embedding_index_contract(cfg, change, column):
    emb = np.eye(3, 384, dtype=np.float32)
    index = pd.DataFrame({"row": [0, 1, 2], "movieId": [1, 2, 3], "encoder": [cfg.encoder_model] * 3})
    change(index)
    frame = va.embedding_frame(emb, index, {1, 2, 3})
    report = va.run_suites({"processed_embeddings": frame},
                           {"processed_embeddings": gs.processed_suites(cfg)["processed_embeddings"]})
    assert not report["success"]
    assert column in {c for _, c in failed_expectations(report, "processed_embeddings")}
