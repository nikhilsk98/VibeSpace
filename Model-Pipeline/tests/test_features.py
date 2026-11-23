"""Feature matrix: shapes, groups, train-only fitting, missing values, unseen categories, leakage guard."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vibespace_model.data_loader import load_dataset
from vibespace_model.errors import DataValidationError, LeakageError
from vibespace_model.features import FEATURE_GROUPS, FeatureBuilder, _as_genre_list, assert_no_leakage


@pytest.fixture
def data(cfg):
    return load_dataset(cfg)


def _builder(cfg) -> FeatureBuilder:
    return FeatureBuilder(cfg.features, cfg.data["embedding_dim"])


def test_shapes_and_groups(cfg, data) -> None:
    builder = _builder(cfg)
    X = builder.fit_transform(data.catalog, data.embeddings)
    assert X.dtype == np.float32
    assert X.shape == (len(data), len(builder.feature_names))
    assert tuple(builder.groups) == FEATURE_GROUPS
    covered = sorted(i for idx in builder.groups.values() for i in idx)
    assert covered == list(range(X.shape[1]))
    assert len(builder.groups["plot_embedding"]) == 16
    assert builder.feature_names[:3] == ["emb_000", "emb_001", "emb_002"]
    assert "runtime_scaled" in builder.feature_names
    assert {"release_year_scaled", "release_year_missing"} <= set(builder.feature_names)
    assert {"log1p_popularity", "log1p_vote_count", "vote_average_scaled"} <= set(builder.feature_names)
    assert builder.group_of("genre=comedy") == "genres"
    with pytest.raises(KeyError):
        builder.group_of("nope")


def test_scalers_fit_on_training_rows_only(cfg, data) -> None:
    train, other = data.subset(np.arange(0, 150)), data.subset(np.arange(150, len(data)))
    builder = _builder(cfg).fit(train.catalog, train.embeddings)
    state_before = repr(builder.state)
    X_train = builder.transform(train.catalog, train.embeddings)
    builder.transform(other.catalog, other.embeddings)
    assert repr(builder.state) == state_before  # transform never refits
    runtime_col = builder.feature_names.index("runtime_scaled")
    assert abs(float(X_train[:, runtime_col].mean())) < 1e-5  # standardised with train statistics
    pop_col = builder.feature_names.index("log1p_popularity")
    expected_mean = np.log1p(train.catalog["popularity"]).mean()
    assert builder.state["popularity"]["popularity"]["mean"] == pytest.approx(expected_mean)
    assert pop_col > runtime_col


def test_missing_year_gets_median_and_flag(cfg, data) -> None:
    builder = _builder(cfg)
    X = builder.fit_transform(data.catalog, data.embeddings)
    missing = data.catalog["year"].isna().to_numpy()
    assert missing.sum() > 0
    flag = X[:, builder.feature_names.index("release_year_missing")]
    np.testing.assert_array_equal(flag, missing.astype(np.float32))
    year = X[:, builder.feature_names.index("release_year_scaled")]
    median = np.nanmedian(pd.to_numeric(data.catalog["year"], errors="coerce").astype(float))
    expected = (median - builder.state["year"]["mean"]) / builder.state["year"]["std"]
    np.testing.assert_allclose(year[missing], expected, rtol=1e-5)


def test_invalid_runtime_uses_training_median(cfg, data) -> None:
    builder = _builder(cfg).fit(data.catalog, data.embeddings)
    rt = builder.state["runtime"]
    zero_runtime = data.catalog["runtime"].to_numpy() <= 0
    X = builder.transform(data.catalog, data.embeddings)
    expected = (np.clip(rt["fill"], *rt["clip"]) - rt["mean"]) / rt["std"]
    np.testing.assert_allclose(
        X[zero_runtime, builder.feature_names.index("runtime_scaled")], expected, rtol=1e-5
    )


def test_unseen_categories_are_all_zero(cfg, data) -> None:
    builder = _builder(cfg).fit(data.catalog, data.embeddings)
    unseen = data.catalog.iloc[:3].copy()
    unseen["language_group"] = "klingon"
    unseen["genres"] = [np.array(["space_opera"], dtype=object)] * 3
    X = builder.transform(unseen, data.embeddings[:3])
    for group in ("language", "genres"):
        assert X[:, builder.groups[group]].sum() == 0


def test_save_load_round_trip(cfg, data, tmp_path) -> None:
    builder = _builder(cfg).fit(data.catalog, data.embeddings)
    builder.save(tmp_path / "features.json")
    loaded = FeatureBuilder.load(tmp_path / "features.json")
    np.testing.assert_array_equal(
        builder.transform(data.catalog, data.embeddings), loaded.transform(data.catalog, data.embeddings)
    )
    assert loaded.feature_names == builder.feature_names


def test_leakage_guard() -> None:
    for column in (
        "genome_tag_12",
        "tag_relevance",
        "relevance",
        "tagId",
        "tags",
        "tag=funny",
        "humor",
        "darkness_raw",
        "mapping_version",
    ):
        with pytest.raises(LeakageError):
            assert_no_leakage([column])
    with pytest.raises(LeakageError, match="energy"):
        assert_no_leakage(["runtime", "energy"], targets=["energy"])
    assert_no_leakage(["runtime", "year_raw", "language_group", "keywords", "emb_001"])


def test_leakage_in_transform_input_is_caught(cfg, data) -> None:
    builder = _builder(cfg).fit(data.catalog, data.embeddings)
    leaked = data.catalog.assign(genome_relevance=0.5)
    with pytest.raises(LeakageError):
        builder.transform(leaked, data.embeddings)


def test_input_checks(cfg, data) -> None:
    builder = _builder(cfg)
    with pytest.raises(RuntimeError, match="before fit"):
        builder.transform(data.catalog, data.embeddings)
    with pytest.raises(ValueError, match="shape"):
        builder.fit(data.catalog, data.embeddings[:, :5])
    with pytest.raises(ValueError, match="number of rows"):
        builder.fit(data.catalog, data.embeddings[:10])


def test_genre_parsing_variants() -> None:
    assert _as_genre_list("Comedy|Drama") == ["comedy", "drama"]
    assert _as_genre_list("(no genres listed)") == []
    assert _as_genre_list(None) == []
    assert _as_genre_list(float("nan")) == []
    assert _as_genre_list(np.array(["Action", " "], dtype=object)) == ["action"]
    assert _as_genre_list(5) == []


def _with_extra_flag(cfg, allow: bool):
    flags = [*cfg.features["flag_columns"], "flag_not_in_catalog"]
    return replace(cfg, features={**cfg.features, "flag_columns": flags, "allow_missing_columns": allow})


def test_missing_flag_column_stops_the_loader(cfg) -> None:
    expected = r"missing columns the model needs: \['flag_not_in_catalog'\]"
    with pytest.raises(DataValidationError, match=expected):
        load_dataset(_with_extra_flag(cfg, allow=False))


def test_missing_flag_column_stops_the_builder(cfg, data) -> None:
    strict = _with_extra_flag(cfg, allow=False)
    with pytest.raises(DataValidationError, match="flag_not_in_catalog"):
        FeatureBuilder(strict.features, cfg.data["embedding_dim"]).fit(data.catalog, data.embeddings)


def test_missing_flag_column_allowed_explicitly(cfg, caplog) -> None:
    relaxed = _with_extra_flag(cfg, allow=True)
    dataset = load_dataset(relaxed)
    builder = FeatureBuilder(relaxed.features, cfg.data["embedding_dim"])
    with caplog.at_level("WARNING"):
        builder.fit(dataset.catalog, dataset.embeddings)
    assert "flag_not_in_catalog" not in builder.feature_names
    assert "flag_short_overview" in builder.feature_names
    assert "allow_missing_columns" in caplog.text


def test_missing_flag_is_off_by_default_in_the_real_config() -> None:
    from vibespace_model.config import load_config

    real = load_config(Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml")
    assert real.features["allow_missing_columns"] is False
