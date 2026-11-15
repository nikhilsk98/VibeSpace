"""Candidate models: shapes, [0, 1] outputs, save/load, seed reproducibility, model-specific behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from vibespace_model.models import CANDIDATES, create_model, load_model
from vibespace_model.models.mlp import parse_hidden_sizes

PARAMS = {
    "baseline": {},
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 30,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_child_samples": 5,
        "early_stopping_rounds": 5,
        "n_jobs": 1,
    },
    "mlp": {"hidden_sizes": [16], "max_epochs": 15, "patience": 4, "batch_size": 32, "num_threads": 1},
}


@pytest.fixture
def xy(rng):
    X = rng.normal(size=(200, 10)).astype(np.float32)
    W = rng.normal(size=(10, 12))
    Y = 1 / (1 + np.exp(-(X @ W) * 0.5))
    return X[:150], Y[:150], X[150:], Y[150:]


@pytest.mark.parametrize("name", CANDIDATES)
def test_shapes_and_range(name, xy) -> None:
    X, Y, Xv, Yv = xy
    model = create_model(name, PARAMS[name], seed=3).fit(X, Y, Xv, Yv)
    pred = model.predict(Xv * 10)  # extreme inputs still map into [0, 1]
    assert pred.shape == (50, 12)
    assert pred.min() >= 0.0 and pred.max() <= 1.0
    assert model.get_params()["seed"] == 3


@pytest.mark.parametrize("name", CANDIDATES)
def test_save_load_identical_predictions(name, xy, tmp_path) -> None:
    X, Y, Xv, Yv = xy
    model = create_model(name, PARAMS[name], seed=3).fit(X, Y, Xv, Yv)
    model.save(tmp_path / name)
    loaded = load_model(tmp_path / name)
    assert type(loaded) is type(model)
    np.testing.assert_array_equal(model.predict(Xv), loaded.predict(Xv))
    assert loaded.fit_info == model.fit_info


@pytest.mark.parametrize("name", CANDIDATES)
def test_seed_reproducibility(name, xy) -> None:
    X, Y, Xv, Yv = xy
    a = create_model(name, PARAMS[name], seed=11).fit(X, Y, Xv, Yv).predict(Xv)
    b = create_model(name, PARAMS[name], seed=11).fit(X, Y, Xv, Yv).predict(Xv)
    np.testing.assert_array_equal(a, b)


def test_models_beat_baseline_on_learnable_data(xy) -> None:
    X, Y, Xv, Yv = xy
    base = np.abs(create_model("baseline").fit(X, Y).predict(Xv) - Yv).mean()
    for name in ("ridge", "lightgbm", "mlp"):
        mae = np.abs(create_model(name, PARAMS[name]).fit(X, Y, Xv, Yv).predict(Xv) - Yv).mean()
        assert mae < base, name


def test_baseline_predicts_training_mean(xy) -> None:
    X, Y, Xv, _ = xy
    pred = create_model("baseline").fit(X, Y).predict(Xv)
    np.testing.assert_allclose(pred[0], Y.mean(axis=0))
    weights = np.where(np.arange(len(X)) < 10, 100.0, 1.0)
    weighted = create_model("baseline").fit(X, Y, sample_weight=weights).predict(Xv)
    np.testing.assert_allclose(weighted[0], np.average(Y, axis=0, weights=weights))


def test_lightgbm_early_stopping_records_best_iterations(xy) -> None:
    X, Y, Xv, Yv = xy
    model = create_model("lightgbm", {**PARAMS["lightgbm"], "n_estimators": 300}).fit(X, Y, Xv, Yv)
    assert len(model.fit_info["best_iterations"]) == 12
    assert all(1 <= b <= 300 for b in model.fit_info["best_iterations"])
    no_val = create_model("lightgbm", {**PARAMS["lightgbm"], "n_estimators": 20}).fit(X, Y)
    assert no_val.fit_info["best_iterations"] == [20] * 12


def test_mlp_epoch_callback_and_early_stopping(xy) -> None:
    X, Y, Xv, Yv = xy
    seen = []
    model = create_model("mlp", {**PARAMS["mlp"], "max_epochs": 8}).fit(
        X, Y, Xv, Yv, epoch_callback=lambda epoch, mae: seen.append((epoch, mae))
    )
    assert [e for e, _ in seen] == list(range(1, len(seen) + 1))
    assert all(np.isfinite(m) for _, m in seen)
    assert 1 <= model.fit_info["best_epoch"] <= 8
    without_val = create_model("mlp", {**PARAMS["mlp"], "max_epochs": 3}).fit(X, Y)
    assert without_val.fit_info["best_epoch"] == 3
    weighted = create_model("mlp", {**PARAMS["mlp"], "max_epochs": 3}).fit(
        X, Y, sample_weight=np.ones(len(X)) * 2
    )
    np.testing.assert_allclose(weighted.predict(Xv), without_val.predict(Xv), atol=1e-6)


def test_ridge_exposes_coefficients(xy) -> None:
    X, Y, _, _ = xy
    model = create_model("ridge", {"alpha": 0.5}).fit(X, Y)
    assert model.coef_.shape == (12, 10)
    assert model.intercept_.shape == (12,)


def test_errors(xy) -> None:
    X, Y, _, _ = xy
    with pytest.raises(ValueError, match="unknown model type"):
        create_model("xgboost")
    with pytest.raises(RuntimeError, match="before fit"):
        create_model("ridge").predict(X)
    with pytest.raises(ValueError, match="expected X"):
        create_model("ridge").fit(X, Y[:10])


def test_parse_hidden_sizes() -> None:
    assert parse_hidden_sizes("256-128") == [256, 128]
    assert parse_hidden_sizes(64) == [64]
    assert parse_hidden_sizes([32, 16]) == [32, 16]
