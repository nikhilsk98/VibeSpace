"""Part 3 explainability: SHAP grouping, explainers per family, permutation importance, plots."""

from __future__ import annotations

import numpy as np
import pytest

from vibespace_model import explain
from vibespace_model.models import create_model

GROUPS = {"plot_embedding": [0, 1, 2, 3], "genres": [4, 5], "runtime": [6], "flags": []}
TARGETS = ["energy", "humor"]


def test_group_sums_equal_member_totals() -> None:
    values = np.random.default_rng(0).normal(size=(30, 7, 2))
    table = explain.group_importance(values, GROUPS, TARGETS)
    assert list(table.index) == list(GROUPS)
    assert list(table.columns) == TARGETS
    expected_embedding = np.abs(values[:, :4, :]).sum(axis=1).mean(axis=0)
    np.testing.assert_allclose(table.loc["plot_embedding"].to_numpy(), expected_embedding)
    np.testing.assert_allclose(table.loc["runtime"].to_numpy(), np.abs(values[:, 6, :]).mean(axis=0))
    assert (table.loc["flags"] == 0).all()
    # all groups together equal the total |SHAP| (each feature belongs to exactly one group)
    np.testing.assert_allclose(table.sum(axis=0).to_numpy(), np.abs(values).sum(axis=1).mean(axis=0))
    shares = explain.importance_shares(table)
    assert shares.sum() == pytest.approx(1.0)
    assert shares.idxmax() == "plot_embedding"


def test_linear_shap_is_exact_for_ridge(rng) -> None:
    X = rng.normal(size=(80, 7)).astype(np.float32)
    Y = 1 / (1 + np.exp(-X[:, :2]))
    model = create_model("ridge", {"alpha": 0.1}).fit(X, Y)
    background = X[:40]
    values = explain.shap_values(model, X[40:], background, seed=0)
    assert values.shape == (40, 7, 2)
    raw = model._predict_raw(X[40:])
    base = model._predict_raw(background).mean(axis=0)
    np.testing.assert_allclose(values.sum(axis=1) + base, raw, rtol=1e-4, atol=1e-6)


def test_tree_and_gradient_explainers(rng) -> None:
    X = rng.normal(size=(120, 7)).astype(np.float32)
    Y = 1 / (1 + np.exp(-(X[:, :2] * 2)))
    lgbm = create_model(
        "lightgbm", {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 5, "n_jobs": 1}
    ).fit(X, Y)
    tree_values = explain.shap_values(lgbm, X[:15], X[100:], seed=0)
    assert tree_values.shape == (15, 7, 2)
    import shap

    expected = shap.TreeExplainer(lgbm.boosters[0]).expected_value
    np.testing.assert_allclose(  # TreeSHAP is additive: contributions + expected value = raw prediction
        tree_values[:, :, 0].sum(axis=1) + expected, lgbm.boosters[0].predict(X[:15]), atol=1e-6
    )
    assert np.abs(tree_values[:, :2, :]).sum() > np.abs(tree_values[:, 2:, :]).sum()
    mlp = create_model("mlp", {"hidden_sizes": [8], "max_epochs": 5, "num_threads": 1}).fit(X, Y)
    grad_values = explain.shap_values(mlp, X[:10], X[100:], seed=0, nsamples=8)
    assert grad_values.shape == (10, 7, 2)
    assert np.isfinite(grad_values).all()
    with pytest.raises(ValueError, match="no SHAP explainer"):
        explain.shap_values(create_model("baseline").fit(X, Y), X[:5], X[:5], seed=0)


def test_permutation_importance_finds_the_known_feature(rng) -> None:
    X = rng.normal(size=(400, 7)).astype(np.float32)
    Y = np.column_stack([1 / (1 + np.exp(-2 * X[:, 4])), 1 / (1 + np.exp(-2 * X[:, 5]))])
    model = create_model("ridge", {"alpha": 0.1}).fit(X, Y)
    table = explain.permutation_importance_by_group(model, X, Y, GROUPS, seed=0, repeats=3)
    assert table["mae_increase"].idxmax() == "genres"  # features 4 and 5 drive the labels
    assert table.loc["genres", "mae_increase"] > 10 * abs(table.loc["plot_embedding", "mae_increase"])
    assert table.loc["flags", "mae_increase"] == 0.0


def test_top_features_and_sampling(rng) -> None:
    values = rng.normal(size=(10, 4, 2))
    values[:, 3, :] *= 10
    names = ["emb_000", "emb_001", "genre=comedy", "runtime_scaled"]
    top = explain.top_features(values, names, k=5)
    assert list(top["feature"]) == ["runtime_scaled", "genre=comedy"]
    rows = explain.sample_rows(100, 10, seed=1)
    assert len(rows) == 10 and (np.diff(rows) > 0).all()
    np.testing.assert_array_equal(explain.sample_rows(5, 10, seed=1), np.arange(5))


def test_plots(tmp_path, rng) -> None:
    values = rng.normal(size=(20, 7, 2))
    table = explain.group_importance(values, GROUPS, TARGETS)
    shares = {"ridge": explain.importance_shares(table)}
    X = rng.normal(size=(40, 7)).astype(np.float32)
    model = create_model("ridge").fit(X, rng.random((40, 2)))
    perm = {"ridge": explain.permutation_importance_by_group(model, X, rng.random((40, 2)), GROUPS, 0, 2)}
    assert explain.plot_group_importance(shares, perm, tmp_path / "g.png").stat().st_size > 0
    assert explain.plot_heatmap(table, tmp_path / "h.png", "t").stat().st_size > 0
    names = ["emb_000", "emb_001", "emb_002", "emb_003", "genre=a", "genre=b", "runtime_scaled"]
    path = explain.plot_metadata_summary(
        values, X[:20], names, TARGETS, ["energy", "humor"], tmp_path / "m.png"
    )
    assert path.stat().st_size > 0
