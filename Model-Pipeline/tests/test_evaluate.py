"""Metrics against hand-computed values, tables and plots."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from vibespace_model.evaluate import (
    comparison_table,
    compute_metrics,
    improvement_over_baseline,
    per_dimension_table,
    plot_model_comparison,
    plot_per_dimension,
    plot_pred_vs_true,
    write_json,
)

TARGETS = ["a", "b"]


def test_metrics_match_hand_computed_values() -> None:
    Y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.5, 1.0]])
    P = np.array([[0.1, 0.9], [0.5, 0.7], [0.7, 0.2], [0.5, 0.6]])
    metrics = compute_metrics(Y, P, TARGETS)
    a, b = metrics["per_dimension"]["a"], metrics["per_dimension"]["b"]
    assert a["mae"] == pytest.approx((0.1 + 0 + 0.3 + 0) / 4)
    assert a["rmse"] == pytest.approx(math.sqrt((0.01 + 0 + 0.09 + 0) / 4))
    # R² = 1 - SSE / SST with mean 0.5 for dimension a: SST = 0.25 + 0 + 0.25 + 0 = 0.5, SSE = 0.1
    assert a["r2"] == pytest.approx(1 - 0.1 / 0.5)
    assert b["mae"] == pytest.approx((0.1 + 0.2 + 0.2 + 0.4) / 4)
    # Spearman of dimension a: ranks y = [1, 2.5, 4, 2.5], p = [1, 2.5, 4, 2.5] -> 1.0
    assert a["spearman"] == pytest.approx(1.0)
    assert metrics["overall"]["mae"] == pytest.approx((a["mae"] + b["mae"]) / 2)
    assert metrics["overall"]["mae"] == pytest.approx(np.abs(Y - P).mean())
    assert metrics["overall"]["n"] == 4


def test_constant_predictions_have_zero_spearman_and_r2() -> None:
    Y = np.array([[0.1, 0.2], [0.3, 0.9], [0.5, 0.4]])
    P = np.tile(Y.mean(axis=0), (3, 1))
    metrics = compute_metrics(Y, P, TARGETS)
    assert metrics["overall"]["spearman"] == 0.0
    assert metrics["overall"]["r2"] == pytest.approx(0.0)
    constant_truth = compute_metrics(np.ones((3, 2)) * 0.5, P, TARGETS)
    assert constant_truth["per_dimension"]["a"]["r2"] == 0.0


def test_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(np.zeros((3, 2)), np.zeros((3, 3)), TARGETS)


def test_improvement_over_baseline() -> None:
    assert improvement_over_baseline(0.17, 0.20) == pytest.approx(0.15)
    assert improvement_over_baseline(0.1, 0.0) == 0.0


def _results() -> dict:
    Y = np.random.default_rng(0).random((20, 2))
    return {
        "baseline": {"val": compute_metrics(Y, np.full_like(Y, 0.5), TARGETS), "train_seconds": 0.0},
        "good": {
            "val": compute_metrics(Y, Y * 0.9 + 0.05, TARGETS),
            "train_seconds": 1.5,
            "test": compute_metrics(Y, Y, TARGETS),
        },
    }


def test_tables_and_plots(tmp_path) -> None:
    results = _results()
    table = comparison_table(results)
    assert list(table["model"]) == ["good", "baseline"]
    assert table.loc[0, "improvement_vs_baseline"] > 0
    assert table.loc[0, "test_mae"] == 0.0
    per_dim = per_dimension_table(results)
    assert list(per_dim.index) == TARGETS
    assert list(per_dim.columns) == ["baseline", "good"]
    assert plot_model_comparison(table, tmp_path / "bar.png").stat().st_size > 0
    assert plot_per_dimension(per_dim, tmp_path / "dims.png").stat().st_size > 0
    Y = np.random.default_rng(1).random((30, 2))
    assert plot_pred_vs_true(Y, Y, TARGETS, ["a"], tmp_path / "scatter.png", "t").stat().st_size > 0


def test_write_json_handles_numpy(tmp_path) -> None:
    write_json(
        tmp_path / "x.json",
        {
            "a": np.float32(1.5),
            "b": np.int64(2),
            "c": np.arange(2),
            "d": tmp_path,
            "e": pd.Timestamp("2026-01-01"),
        },
    )
    text = (tmp_path / "x.json").read_text()
    assert '"a": 1.5' in text and '"b": 2' in text and "2026-01-01" in text
