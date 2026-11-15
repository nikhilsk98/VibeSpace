"""Part 3 bias: slice metrics with a built-in gap, gate logic, bootstrap CI, mitigation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vibespace_model import bias

CFG = {
    "min_gated_slice_size": 30,
    "max_mae_ratio": 1.25,
    "bootstrap": {"n_resamples": 300, "confidence": 0.95},
}


def _synthetic(gap: float = 2.0, seed: int = 0):
    """300 'good' movies, 100 'bad' movies with `gap` x larger errors, 10 'tiny' movies."""
    rng = np.random.default_rng(seed)
    groups = np.array(["good"] * 300 + ["bad"] * 100 + ["tiny"] * 10)
    Y = rng.random((410, 12))
    scale = np.where(groups == "bad", 0.1 * gap, 0.1)[:, None]
    noise = rng.choice([-1.0, 1.0], size=Y.shape) * scale
    catalog = pd.DataFrame({"language_group": groups, "era": rng.choice(["old", "new"], size=410)})
    return Y, Y + noise, catalog


def test_metric_frame_matches_manual() -> None:
    Y, P, catalog = _synthetic()
    frame = bias.metric_frame(Y, P, catalog["language_group"].to_numpy()).by_group
    manual = np.abs(Y[:300] - P[:300]).mean()
    assert frame.loc["good", "mae"] == pytest.approx(manual)
    assert int(frame.loc["tiny", "n"]) == 10


def test_slice_table_detects_gap_and_small_slices() -> None:
    Y, P, catalog = _synthetic(gap=2.0)
    table, overall = bias.evaluate_slices(Y, P, catalog, ["language_group", "era", "missing_column"], CFG)
    assert overall == pytest.approx(np.abs(Y - P).mean())
    rows = table.set_index("slice")
    assert rows.loc["bad", "status"] == "fail"
    assert rows.loc["bad", "mae_ratio"] > 1.25
    assert rows.loc["good", "status"] == "pass"
    assert rows.loc["tiny", "status"] == "insufficient data" and not rows.loc["tiny", "gated"]
    assert (
        rows.loc["tiny", "ci_low"] - 1e-12 <= rows.loc["tiny", "mae"] <= rows.loc["tiny", "ci_high"] + 1e-12
    )
    assert set(table["slice_column"]) == {"language_group", "era"}


def test_bias_gate_pass_fail_and_small_slice_exemption() -> None:
    Y, P, catalog = _synthetic(gap=2.0)
    table, overall = bias.evaluate_slices(Y, P, catalog, ["language_group"], CFG)
    failed = bias.bias_gate(table, overall, CFG)
    assert not failed.passed
    assert failed.checks["failing_slices"][0].startswith("language_group=bad")
    assert failed.checks["worst_slice"] == "language_group=bad"

    Y, P, catalog = _synthetic(gap=1.0)
    P[400:] = Y[400:] + 0.9  # the tiny slice is terrible, but too small to gate
    table, overall = bias.evaluate_slices(Y, P, catalog, ["language_group"], CFG)
    passed = bias.bias_gate(table, overall, CFG)
    assert passed.passed
    assert passed.checks["n_small_slices"] == 1
    assert table.set_index("slice").loc["tiny", "mae_ratio"] > 1.25
    assert bias.worst_ratio(table) < 1.25


def test_bootstrap_ci() -> None:
    errors = np.random.default_rng(3).random(200)
    lo, hi = bias.bootstrap_ci(errors, 500, 0.95, seed=1)
    assert lo < errors.mean() < hi
    assert (lo, hi) == bias.bootstrap_ci(errors, 500, 0.95, seed=1)
    narrow = bias.bootstrap_ci(errors, 500, 0.5, seed=1)
    assert narrow[0] > lo and narrow[1] < hi
    assert all(np.isnan(v) for v in bias.bootstrap_ci(np.array([]), 10))


def test_reweighting_weights() -> None:
    values = np.array(["a"] * 90 + ["b"] * 9 + ["c"] * 1)
    weights = bias.reweighting_weights(values, power=1.0, max_weight=1000)
    assert weights.mean() == pytest.approx(1.0)
    # inverse frequency: each slice gets the same total weight
    totals = {v: weights[values == v].sum() for v in "abc"}
    assert totals["a"] == pytest.approx(totals["b"]) == pytest.approx(totals["c"])
    capped = bias.reweighting_weights(values, power=1.0, max_weight=5)
    assert capped.mean() == pytest.approx(1.0)
    assert capped[values == "c"][0] / capped[values == "a"][0] < 100
    sqrt = bias.reweighting_weights(values, power=0.5, max_weight=1000)
    assert sqrt[values == "c"][0] < weights[values == "c"][0]


def test_resampling_only_touches_training_rows() -> None:
    values = np.array(["a"] * 100 + ["b"] * 5 + ["c"] * 20)
    idx = bias.resample_indices(values, min_share_of_largest=0.1, seed=0)
    assert idx.min() >= 0 and idx.max() < len(values)  # indices into the training split only
    assert set(range(len(values))) <= set(idx.tolist())  # every original row kept
    resampled = values[idx]
    assert (
        (resampled == "b").sum() == 10 and (resampled == "c").sum() == 20 and (resampled == "a").sum() == 100
    )
    np.testing.assert_array_equal(idx, bias.resample_indices(values, 0.1, seed=0))
    np.testing.assert_array_equal(bias.resample_indices(np.array(["a", "b"]), 0.1), [0, 1])


def test_choose_mitigation_column_and_gaps(tmp_path) -> None:
    Y, P, catalog = _synthetic(gap=2.0)
    table, overall = bias.evaluate_slices(Y, P, catalog, ["era", "language_group"], CFG)
    assert bias.choose_mitigation_column(table) == "language_group"
    assert bias.choose_mitigation_column(table, "era") == "era"
    gaps = bias.gap_by_column(table)
    assert gaps.loc[0, "slice_column"] == "language_group"
    assert gaps.loc[0, "worst_over_best"] > 1.5
    Y, P, catalog = _synthetic(gap=1.1)
    passing, _ = bias.evaluate_slices(Y, P, catalog, ["era", "language_group"], CFG)
    assert bias.choose_mitigation_column(passing) in {"era", "language_group"}
    assert bias.plot_slice_mae(table, overall, 1.25, tmp_path / "s.png", "t").stat().st_size > 0


def test_slice_values_handles_missing_and_bool() -> None:
    frame = pd.DataFrame({"g": ["x", None], "flag": [True, False]})
    assert bias.slice_values(frame, "g").tolist() == ["x", "unknown"]
    assert bias.slice_values(frame, "flag").tolist() == ["True", "False"]
