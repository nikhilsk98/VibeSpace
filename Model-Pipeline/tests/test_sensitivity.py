"""Part 2 hyperparameter sensitivity: importance, sweep values, sweeps on fixtures, plots."""

from __future__ import annotations

import optuna
import pytest

from vibespace_model.sensitivity import (
    param_importances,
    plot_importances,
    plot_sweeps,
    run_sweeps,
    sweep_summary,
    sweep_values,
)
from vibespace_model.train import prepare
from vibespace_model.tuning import ParamSpec, TuningData, parse_search_space

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_sweep_values() -> None:
    log_spec = ParamSpec("lr", "float", 0.001, 0.1, log=True)
    values = sweep_values(log_spec, 0.02, 3)
    assert values == pytest.approx([0.001, 0.01, 0.02, 0.1])
    int_spec = ParamSpec("leaves", "int", 4, 16)
    assert sweep_values(int_spec, 7, 3) == [4, 7, 10, 16]
    cat = ParamSpec("sizes", "categorical", choices=("16", "32"))
    assert sweep_values(cat, "16", 5) == ["16", "32"]


def test_param_importances() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    assert param_importances(study, seed=0) == {}
    study.optimize(
        lambda t: (t.suggest_float("x", -3, 3) ** 2) + 0.01 * t.suggest_float("y", -3, 3), n_trials=30
    )
    importances = param_importances(study, seed=0)
    assert set(importances) == {"x", "y"}
    assert importances["x"] > importances["y"]
    assert sum(importances.values()) == pytest.approx(1.0)
    assert set(param_importances(study, seed=0, evaluator="mdi")) == {"x", "y"}


def test_sweeps_shape_and_plots(part1_tree, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    data = TuningData.from_prepared(prepare(cfg, create_split=False))
    specs = parse_search_space(cfg.tuning["models"]["mlp"]["search_space"])
    fixed = {**cfg.model_params("mlp"), **cfg.tuning["models"]["mlp"]["fixed"]}
    best = {"hidden_sizes": "16", "learning_rate": 0.01, "dropout": 0.1}
    sweeps = run_sweeps(
        "mlp",
        best,
        fixed,
        specs,
        ["learning_rate", "hidden_sizes"],
        data,
        cfg.random_seed,
        cfg.targets,
        points=3,
    )
    assert list(sweeps.columns) == ["param", "value", "val_mae", "is_best"]
    assert (sweeps["param"] == "learning_rate").sum() == 4  # 3 grid points + best value
    assert (sweeps["param"] == "hidden_sizes").sum() == 2
    assert sweeps["is_best"].sum() == 2
    assert sweeps["val_mae"].between(0, 1).all()
    summary = sweep_summary(sweeps)
    assert set(summary["param"]) == {"learning_rate", "hidden_sizes"}
    assert (summary["spread"] >= 0).all()
    assert plot_sweeps(sweeps, specs, tmp_path / "sweeps.png", "t").stat().st_size > 0
    imp = {"mlp": {"learning_rate": 0.7, "dropout": 0.3}, "ridge": {}}
    assert plot_importances(imp, tmp_path / "imp.png").stat().st_size > 0
