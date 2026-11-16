"""Part 2 tuning: search spaces, objective, determinism, pruning, MLflow trial logs, resume, test guard."""

from __future__ import annotations

import math

import numpy as np
import optuna
import pytest
from mlflow import MlflowClient

from vibespace_model import tracking
from vibespace_model.errors import ConfigError, TestSplitAccessError
from vibespace_model.train import prepare
from vibespace_model.tuning import (
    ParamSpec,
    TuningData,
    best_params,
    finished_trials,
    load_best_params,
    make_objective,
    make_pruner,
    parse_search_space,
    resolve_params,
    run_study,
    run_tuning,
    save_best_params,
    study_summary,
    suggest_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture(scope="module")
def data(part1_tree):
    return TuningData.from_prepared(prepare(part1_tree["cfg"], create_split=False))


def _storage(tmp_path, name="optuna.db"):
    return f"sqlite:///{tmp_path / name}"


def test_parse_valid_search_space() -> None:
    specs = parse_search_space(
        {
            "alpha": {"type": "float", "low": 0.01, "high": 10, "log": True},
            "leaves": {"type": "int", "low": 4, "high": 64},
            "sizes": {"type": "categorical", "choices": ["16", "32-16"]},
        }
    )
    assert specs["alpha"] == ParamSpec("alpha", "float", 0.01, 10.0, True, None)
    assert specs["sizes"].choices == ("16", "32-16")
    trial = optuna.trial.FixedTrial({"alpha": 0.5, "leaves": 8, "sizes": "32-16"})
    assert suggest_params(trial, specs) == {"alpha": 0.5, "leaves": 8, "sizes": "32-16"}


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({}, "empty"),
        ({"a": {"type": "uniform", "low": 0, "high": 1}}, "type must be"),
        ({"a": {"type": "categorical", "choices": []}}, "non-empty"),
        ({"a": {"type": "float", "low": 0}}, "'low' and 'high'"),
        ({"a": {"type": "float", "low": 2, "high": 1}}, "must be below"),
        ({"a": {"type": "float", "low": 0, "high": 1, "log": True}}, "low > 0"),
        ({"a": {"type": "float", "low": 1, "high": 9, "log": True, "step": 1}}, "step"),
    ],
)
def test_invalid_search_spaces(space, message) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_search_space(space)


def test_repository_search_spaces_parse() -> None:
    from pathlib import Path

    import yaml

    tuning = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "tuning.yaml").read_text())
    for name, model_cfg in tuning["models"].items():
        assert parse_search_space(model_cfg["search_space"]), name


def test_resolve_params_parses_hidden_sizes() -> None:
    assert resolve_params("mlp", {"max_epochs": 5}, {"hidden_sizes": "64-32"}) == {
        "max_epochs": 5,
        "hidden_sizes": [64, 32],
    }
    assert resolve_params("ridge", {}, {"alpha": 1.0}) == {"alpha": 1.0}


def test_tuning_data_rejects_test_movies(data) -> None:
    with pytest.raises(TestSplitAccessError):
        TuningData(
            data.X_train,
            data.Y_train,
            data.X_val,
            data.Y_val,
            np.concatenate([data.train_ids, data.split.ids("test")[:2]]),
            data.val_ids,
            data.split,
        )


def test_objective_returns_finite_mae(part1_tree, data) -> None:
    cfg = part1_tree["cfg"]
    specs = parse_search_space(cfg.tuning["models"]["ridge"]["search_space"])
    objective = make_objective("ridge", specs, {}, data, cfg.random_seed, cfg.targets, log_to_mlflow=False)
    value = objective(optuna.trial.FixedTrial({"alpha": 1.0}))
    assert math.isfinite(value) and 0 < value < 0.5


class _PruneAlways(optuna.trial.FixedTrial):
    def should_prune(self) -> bool:
        return True


def test_pruning_raises_and_is_logged(part1_tree, data) -> None:
    cfg = part1_tree["cfg"]
    tracking.setup_tracking(cfg)
    specs = parse_search_space(cfg.tuning["models"]["mlp"]["search_space"])
    fixed = {**cfg.model_params("mlp"), **cfg.tuning["models"]["mlp"]["fixed"]}
    objective = make_objective("mlp", specs, fixed, data, cfg.random_seed, cfg.targets, log_to_mlflow=True)
    with tracking.run("prune-test"), pytest.raises(optuna.TrialPruned):
        objective(_PruneAlways({"hidden_sizes": "16", "learning_rate": 0.01, "dropout": 0.1}))
    runs = MlflowClient().search_runs(
        [MlflowClient().get_experiment_by_name(cfg.experiment_name).experiment_id],
        filter_string="tags.trial_state = 'pruned'",
    )
    assert runs and runs[0].data.metrics["pruned"] == 1.0


def test_tiny_study_is_deterministic(part1_tree, data, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    a = run_study(
        cfg,
        "lightgbm",
        data,
        n_trials=3,
        timeout_seconds=None,
        log_to_mlflow=False,
        storage=_storage(tmp_path, "a.db"),
    )
    b = run_study(
        cfg,
        "lightgbm",
        data,
        n_trials=3,
        timeout_seconds=None,
        log_to_mlflow=False,
        storage=_storage(tmp_path, "b.db"),
    )
    assert [t.params for t in a.trials] == [t.params for t in b.trials]
    assert [t.value for t in a.trials] == [t.value for t in b.trials]
    params = best_params(a, "lightgbm")
    assert params["n_estimators"] == 40 and "num_leaves" in params
    summary = study_summary(a)
    assert summary["n_complete"] == 3 and summary["best_value"] == pytest.approx(a.best_value)


def test_study_resumes_from_sqlite(part1_tree, data, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    storage = _storage(tmp_path)
    first = run_study(
        cfg, "ridge", data, n_trials=2, timeout_seconds=None, log_to_mlflow=False, storage=storage
    )
    assert finished_trials(first) == 2
    resumed = run_study(
        cfg, "ridge", data, n_trials=3, timeout_seconds=None, log_to_mlflow=False, storage=storage
    )
    assert finished_trials(resumed) == 3  # only one new trial
    again = run_study(
        cfg, "ridge", data, n_trials=3, timeout_seconds=None, log_to_mlflow=False, storage=storage
    )
    assert len(again.trials) == 3


def test_trials_are_nested_mlflow_runs(part1_tree, data, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    tracking.setup_tracking(cfg)
    with tracking.run("tuning-log-test") as parent:
        run_study(
            cfg,
            "ridge",
            data,
            n_trials=2,
            timeout_seconds=None,
            log_to_mlflow=True,
            storage=_storage(tmp_path),
        )
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.experiment_name).experiment_id
    children = client.search_runs([exp], filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 2
    for child in children:
        assert child.info.run_name.startswith("ridge-trial-")
        assert {"val_mae", "duration_seconds", "pruned"} <= set(child.data.metrics)
        assert "alpha" in child.data.params
        assert child.data.tags["trial_state"] == "complete"


def test_run_tuning_respects_caps(part1_tree, data, tmp_path) -> None:
    cfg = part1_tree["cfg"]
    cfg.tuning["total_time_cap_minutes"] = 0
    try:
        studies = run_tuning(cfg, data, log_to_mlflow=False, storage=_storage(tmp_path))
    finally:
        cfg.tuning["total_time_cap_minutes"] = 10
    assert set(studies) == {"ridge", "lightgbm", "mlp"}
    assert all(len(s.trials) == 0 for s in studies.values())


def test_pruner_options() -> None:
    assert isinstance(make_pruner({"pruner": {"type": "none"}}), optuna.pruners.NopPruner)
    assert isinstance(make_pruner({}), optuna.pruners.MedianPruner)


def test_best_params_file_round_trip(tmp_path) -> None:
    save_best_params(tmp_path / "best.json", {"ridge": {"alpha": 2.0}})
    assert load_best_params(tmp_path / "best.json") == {"ridge": {"alpha": 2.0}}
    with pytest.raises(ConfigError):
        load_best_params(tmp_path / "missing.json")
