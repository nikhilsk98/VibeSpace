"""Part 2 end to end on fixtures: tune (tiny budget) -> retrain best -> evaluate -> select -> register."""

from __future__ import annotations

import json

import pytest
from mlflow import MlflowClient

from vibespace_model import tracking
from vibespace_model.cli import EXIT_OK, main
from vibespace_model.config import load_config
from vibespace_model.errors import VibeSpaceModelError
from vibespace_model.train import run_part1
from vibespace_model.workflows import run_sensitivity, run_tune

pytestmark = pytest.mark.integration


def test_tune_requires_part1(cfg) -> None:
    with pytest.raises(VibeSpaceModelError, match="run the `train` command first"):
        run_tune(cfg)


def test_part2_end_to_end(tree) -> None:
    cfg = load_config(tree["config"])
    part1 = run_part1(cfg)
    summary = run_tune(cfg)
    out = cfg.reports_subdir("part2")

    assert summary["split_hash"] == part1["split_hash"]
    assert summary["test_split_used_for_tuning"] is False
    assert set(summary["studies"]) == {"ridge", "lightgbm", "mlp"}
    assert summary["studies"]["lightgbm"]["n_trials"] == 3
    assert summary["studies"]["mlp"]["n_trials"] == 4
    assert set(summary["best_params"]) == {"ridge", "lightgbm", "mlp"}
    assert {"ridge_tuned", "lightgbm_tuned", "mlp_tuned", "baseline_default"} <= set(summary["variants"])
    assert summary["selected"] in summary["variants"]
    assert summary["validation_gate"]["passed"]
    for name in (
        "best_params.json",
        "default_vs_tuned.csv",
        "default_vs_tuned.png",
        "final_candidate_val_vs_test.csv",
        "final_candidate_per_dimension_val_mae.png",
        "optuna_lightgbm_optimization_history.png",
        "optuna_lightgbm_param_importances.png",
        "optuna_mlp_optimization_history.png",
        "optuna_ridge_parallel_coordinate.png",
        "variants_val_mae.png",
    ):
        assert (out / name).is_file(), name

    tracking.setup_tracking(cfg)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")
    assert len(versions) == 2  # Part 1 version kept, Part 2 version added
    assert str(client.get_model_version_by_alias(cfg.registered_model_name, "candidate").version) == "2"

    sens = run_sensitivity(cfg)
    assert set(sens["importances"]) == {"ridge", "lightgbm", "mlp"}
    if sens["final_family"] in ("ridge", "lightgbm", "mlp"):
        assert len(sens["swept_params"]) <= 2
        assert (out / "sensitivity" / f"sweeps_{sens['final_family']}.png").is_file()

    # resume: a second tune run finds the finished studies and adds no trials
    again = run_tune(cfg)
    assert again["studies"]["lightgbm"]["n_trials"] == 3

    # reuse mode (what the Docker regression run uses): no search, same retrained models
    assert (
        main(
            [
                "tune",
                "--config",
                str(tree["config"]),
                "--tuned-params",
                str(out / "best_params.json"),
                "--log-level",
                "WARNING",
            ]
        )
        == EXIT_OK
    )
    reused = json.loads((out / "part2_summary.json").read_text())
    assert reused["mode"] == "reuse" and reused["studies"] == {}
    assert reused["selected"] == summary["selected"]
    assert reused["variants"][summary["selected"]]["val"]["overall"]["mae"] == pytest.approx(
        summary["variants"][summary["selected"]]["val"]["overall"]["mae"]
    )
    skipped = run_sensitivity(cfg)
    assert skipped["skipped"] is True
