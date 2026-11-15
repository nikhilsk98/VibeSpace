"""Part 1 end to end on fixtures: load -> features -> split -> train all -> evaluate -> select -> register."""

from __future__ import annotations

import json

import pytest
from mlflow import MlflowClient

from vibespace_model import tracking
from vibespace_model.models import CANDIDATES
from vibespace_model.select import load_bundle
from vibespace_model.train import prepare, run_evaluate

pytestmark = pytest.mark.integration


def test_part1_outputs(part1_tree) -> None:
    cfg, summary = part1_tree["cfg"], part1_tree["summary"]
    reports, artifacts = cfg.reports_subdir("part1"), cfg.artifacts_subdir("part1")

    assert set(summary["results"]) == set(CANDIDATES)
    assert summary["selected"] in CANDIDATES and summary["selected"] != "baseline"
    assert summary["validation_gate"]["passed"]
    assert (
        summary["split_counts"]["train"] + summary["split_counts"]["val"] + summary["split_counts"]["test"]
        == 232
    )
    assert summary["join_report"]["dropped_no_embedding"] == 3
    for name in (
        "part1_summary.json",
        "model_comparison.csv",
        "per_dimension_val_mae.csv",
        "val_mae_by_model.png",
        "val_mae_per_dimension.png",
        "pred_vs_true_val.png",
        "split_slice_counts.csv",
    ):
        assert (reports / name).is_file(), name
    for name in CANDIDATES:
        assert (artifacts / "models" / name / "params.json").is_file()

    model, builder, card = load_bundle(artifacts / "selected")
    assert card["model_type"] == summary["selected"]
    assert card["split_hash"] == summary["split_hash"]
    assert card["validation_gate"]["passed"]
    assert card["data_version"]["hash_check"] == "passed"
    assert builder.feature_names[0] == "emb_000"

    tracking.setup_tracking(cfg)
    alias = MlflowClient().get_model_version_by_alias(cfg.registered_model_name, "candidate")
    assert str(alias.version) == summary["registration"]["version"]
    runs = MlflowClient().search_runs(
        [MlflowClient().get_experiment_by_name(cfg.experiment_name).experiment_id]
    )
    assert {"part1-train", *CANDIDATES} <= {r.info.run_name for r in runs}


def test_split_is_reused_and_evaluate_matches(part1_tree) -> None:
    cfg, summary = part1_tree["cfg"], part1_tree["summary"]
    prep = prepare(cfg, create_split=False)
    assert prep.split.split_hash == summary["split_hash"]
    evaluated = run_evaluate(cfg)
    for name, res in evaluated["results"].items():
        assert res["val"]["overall"]["mae"] == pytest.approx(
            summary["results"][name]["val"]["overall"]["mae"]
        )
    stored = json.loads((cfg.reports_subdir("part1") / "part1_summary.json").read_text())
    assert stored["registration"]["alias"] == "candidate"
