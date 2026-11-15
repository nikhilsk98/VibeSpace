"""Part 3 end to end: ``run-all`` on fixtures (Part 1 -> Part 2 tiny budget -> Part 3) and the registry."""

from __future__ import annotations

import json

import pytest
import yaml
from mlflow import MlflowClient

from vibespace_model import tracking
from vibespace_model.cli import EXIT_BIAS_GATE, EXIT_OK, main
from vibespace_model.config import load_config
from vibespace_model.select import load_bundle

pytestmark = pytest.mark.integration


def _run_all(tree) -> int:
    return main(["run-all", "--config", str(tree["config"]), "--log-level", "WARNING"])


def test_run_all_registers_champion_candidate(tree) -> None:
    code = _run_all(tree)
    cfg = load_config(tree["config"])
    summary = json.loads((cfg.reports_subdir("part3") / "part3_summary.json").read_text())
    final = summary["final"]
    assert code == (EXIT_OK if final["bias_gate"]["passed"] else EXIT_BIAS_GATE)

    # final selection considered the tuned families and both mitigations
    kinds = {e["kind"] for e in summary["entries"]}
    assert {"reweighted", "resampled"} <= kinds
    assert summary["decision"]["name"] == final["name"]

    out = cfg.reports_subdir("part3")
    for name in (
        "bias_report.md",
        "explain_summary.json",
        "grouped_importance.png",
        "mitigation_comparison.csv",
        "slices_val_final.csv",
        "slices_test_final.csv",
        "slice_mae_val_final.png",
        "slice_mae_test_final.png",
    ):
        assert (out / name).is_file(), name
    report = (out / "bias_report.md").read_text()
    assert "## Final decision" in report and final["name"] in report

    explain = json.loads((out / "explain_summary.json").read_text())
    assert 1 <= len(explain["models"]) <= 2
    for model in explain["models"].values():
        assert set(model["shap_share"]) == {
            "plot_embedding",
            "genres",
            "runtime",
            "year",
            "language",
            "popularity",
            "flags",
        }
        assert sum(model["shap_share"].values()) == pytest.approx(1.0)

    _, _, card = load_bundle(cfg.artifacts_subdir("part3") / "final_model")
    for key in (
        "validation_gate",
        "bias_gate",
        "bias_gate_test",
        "slice_summary_val",
        "shap_top_groups",
        "data_version",
        "split_hash",
        "params",
        "metrics",
        "thresholds",
        "selection",
    ):
        assert key in card, key
    assert card["stage"] == "part3-final"
    assert "bias_gate" in card["thresholds"] and "validation_gate" in card["thresholds"]
    assert card["metrics"]["test"]["overall"]["n"] > 0

    tracking.setup_tracking(cfg)
    client = MlflowClient()
    champion = client.get_model_version_by_alias(cfg.registered_model_name, "champion-candidate")
    candidate = client.get_model_version_by_alias(cfg.registered_model_name, "candidate")
    assert int(champion.version) == 3 and int(candidate.version) == 2  # Part 1, Part 2, Part 3 versions
    assert main(["select", "--config", str(tree["config"]), "--log-level", "WARNING"]) == code


def test_none_pass_bias_gate_is_flagged(tree) -> None:
    bias_file = tree["configs"] / "bias.yaml"
    raw = yaml.safe_load(bias_file.read_text())
    raw["max_mae_ratio"] = 1.0  # no slice can be strictly better than average everywhere
    raw["min_gated_slice_size"] = 5
    bias_file.write_text(yaml.safe_dump(raw))
    assert _run_all(tree) == EXIT_BIAS_GATE
    cfg = load_config(tree["config"])
    summary = json.loads((cfg.reports_subdir("part3") / "part3_summary.json").read_text())
    assert summary["decision"]["flagged"] is True
    assert "smallest slice gap" in summary["decision"]["reason"]
    assert main(
        ["select", "--config", str(tree["config"]), "--stage", "part3", "--log-level", "WARNING"]
    ) == (EXIT_BIAS_GATE)
