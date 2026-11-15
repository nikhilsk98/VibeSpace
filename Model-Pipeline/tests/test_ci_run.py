"""Part 5 ci-run: sequencing, exit codes, gate failures block the registry, overrides, dry-run safety."""

from __future__ import annotations

import json
import subprocess

import pytest

from vibespace_model import registry as registry_mod
from vibespace_model.cli import main
from vibespace_model.config import deep_merge, load_config
from vibespace_model.errors import ConfigError


def _run(tree, out, *extra: str) -> int:
    return main(
        [
            "ci-run",
            "--config",
            str(tree["config"]),
            "--skip-hash-check",
            "--dry-run",
            "--out",
            str(out),
            "--log-level",
            "WARNING",
            *extra,
        ]
    )


def _state(out) -> dict:
    return json.loads((out / "state.json").read_text())


def _registry_versions(tree) -> list:
    cfg = load_config(tree["config"])
    folder = cfg.root / "artifacts" / "registry-dry-run" / "artifacts"
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


@pytest.fixture
def no_gcloud(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("ci-run --dry-run must never call gcloud")

    monkeypatch.setattr(registry_mod, "_run_gcloud", refuse)
    return refuse


def test_full_sequence_promotes_then_compares(tree, tmp_path, no_gcloud) -> None:
    out = tmp_path / "ci-out"
    assert _run(tree, out) == 0
    state = _state(out)
    assert state["event"] == "promoted"
    assert [state["steps"][s]["status"] for s in ("train", "validate", "bias", "promote")] == ["passed"] * 4
    assert state["decision"]["label"] == "promoted" and state["champion_after"]["version"] == state["version"]
    assert state["registration"]["alias"] == "ci-candidate"
    for name in (
        "metrics.json",
        "model_card.json",
        "bias_report.md",
        "bias_slices_val.csv",
        "decision.json",
        "job_summary.md",
        "val_mae_per_dimension.png",
        "validation_gate.json",
        "bias_gate.json",
    ):
        assert (out / name).is_file(), name
    summary = (out / "job_summary.md").read_text()
    assert "Promotion: promoted" in summary and "| train | passed |" in summary and "dry run" in summary
    card = json.loads((out / "model_card.json").read_text())
    assert card["validation_gate"]["passed"] and card["bias_gate"]["passed"] and card["ci_family"] == "mlp"

    first = state["version"]
    assert _run(tree, out) == 0  # same params, same split: within tolerance of the champion
    again = _state(out)
    assert again["champion_before"]["version"] == first
    assert again["decision"]["candidate_val_mae"] == pytest.approx(again["decision"]["champion_val_mae"])
    assert again["event"] in ("promoted", "not_promoted")
    assert len(_registry_versions(tree)) == 2


def test_validation_failure_blocks_registry(tree, tmp_path, monkeypatch, no_gcloud) -> None:
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", json.dumps({"gate": {"min_r2": 0.999}}))
    out = tmp_path / "ci-out"
    assert _run(tree, out) == 1
    state = _state(out)
    assert state["event"] == "validation_failed"
    assert state["steps"]["validate"]["status"] == "failed"
    assert "bias" not in state["steps"] and "promote" not in state["steps"]
    assert _registry_versions(tree) == []
    assert "Validation Gate: FAILED" in (out / "job_summary.md").read_text()


def test_bias_failure_blocks_registry(tree, tmp_path, monkeypatch, no_gcloud) -> None:
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", json.dumps({"bias": {"max_mae_ratio": 0.5}}))
    out = tmp_path / "ci-out"
    assert _run(tree, out) == 2
    state = _state(out)
    assert state["event"] == "bias_failed" and state["steps"]["bias"]["status"] == "failed"
    assert "promote" not in state["steps"]
    assert _registry_versions(tree) == []
    assert "failed" in (out / "bias_report.md").read_text()


def test_steps_one_by_one_and_promote_guard(tree, tmp_path, no_gcloud) -> None:
    out = tmp_path / "ci-out"
    assert _run(tree, out, "--ci-step", "validate") == 3  # train has not run in this folder
    assert _state(out)["event"] == "pipeline_error"
    assert _run(tree, out, "--ci-step", "train") == 0
    assert _state(out)["event"] == "training_completed"
    assert _run(tree, out, "--ci-step", "promote") == 1  # gates not evaluated yet: blocked
    assert _state(out)["steps"]["promote"]["status"] == "blocked"
    assert _registry_versions(tree) == []
    assert _run(tree, out, "--ci-step", "validate") == 0
    assert _run(tree, out, "--ci-step", "promote") == 2  # bias gate still missing: blocked
    assert _run(tree, out, "--ci-step", "bias") == 0
    assert _run(tree, out, "--ci-step", "promote") == 0
    assert _state(out)["event"] == "promoted" and len(_registry_versions(tree)) == 1


def test_tuned_params_override(tree, tmp_path, no_gcloud) -> None:
    params = {
        "mlp": {
            "hidden_sizes": [16],
            "dropout": 0.0,
            "learning_rate": 0.02,
            "max_epochs": 10,
            "patience": 3,
            "num_threads": 1,
        }
    }
    file = tmp_path / "best_params.json"
    file.write_text(json.dumps(params))
    out = tmp_path / "ci-out"
    assert _run(tree, out, "--ci-step", "train", "--tuned-params", str(file)) == 0
    assert _state(out)["params"] == params["mlp"]
    file.write_text(json.dumps({"ridge": {"alpha": 1.0}}))
    assert _run(tree, out, "--ci-step", "train", "--tuned-params", str(file)) == 3


def test_config_errors_are_pipeline_errors(tree, tmp_path, monkeypatch, no_gcloud) -> None:
    out = tmp_path / "ci-out"
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", json.dumps({"ci": {"family": ""}}))
    assert _run(tree, out) == 3 and "ci.family" in _state(out)["error"]
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", json.dumps({"ci": {"family": "mlp", "params": None}}))
    cfg = load_config(tree["config"])
    assert cfg.ci["params"] is None
    monkeypatch.setenv(
        "VIBESPACE_MODEL_OVERRIDES", json.dumps({"ci": {"expected_split_hash": "not-the-hash"}})
    )
    assert (
        main(
            [
                "ci-run",
                "--config",
                str(tree["config"]),
                "--skip-hash-check",
                "--dry-run",
                "--out",
                str(out),
                "--log-level",
                "WARNING",
            ]
        )
        == 3
    )
    assert "expected_split_hash" in _state(out)["error"]
    assert "Pipeline error" in (out / "job_summary.md").read_text()


def test_missing_params_without_override(tree, tmp_path, monkeypatch) -> None:
    raw = tree["config"].read_text().replace("  params:\n", "  unused_params:\n", 1)
    tree["config"].write_text(raw)
    assert "params" not in load_config(tree["config"]).ci
    assert _run(tree, tmp_path / "o", "--ci-step", "train") == 3


def test_env_overrides(tree, monkeypatch) -> None:
    monkeypatch.setenv(
        "VIBESPACE_MODEL_OVERRIDES",
        json.dumps(
            {"gate": {"min_r2": 0.5}, "tuning": {"total_time_cap_minutes": 1}, "bias": {"max_mae_ratio": 2.0}}
        ),
    )
    cfg = load_config(tree["config"])
    assert cfg.gate == {"min_mae_improvement": 0.15, "min_r2": 0.5}
    assert cfg.tuning["total_time_cap_minutes"] == 1 and cfg.tuning["models"]
    assert cfg.bias["max_mae_ratio"] == 2.0 and cfg.bias["min_gated_slice_size"] == 10
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", "{not json")
    with pytest.raises(ConfigError, match="valid JSON"):
        load_config(tree["config"])
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", "[1, 2]")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(tree["config"])
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}, "d": 4}) == {"a": {"b": 3, "c": 2}, "d": 4}


def test_gcp_mode_uses_gcloud(tree, tmp_path, monkeypatch) -> None:
    """Without --dry-run the promote step goes to the GCP backend (here a fake gcloud that always fails)."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "PERMISSION_DENIED")

    monkeypatch.setattr(registry_mod, "_run_gcloud", fake_run)
    out = tmp_path / "ci-out"
    code = main(
        [
            "ci-run",
            "--config",
            str(tree["config"]),
            "--skip-hash-check",
            "--out",
            str(out),
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 3 and _state(out)["event"] == "pipeline_error"
    assert calls and calls[0][0] == "gcloud" and "--project=fixture-project" in calls[0]


def test_not_promoted_is_visible_but_successful(tree, tmp_path, monkeypatch, no_gcloud, capsys) -> None:
    from ci import notify

    out = tmp_path / "ci-out"
    assert _run(tree, out) == 0  # first run becomes champion
    champion = _state(out)["version"]
    # the candidate must now beat the champion by 0.01 MAE: an identical retrain is not promoted
    monkeypatch.setenv("VIBESPACE_MODEL_OVERRIDES", json.dumps({"registry": {"promotion_tolerance": -0.01}}))
    assert _run(tree, out) == 0  # exit code stays 0
    state = _state(out)
    assert state["event"] == "not_promoted"
    assert state["decision"]["label"] == "not promoted (rollback)"
    assert state["champion_after"]["version"] == champion
    summary = (out / "job_summary.md").read_text()
    assert summary.startswith(
        f"> **NOT PROMOTED: champion unchanged (rollback): `{champion}` stays champion**"
    )
    assert "more than the tolerance -0.0100" in summary.split("## VibeSpace")[0]  # reasons at the top
    capsys.readouterr()
    assert notify.main(["--state", str(out / "state.json")]) == 0
    message = capsys.readouterr().out
    assert "NOT PROMOTED: champion unchanged (rollback)" in message and "more than the tolerance" in message
    assert notify.annotation(state).startswith("::warning title=NOT PROMOTED")


def test_headline_for_each_event() -> None:
    from vibespace_model.ci_run import headline

    assert headline({"event": "promoted", "version": "v2"}).startswith("PROMOTED: `v2`")
    assert headline({"event": "validation_failed"}).startswith("VALIDATION GATE FAILED")
    assert headline({"event": "bias_failed"}).startswith("BIAS GATE FAILED")
    assert headline({"event": "pipeline_error"}).startswith("PIPELINE ERROR")
    assert headline({"event": "training_completed"}) is None
