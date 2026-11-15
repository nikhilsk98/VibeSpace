"""Part 4 promotion rule, promote / rollback flows (dry-run registry), MLflow alias mirroring, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow import MlflowClient

from vibespace_model import registry as registry_mod
from vibespace_model import tracking
from vibespace_model.cli import EXIT_ERROR, EXIT_OK, main
from vibespace_model.config import load_config
from vibespace_model.models import create_model
from vibespace_model.promote import (
    Evaluation,
    build_registry,
    decide,
    local_registration,
    mirror_champion_alias,
    promote,
    rollback,
)
from vibespace_model.registry import ChecksumError, LocalBackend, RegistryError
from vibespace_model.select import GateResult, build_model_card, save_bundle, validation_gate
from vibespace_model.train import prepare, run_part1


def _eval(mae: float, val_ok: bool = True, bias_ok: bool = True) -> Evaluation:
    overall = {"mae": mae, "rmse": mae, "r2": 0.5, "spearman": 0.7}
    return Evaluation(
        "x",
        {"overall": overall},
        {"overall": overall},
        GateResult("validation", val_ok, ["v"]),
        GateResult("bias", bias_ok, ["b"]),
        1.1,
        pd.DataFrame(),
    )


@pytest.mark.parametrize(
    ("candidate", "champion", "promoted", "text"),
    [
        (_eval(0.150), _eval(0.155), True, "better than champion"),
        (_eval(0.1565), _eval(0.155), True, "within tolerance"),
        (_eval(0.158), _eval(0.155), False, "more than the tolerance"),
        (_eval(0.140, val_ok=False), _eval(0.155), False, "validation gate failed"),
        (_eval(0.140, bias_ok=False), _eval(0.155), False, "bias gate failed"),
        (_eval(0.150), None, True, "no champion yet"),
        (_eval(0.150, bias_ok=False), None, False, "none yet"),
    ],
)
def test_promotion_rule(candidate, champion, promoted, text) -> None:
    decision = decide(candidate, champion, "v-champ" if champion else None, 0.002)
    assert decision.promote is promoted
    assert text in " ".join(decision.reasons)
    assert decision.label == ("promoted" if promoted else "not promoted (rollback)")
    assert decision.to_dict()["tolerance"] == 0.002


@pytest.fixture
def trained(tree):
    """Fixture tree after Part 1, bias gate relaxed so the fixture model can pass it."""
    cfg = load_config(tree["config"])
    run_part1(cfg)
    cfg.bias["max_mae_ratio"] = 10.0
    prep = prepare(cfg, create_split=False)
    return cfg, prep


def _baseline_bundle(cfg, prep, path: Path) -> Path:
    """A deliberately worse candidate: the mean baseline, labelled as a test."""
    model = create_model("baseline").fit(prep.X["train"], prep.Y["train"])
    gate = validation_gate({"mae": 1.0, "r2": 0.0}, {"mae": 1.0}, cfg.gate)
    card = build_model_card(
        model=model,
        stage="test-candidate",
        data_version=prep.dataset.data_version,
        split_hash=prep.split.split_hash,
        metrics={},
        gate_cfg=cfg.gate,
        validation_gate_result=gate,
        extra={"targets": cfg.targets, "test_candidate": True},
    )
    return save_bundle(path, model, prep.builder, card)


def test_dry_run_flow_promote_reject_rollback(trained, tmp_path, monkeypatch) -> None:
    """Integration (dry-run): train -> bundle -> promote -> worse candidate not promoted -> rollback."""
    cfg, prep = trained

    def no_gcloud(*args, **kwargs):
        raise AssertionError("dry-run must never call gcloud")

    monkeypatch.setattr(registry_mod, "_run_gcloud", no_gcloud)
    registry = build_registry(cfg, dry_run=True)
    assert isinstance(registry.backend, LocalBackend)
    candidate_dir = cfg.artifacts_subdir("part1") / "selected"

    first = promote(
        cfg,
        registry,
        candidate_dir,
        prep,
        extra=local_registration(cfg, "candidate"),
        version="v20260101-000000-aaaaaaa",
    )
    assert first["decision"]["promote"] is True
    assert "no champion yet" in first["decision"]["reasons"][0]
    assert first["champion_after"]["version"] == "v20260101-000000-aaaaaaa"
    assert first["local_mlflow_alias_version"] == "1"

    worse_dir = _baseline_bundle(cfg, prep, tmp_path / "worse")
    second = promote(
        cfg, registry, worse_dir, prep, extra={"test_candidate": True}, version="v20260101-000001-test000"
    )
    assert second["decision"]["promote"] is False
    assert second["decision"]["label"] == "not promoted (rollback)"
    assert registry.get_champion()["version"] == "v20260101-000000-aaaaaaa"  # champion unchanged
    meta = registry.version_meta("v20260101-000001-test000")
    assert meta["status"] == "not promoted (rollback)" and meta["decision"]["reasons"]
    assert second["champion_evaluation"]["val"]["mae"] == pytest.approx(first["candidate"]["val"]["mae"])

    registry.set_champion("v20260101-000001-test000", "temporary, rollback test", event="test")
    back = rollback(cfg, registry)
    assert back["champion"]["version"] == "v20260101-000000-aaaaaaa"
    assert back["champion"]["event"] == "rollback"
    assert back["local_mlflow_alias_version"] == "1"
    assert [h["version"] for h in registry.history()] == [
        "v20260101-000000-aaaaaaa",
        "v20260101-000001-test000",
        "v20260101-000000-aaaaaaa",
    ]
    assert registry.version_meta("v20260101-000001-test000")["status"] == "rolled back"

    specific = rollback(cfg, registry, to_version="v20260101-000001-test000")
    assert specific["champion"]["version"] == "v20260101-000001-test000"
    with pytest.raises(RegistryError, match="already the champion"):
        rollback(cfg, registry, to_version="v20260101-000001-test000")
    with pytest.raises(RegistryError, match="not found"):
        rollback(cfg, registry, to_version="v-missing")

    archive = (
        registry.backend.root / "artifacts" / "v20260101-000000-aaaaaaa" / "v20260101-000000-aaaaaaa.tar.gz"
    )
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ChecksumError):
        rollback(cfg, registry, to_version="v20260101-000000-aaaaaaa")
    assert registry.get_champion()["version"] == "v20260101-000001-test000"  # unchanged after failure


def test_rollback_without_history(trained) -> None:
    cfg, _ = trained
    with pytest.raises(RegistryError, match="nothing to roll back"):
        rollback(cfg, build_registry(cfg, dry_run=True))


def test_mlflow_alias_mirroring(trained) -> None:
    cfg, _ = trained
    info = local_registration(cfg, "candidate")
    assert info["local_registry_version"] == "1" and info["mlflow_run_id"]
    assert local_registration(cfg, "does-not-exist") == {}
    assert mirror_champion_alias(cfg, {"version": "v1", **info}) == "1"
    tracking.setup_tracking(cfg)
    client = MlflowClient()
    assert client.get_model_version_by_alias(cfg.registered_model_name, "champion").version == 1
    assert client.get_model_version(cfg.registered_model_name, "1").tags["gcp_registry_version"] == "v1"
    assert mirror_champion_alias(cfg, {"version": "v1"}) is None
    assert mirror_champion_alias(cfg, {"version": "v1", "local_registry_version": "99"}) is None
    assert (
        mirror_champion_alias(cfg, {"version": "v1", "local_registry_version": "1", "mlflow_run_id": "other"})
        is None
    )


def test_gcp_registry_is_built_from_config(cfg) -> None:
    registry = build_registry(cfg, dry_run=False)
    assert (
        registry.backend.repository == "fixture-models"
        and registry.backend.bucket == "fixture-registry-bucket"
    )
    assert "fixture-models" in registry.backend.description


def test_registry_cli_commands(tree, capsys) -> None:
    args = ["--config", str(tree["config"]), "--dry-run", "--log-level", "WARNING"]
    assert main(["champion", *args]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["champion"] is None
    assert main(["pull", *args]) == EXIT_ERROR  # no champion yet
    assert main(["promote", *args]) == EXIT_ERROR  # no Part 3 bundle yet
    assert main(["train", "--config", str(tree["config"]), "--log-level", "WARNING"]) == EXIT_OK
    capsys.readouterr()
    cfg = load_config(tree["config"])
    bundle = str(cfg.artifacts_subdir("part1") / "selected")
    assert main(["push", *args, "--bundle", bundle, "--version", "v-pushed"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["version"] == "v-pushed"
    assert main(["promote", *args, "--bundle", bundle, "--local-alias", "candidate"]) == EXIT_OK
    promoted = json.loads(capsys.readouterr().out)
    assert promoted["decision"]["label"] in ("promoted", "not promoted (rollback)")
    assert main(["list-versions", *args]) == EXIT_OK
    listed = json.loads(capsys.readouterr().out)["versions"]
    assert {v["version"] for v in listed} >= {"v-pushed", promoted["version"]}
    assert main(["pull", *args, "--version", "v-pushed", "--dest", str(tree["root"] / "pulled")]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["model_type"] in ("ridge", "lightgbm", "mlp")
    assert main(["rollback", *args, "--to-version", "v-missing"]) == EXIT_ERROR
