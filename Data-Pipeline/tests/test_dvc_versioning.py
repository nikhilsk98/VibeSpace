"""Tests for scripts.dvc_versioning (DVC commands are faked)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import dvc_versioning as dv
from scripts.exceptions import DVCError


class Recorder:
    """Fake ``subprocess.run`` that records commands and returns canned results."""

    def __init__(self, fail_on: str | None = None, status_out: str = "Data and pipelines are up to date.\n"):
        self.commands: list[list[str]] = []
        self.fail_on = fail_on
        self.status_out = status_out

    def __call__(self, cmd, cwd, capture_output, text, timeout):
        self.commands.append(cmd[1:])
        if self.fail_on and cmd[1] == self.fail_on:
            return SimpleNamespace(returncode=1, stdout="", stderr="ERROR: failed")
        out = self.status_out if cmd[1] == "status" else ""
        return SimpleNamespace(returncode=0, stdout=out, stderr="")


@pytest.fixture
def dvc_cfg(cfg, monkeypatch):
    (cfg.root / "dvc.yaml").write_text("stages: {}\n")
    monkeypatch.setattr(dv.shutil, "which", lambda name: "/usr/bin/dvc")
    return cfg


def test_commit_push_status(dvc_cfg):
    rec = Recorder()
    out = dv.version_outputs(replace(dvc_cfg, dvc_push=True), runner=rec)
    assert rec.commands == [["commit", "--force"], ["push"], ["status"]]
    assert out["pushed"] and out["status_clean"]


def test_push_can_be_disabled(dvc_cfg):
    rec = Recorder(status_out="")
    out = dv.version_outputs(dvc_cfg, push=False, runner=rec)
    assert ["push"] not in rec.commands and not out["pushed"] and out["status_clean"]


def test_failure_raises_dvc_error(dvc_cfg):
    with pytest.raises(DVCError, match="push"):
        dv.version_outputs(replace(dvc_cfg, dvc_push=True), runner=Recorder(fail_on="push"))


def test_missing_dvc_yaml(cfg, monkeypatch):
    monkeypatch.setattr(dv.shutil, "which", lambda name: "/usr/bin/dvc")
    with pytest.raises(DVCError, match="dvc.yaml"):
        dv.version_outputs(cfg, runner=Recorder())


def test_dvc_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(dv.shutil, "which", lambda name: None)
    with pytest.raises(DVCError, match="not installed"):
        dv.run_dvc(["status"], tmp_path)
