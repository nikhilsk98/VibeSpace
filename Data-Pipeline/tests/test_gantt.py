"""Tests for scripts.gantt (fake Airflow API)."""

from __future__ import annotations

import json

import pytest

from scripts import gantt
from tests.helpers import FakeResponse, FakeSession

TIS = [
    {"task_id": "a", "state": "success", "start_date": "2024-01-01T00:00:00+00:00",
     "end_date": "2024-01-01T00:00:10+00:00", "duration": 10, "try_number": 1},
    {"task_id": "b", "state": "success", "start_date": "2024-01-01T00:00:10+00:00",
     "end_date": "2024-01-01T00:00:40+00:00", "duration": 30, "try_number": 1},
    {"task_id": "c", "state": "failed", "start_date": "2024-01-01T00:00:10Z",
     "end_date": "2024-01-01T00:00:15Z", "duration": 5, "try_number": 2},
    {"task_id": "d", "state": "upstream_failed", "start_date": None, "end_date": None},
]


def test_fetch_and_summarise():
    session = FakeSession(responses=[FakeResponse(200, {"task_instances": TIS})])
    tis = gantt.fetch_task_instances("http://af:8080/", "dag", "run1", ("u", "p"), session=session)
    assert session.calls[0]["url"] == "http://af:8080/api/v1/dags/dag/dagRuns/run1/taskInstances"
    summary = gantt.summarise(tis)
    assert summary["wall_seconds"] == 40 and summary["sum_task_seconds"] == 45
    assert list(summary["tasks"]) == ["b", "a", "c"]
    assert gantt.summarise([])["wall_seconds"] == 0.0


def test_fetch_http_error():
    with pytest.raises(RuntimeError, match="401"):
        gantt.fetch_task_instances("http://af", "dag", "r", ("u", "p"), session=FakeSession(responses=[
            FakeResponse(401)]))


def test_plot_gantt(tmp_path):
    out = gantt.plot_gantt(TIS, "demo", tmp_path / "g.png")
    assert out.stat().st_size > 1000
    with pytest.raises(ValueError):
        gantt.plot_gantt(TIS[3:], "empty", tmp_path / "e.png")


def test_cli(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gantt, "fetch_task_instances", lambda *a, **k: TIS)
    monkeypatch.setattr(gantt, "screenshot_ui_gantt", lambda *a, **k: False)
    code = gantt.main(["--dag-id", "d", "--run-id", "r", "--out", str(tmp_path / "x.png"),
                       "--json", str(tmp_path / "x.json"), "--ui-screenshot", str(tmp_path / "ui.png")])
    assert code == 0 and json.loads((tmp_path / "x.json").read_text())["wall_seconds"] == 40
    assert "not available" in capsys.readouterr().out
