"""Tests for the task layer, the CLI runner and logging helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from scripts import logging_utils, run_pipeline, tasks
from scripts.exceptions import PipelineError


def test_build_config_params_and_conf_override():
    context = {"params": {"tmdb_stage": "genome", "sample_size": 50, "use_cache": True, "tmdb_max_workers": 4},
               "dag_run": SimpleNamespace(conf={"tmdb_stage": "rest", "sample_size": None})}
    cfg = tasks.build_config(context)
    assert cfg.tmdb_stage == "rest" and cfg.sample_size is None and cfg.tmdb_max_workers == 4


def test_build_config_defaults_from_env(monkeypatch):
    monkeypatch.setenv("SAMPLE_SIZE", "12")
    assert tasks.build_config({}).sample_size == 12
    assert tasks.build_config({"params": {"sample_size": "7"}}).sample_size == 7


def test_pipeline_steps_cover_dag_tasks():
    names = [n for n, _ in tasks.PIPELINE_STEPS]
    assert names[0] == "download_movielens" and names[-1] == "dvc_version" and len(set(names)) == 13


def test_task_wrappers_call_step_functions(monkeypatch):
    import scripts.clean as clean_mod

    monkeypatch.setattr(clean_mod, "run_clean", lambda cfg: {"clean_rows": 3, "stage": cfg.tmdb_stage})
    assert tasks.clean(params={"tmdb_stage": "all"}, run_id="r") == {"clean_rows": 3, "stage": "all"}


def test_cli_rejects_unknown_step(capsys):
    assert run_pipeline.main(["nonsense"]) == 2
    assert "Unknown step" in capsys.readouterr().err


def test_cli_context_from_args():
    args = run_pipeline.parse_args(["all", "--stage", "rest", "--sample-size", "0", "--no-cache"])
    ctx = run_pipeline.build_context(args)
    assert ctx["params"] == {"tmdb_stage": "rest", "sample_size": None, "use_cache": False}
    assert ctx["run_id"].startswith("cli__")


def test_cli_runs_selected_steps_and_reports_failure(monkeypatch, capsys):
    calls = []

    def ok(**ctx):
        calls.append("ok")
        return {"fine": True}

    def bad(**ctx):
        raise PipelineError("broken input")

    monkeypatch.setattr(run_pipeline, "PIPELINE_STEPS", (("a", ok), ("b", bad), ("c", ok)))
    monkeypatch.setattr(run_pipeline, "STEP_NAMES", ["a", "b", "c"])
    assert run_pipeline.main(["a", "c"]) == 0 and calls == ["ok", "ok"]
    assert run_pipeline.main(["all", "--skip", "c"]) == 1
    assert "broken input" in capsys.readouterr().err


def test_log_step_logs_duration_and_failure(caplog):
    logger = logging_utils.get_logger("scripts.test_mod")
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger="vibespace"):
        with logging_utils.log_step(logger, "demo"):
            pass
        with pytest.raises(ValueError):
            with logging_utils.log_step(logger, "boom"):
                raise ValueError("x")
    text = caplog.text
    assert "START demo" in text and "DONE demo" in text and "FAILED boom" in text


def test_setup_logging_writes_file(tmp_path):
    logger = logging_utils.setup_logging(tmp_path / "a")
    logging_utils.setup_logging(tmp_path / "a")  # idempotent
    logging_utils.setup_logging(tmp_path / "b")  # switches file
    logging_utils.get_logger("x").info("hello file")
    for h in logger.handlers:
        h.flush()
    assert "hello file" in (tmp_path / "b" / "pipeline.log").read_text()
    assert logging_utils.get_logger("scripts.clean").name == "vibespace.clean"
