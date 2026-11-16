"""CLI commands and exit codes."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from vibespace_model.cli import EXIT_BIAS_GATE, EXIT_ERROR, EXIT_OK, EXIT_VALIDATION_GATE, build_parser, main


def _args(tree, *extra: str) -> list[str]:
    return ["--config", str(tree["config"]), "--log-level", "WARNING", *extra]


def test_parser_lists_all_commands() -> None:
    parser = build_parser()
    for command in ("train", "evaluate", "tune", "sensitivity", "explain", "bias", "select", "run-all"):
        assert parser.parse_args([command]).command == command
    with pytest.raises(SystemExit):
        parser.parse_args(["deploy"])


def test_train_select_evaluate_exit_codes(tree, capsys) -> None:
    assert main(["select", *_args(tree)]) == EXIT_ERROR  # nothing trained yet
    assert main(["train", *_args(tree)]) == EXIT_OK
    out = capsys.readouterr().out
    assert json.loads(out)["validation_gate"]["passed"] is True
    assert main(["select", *_args(tree, "--stage", "part1")]) == EXIT_OK
    assert main(["select", *_args(tree)]) == EXIT_OK
    assert main(["evaluate", *_args(tree)]) == EXIT_OK


def test_gate_failure_gives_exit_code_1(tree) -> None:
    raw = yaml.safe_load(tree["config"].read_text())
    raw["gate"]["min_r2"] = 0.99
    tree["config"].write_text(yaml.safe_dump(raw))
    assert main(["train", *_args(tree)]) == EXIT_VALIDATION_GATE
    assert main(["select", *_args(tree, "--stage", "part1")]) == EXIT_VALIDATION_GATE


def test_data_error_gives_exit_code_3(tree) -> None:
    labels = pd.read_parquet(tree["processed"] / "movie_dna_labels.parquet")
    labels.loc[0, "energy"] = 0.42
    labels.to_parquet(tree["processed"] / "movie_dna_labels.parquet", index=False)
    assert main(["train", *_args(tree)]) == EXIT_ERROR  # hash mismatch with dvc.lock
    assert main(["train", *_args(tree, "--skip-hash-check")]) == EXIT_OK


def test_sample_size_writes_to_sample_folders(tree) -> None:
    assert main(["train", *_args(tree, "--sample-size", "120")]) == EXIT_OK
    reports = tree["root"] / "Model-Pipeline" / "reports"
    summary = json.loads((reports / "sample" / "part1" / "part1_summary.json").read_text())
    assert summary["join_report"]["sample_size"] == 120
    assert sum(summary["split_counts"].values()) == 120
    assert not (reports / "part1").exists()


def test_bias_exit_code_mapping() -> None:
    from vibespace_model.cli import _bias_code

    final = {"validation_gate": {"passed": True}, "bias_gate": {"passed": False}}
    assert _bias_code({"final": final}) == EXIT_BIAS_GATE
    assert _bias_code({"final": {**final, "validation_gate": {"passed": False}}}) == EXIT_VALIDATION_GATE
    assert _bias_code({"final": {**final, "bias_gate": {"passed": True}}}) == EXIT_OK
