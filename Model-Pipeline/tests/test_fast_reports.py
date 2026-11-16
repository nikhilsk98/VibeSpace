"""Fast runs (--tuned-params) never overwrite the committed full-run reports in reports/part1-3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vibespace_model.cli import EXIT_ERROR, EXIT_OK, main
from vibespace_model.config import load_config

pytestmark = pytest.mark.integration


def _args(tree, *extra: str) -> list[str]:
    return ["--config", str(tree["config"]), "--log-level", "WARNING", *extra]


def _hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def committed(tree, tmp_path):
    """A normal Part 1 run plus stand-ins for committed Part 2 / Part 3 reports, and a best_params.json."""
    assert main(["train", *_args(tree)]) == EXIT_OK
    reports = load_config(tree["config"]).paths.reports_dir
    for part in ("part2", "part3"):
        (reports / part).mkdir(parents=True, exist_ok=True)
        (reports / part / f"{part}_summary.json").write_text('{"committed": true}\n')
    models = yaml.safe_load(tree["config"].read_text())["models"]
    best = tmp_path / "best_params.json"
    best.write_text(json.dumps({name: models[name] for name in ("ridge", "lightgbm", "mlp")}))
    return reports, best


def test_fast_run_leaves_committed_reports_unchanged(tree, committed) -> None:
    reports, best = committed
    before = {part: _hashes(reports / part) for part in ("part1", "part2", "part3")}
    assert main(["run-all", *_args(tree, "--tuned-params", str(best))]) == EXIT_OK
    assert {part: _hashes(reports / part) for part in ("part1", "part2", "part3")} == before
    fast = reports / "fast"
    assert (fast / "part1" / "part1_summary.json").is_file()
    assert json.loads((fast / "part2" / "part2_summary.json").read_text())["mode"] == "reuse"
    assert (fast / "part3" / "part3_summary.json").is_file()


def test_single_fast_command_reads_the_committed_earlier_part(tree, committed) -> None:
    reports, best = committed
    part1_before = _hashes(reports / "part1")
    assert main(["tune", *_args(tree, "--tuned-params", str(best))]) == EXIT_OK
    assert _hashes(reports / "part1") == part1_before
    assert (reports / "part2" / "part2_summary.json").read_text() == '{"committed": true}\n'
    assert (reports / "fast" / "part2" / "part2_summary.json").is_file()


def test_write_reports_flag_updates_committed_reports(tree, committed) -> None:
    reports, best = committed
    assert main(["tune", *_args(tree, "--tuned-params", str(best), "--write-reports")]) == EXIT_OK
    summary = json.loads((reports / "part2" / "part2_summary.json").read_text())
    assert summary["mode"] == "reuse"
    assert not (reports / "fast").exists()


def test_write_reports_rejected_for_sample_runs(tree, committed) -> None:
    _, best = committed
    args = _args(tree, "--tuned-params", str(best), "--write-reports", "--sample-size", "50")
    assert main(["tune", *args]) == EXIT_ERROR


def test_full_runs_still_write_committed_reports(tree) -> None:
    cfg = load_config(tree["config"])
    assert not cfg.fast_reports and cfg.reports_base == cfg.paths.reports_dir
    cfg.fast_reports = True
    assert cfg.reports_base == cfg.paths.reports_dir / "fast"
    cfg.sample_size = 10
    assert cfg.reports_base == cfg.paths.reports_dir / "sample"
