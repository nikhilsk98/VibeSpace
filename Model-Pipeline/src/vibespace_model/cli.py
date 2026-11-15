"""Command-line entry point.

``python -m vibespace_model.cli <command> --config configs/model_config.yaml [--sample-size N]
[--skip-hash-check] [--tuned-params PATH]``

Commands: ``train`` (Part 1), ``evaluate``, ``select``, ``tune`` and ``sensitivity`` (Part 2), ``explain``
and ``bias`` (Part 3), ``run-all`` (Parts 1-3 end to end); registry (Part 4): ``push``, ``pull``,
``champion``, ``promote``, ``rollback``, ``list-versions`` (all accept ``--dry-run``); CI: ``ci-run``.

Exit codes: 0 ok (for ``promote`` / ``ci-run`` this includes "not promoted"), 1 validation gate failed,
2 bias gate failed, 3 data, configuration or registry error.
"""

from __future__ import annotations

from vibespace_model.runtime import ensure_single_openmp

ensure_single_openmp()  # must run before LightGBM / PyTorch are imported (macOS only)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from vibespace_model.config import load_config  # noqa: E402
from vibespace_model.errors import VibeSpaceModelError  # noqa: E402

EXIT_OK, EXIT_VALIDATION_GATE, EXIT_BIAS_GATE, EXIT_ERROR = 0, 1, 2, 3
REGISTRY_COMMANDS = ("push", "pull", "champion", "promote", "rollback", "list-versions", "protect")
COMMANDS = (
    "train",
    "evaluate",
    "select",
    "tune",
    "sensitivity",
    "explain",
    "bias",
    "run-all",
    *REGISTRY_COMMANDS,
    "ci-run",
)
LOGGER = logging.getLogger("vibespace_model")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser shared by all commands."""
    parser = argparse.ArgumentParser(
        prog="vibespace_model.cli", description="VibeSpace Movie DNA model pipeline"
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="configs/model_config.yaml", help="path to model_config.yaml")
    parser.add_argument("--tuning-config", default=None, help="default: tuning.yaml next to --config")
    parser.add_argument("--bias-config", default=None, help="default: bias.yaml next to --config")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="use a random subset of N movies (outputs go to reports/sample, artifacts/sample)",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="do not compare files with dvc.lock (fixture data only)",
    )
    parser.add_argument(
        "--tuned-params",
        default=None,
        help="reuse best_params.json from an earlier tune run instead of searching",
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=("part1", "part2", "part3"),
        help="select: which stage's gates to check (default: latest available)",
    )
    registry = parser.add_argument_group("registry and CI (Parts 4-5)")
    registry.add_argument(
        "--dry-run",
        action="store_true",
        help="use a local folder instead of GCP (artifacts/registry-dry-run)",
    )
    registry.add_argument(
        "--bundle", default=None, help="saved model bundle folder (default: artifacts/part3/final_model)"
    )
    registry.add_argument("--version", default=None, help="registry version (pull; default: champion)")
    registry.add_argument("--dest", default=None, help="pull: destination folder")
    registry.add_argument("--to-version", default=None, help="rollback: target version (default: previous)")
    registry.add_argument(
        "--local-alias",
        default="champion-candidate",
        help="local MLflow alias of the bundle being pushed / promoted",
    )
    registry.add_argument("--out", default=None, help="ci-run: folder for the CI outputs (default: ci-out)")
    registry.add_argument(
        "--ci-step",
        default="all",
        choices=("train", "validate", "bias", "promote", "all"),
        help="ci-run: run one step (the workflow) or all of them (default)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _validation_code(summary: dict[str, Any]) -> int:
    return EXIT_OK if summary["validation_gate"]["passed"] else EXIT_VALIDATION_GATE


def _bias_code(summary: dict[str, Any]) -> int:
    final = summary["final"]
    if not final["validation_gate"]["passed"]:
        return EXIT_VALIDATION_GATE
    return EXIT_OK if final["bias_gate"]["passed"] else EXIT_BIAS_GATE


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _cmd_select(cfg: Any, stage: str | None) -> int:
    from vibespace_model.workflows import stage_gate

    stages = [stage] if stage else ["part3", "part2", "part1"]
    for candidate in stages:
        if (cfg.reports_subdir(candidate) / f"{candidate}_summary.json").is_file():
            _, gates = stage_gate(cfg, candidate)
            _print({"stage": candidate, "gates": [g.to_dict() for g in gates]})
            for gate in gates:
                if not gate.passed:
                    return EXIT_VALIDATION_GATE if gate.name == "validation" else EXIT_BIAS_GATE
            return EXIT_OK
    raise VibeSpaceModelError("no stage summary found; run `train` first")


def run_command(args: argparse.Namespace) -> int:
    """Execute one command and return its exit code."""
    from vibespace_model import train, workflows

    cfg = load_config(args.config, args.tuning_config, args.bias_config, sample_size=args.sample_size)
    skip = bool(args.skip_hash_check)
    tuned = Path(args.tuned_params) if args.tuned_params else None
    command = args.command

    if command == "train":
        summary = train.run_part1(cfg, skip)
        _print({"selected": summary["selected"], "validation_gate": summary["validation_gate"]})
        return _validation_code(summary)
    if command == "evaluate":
        summary = train.run_evaluate(cfg, skip)
        _print({name: res["val"]["overall"] for name, res in summary["results"].items()})
        return EXIT_OK
    if command == "select":
        return _cmd_select(cfg, args.stage)
    if command == "tune":
        summary = workflows.run_tune(cfg, skip, tuned)
        _print({"selected": summary["selected"], "validation_gate": summary["validation_gate"]})
        return _validation_code(summary)
    if command == "sensitivity":
        summary = workflows.run_sensitivity(cfg, skip)
        _print({"swept_params": summary.get("swept_params"), "skipped": summary.get("skipped", False)})
        return EXIT_OK
    if command == "explain":
        summary = workflows.run_explain(cfg, skip)
        _print({m: v["top_groups"] for m, v in summary["models"].items()})
        return EXIT_OK
    if command == "bias":
        summary = workflows.run_bias(cfg, skip)
        _print({"final": summary["final"]["name"], "decision": summary["decision"]})
        return _bias_code(summary)
    if command in REGISTRY_COMMANDS:
        from vibespace_model import registry_cli

        payload = registry_cli.run(command, cfg, args)
        _print(payload)
        return EXIT_OK
    if command == "ci-run":
        from vibespace_model import ci_run

        return ci_run.main(cfg, args)

    # run-all: Part 1 -> Part 2 -> Part 3; continue past a failed gate, report the worst code at the end
    codes = [_validation_code(train.run_part1(cfg, skip))]
    codes.append(_validation_code(workflows.run_tune(cfg, skip, tuned)))
    workflows.run_sensitivity(cfg, skip)
    workflows.run_explain(cfg, skip)
    final = workflows.run_bias(cfg, skip)
    codes.append(_bias_code(final))
    _print({"final": final["final"]["name"], "decision": final["decision"], "exit_codes": codes})
    return _bias_code(final)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run, and map known errors to exit code 3."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    for noisy in ("alembic", "mlflow", "optuna", "matplotlib", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return run_command(args)
    except VibeSpaceModelError as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
