"""Run pipeline steps without Airflow (for debugging, tests and ``dvc repro``).

Examples::

    python -m scripts.run_pipeline all --sample-size 200
    python -m scripts.run_pipeline clean
    python -m scripts.run_pipeline all --skip dvc_version --stage genome
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

from scripts.exceptions import PipelineError
from scripts.tasks import PIPELINE_STEPS

STEP_NAMES = [name for name, _ in PIPELINE_STEPS]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Run VibeSpace data pipeline steps outside Airflow.")
    parser.add_argument("steps", nargs="+", help=f"'all' or any of: {', '.join(STEP_NAMES)}")
    parser.add_argument("--stage", choices=["genome", "rest", "all"], help="TMDB stage (default: TMDB_STAGE env)")
    parser.add_argument("--sample-size", type=int, help="Run on N movies only (0 = all)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore TMDB / embedding caches")
    parser.add_argument("--skip", nargs="*", default=[], help="Steps to skip when running 'all'")
    parser.add_argument("--run-id", default=None, help="Run identifier used for stats history")
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace) -> dict[str, Any]:
    """Fake the parts of the Airflow context the task layer reads."""
    params: dict[str, Any] = {}
    if args.stage:
        params["tmdb_stage"] = args.stage
    if args.sample_size is not None:
        params["sample_size"] = args.sample_size or None
    if args.no_cache:
        params["use_cache"] = False
    run_id = args.run_id or f"cli__{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    return {"params": params, "run_id": run_id}


def main(argv: list[str] | None = None) -> int:
    """Run the requested steps in pipeline order; return a process exit code."""
    args = parse_args(argv)
    wanted = STEP_NAMES if args.steps == ["all"] else args.steps
    unknown = [s for s in wanted if s not in STEP_NAMES]
    if unknown:
        print(f"Unknown step(s): {unknown}. Valid: {STEP_NAMES}", file=sys.stderr)
        return 2
    context = build_context(args)
    results: dict[str, Any] = {}
    for name, fn in PIPELINE_STEPS:
        if name not in wanted or name in args.skip:
            continue
        try:
            results[name] = fn(**context)
        except PipelineError as exc:
            print(f"Step {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
