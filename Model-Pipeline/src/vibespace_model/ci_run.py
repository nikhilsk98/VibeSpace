"""Part 5: the CI training sequence ``train -> validate -> bias -> promote`` with clear exit codes.

``python -m vibespace_model.cli ci-run [--ci-step train|validate|bias|promote|all] [--dry-run]
[--sample-size N] [--skip-hash-check] [--tuned-params best_params.json] [--out ci-out]``

- **train**: recreate the fixed Part 1 split (checked against ``ci.expected_split_hash``), train the final
  model family with the tuned parameters from ``ci.params`` (no Optuna), save the bundle and metrics.
- **validate**: validation gate (exit 1 on failure).
- **bias**: bias gate on validation slices (exit 2 on failure; the registry push is then blocked).
- **promote**: push to the registry and apply the champion rule (``--dry-run``: local folder, no GCP).

Each step reads and updates ``<out>/state.json``, so the workflow can run the steps separately and fail at
the right one. ``state.json["event"]`` is what the notification step reports: ``training_completed``,
``validation_failed``, ``bias_failed``, ``promoted``, ``not_promoted`` or ``pipeline_error``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from vibespace_model import tracking
from vibespace_model.config import ModelConfig
from vibespace_model.errors import ConfigError, SplitError
from vibespace_model.evaluate import plot_per_dimension, write_json
from vibespace_model.promote import (
    Evaluation,
    baseline_val_overall,
    build_registry,
    evaluate_bundle_dir,
    promote,
)
from vibespace_model.select import GateResult, build_model_card, save_bundle
from vibespace_model.train import Prepared, fit_model, prepare
from vibespace_model.tuning import load_best_params

LOGGER = logging.getLogger(__name__)
STEPS = ("train", "validate", "bias", "promote")
EXIT_OK, EXIT_VALIDATION, EXIT_BIAS, EXIT_ERROR = 0, 1, 2, 3


# ---------------------------------------------------------------------------- state


def out_dir(cfg: ModelConfig, args: argparse.Namespace) -> Path:
    """Output folder of the CI run (``--out`` or ``<Model-Pipeline>/ci-out``)."""
    path = Path(args.out) if getattr(args, "out", None) else cfg.root / "ci-out"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_state(out: Path) -> dict[str, Any]:
    """``state.json`` of the run (empty when starting)."""
    path = out / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"steps": {}}


def save_state(out: Path, state: dict[str, Any]) -> None:
    """Persist the state and refresh the Markdown job summary."""
    write_json(out / "state.json", state)
    (out / "job_summary.md").write_text(job_summary(state), encoding="utf-8")


def _mark(state: dict[str, Any], step: str, status: str, started: float, **info: Any) -> None:
    state["steps"][step] = {"status": status, "seconds": round(time.perf_counter() - started, 1), **info}


# ---------------------------------------------------------------------------- helpers


def ci_family_and_params(cfg: ModelConfig, tuned_params: Path | None) -> tuple[str, dict[str, Any]]:
    """Final model family and its tuned parameters (config, or a ``best_params.json`` override)."""
    family = cfg.ci.get("family")
    if not family:
        raise ConfigError("ci.family is not set in the config")
    if tuned_params:
        best = load_best_params(tuned_params)
        if family not in best:
            raise ConfigError(f"{tuned_params} has no parameters for '{family}'")
        return family, dict(best[family])
    if "params" not in cfg.ci:
        raise ConfigError("ci.params is not set in the config (or pass --tuned-params)")
    return family, dict(cfg.ci["params"])


def prepare_ci(cfg: ModelConfig, skip_hash_check: bool) -> Prepared:
    """Recreate the deterministic Part 1 split and verify its hash (full runs only)."""
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=True)
    expected = cfg.ci.get("expected_split_hash")
    if expected and not cfg.sample_size and prep.split.split_hash != expected:
        raise SplitError(
            f"split hash {prep.split.split_hash} differs from ci.expected_split_hash {expected}: "
            "the data or the split settings changed"
        )
    return prep


def _evaluation(cfg: ModelConfig, prep: Prepared, out: Path) -> Evaluation:
    return evaluate_bundle_dir(cfg, prep, out / "bundle", "ci-candidate", baseline_val_overall(cfg, prep))


# ---------------------------------------------------------------------------- steps


def step_train(cfg: ModelConfig, args: argparse.Namespace, out: Path, state: dict[str, Any]) -> int:
    """Train the final family with the tuned parameters and save the bundle."""
    started = time.perf_counter()
    family, params = ci_family_and_params(cfg, Path(args.tuned_params) if args.tuned_params else None)
    prep = prepare_ci(cfg, args.skip_hash_check)
    model, seconds = fit_model(family, params, prep, cfg.random_seed)
    tracking.setup_tracking(cfg)
    tags = tracking.session_tags(cfg, prep.dataset.data_version, prep.split.split_hash, "ci")
    with tracking.run("ci-train", tags=tags) as active:
        tracking.log_params(model.get_params())
        card = build_model_card(
            model=model,
            stage="ci-candidate",
            data_version=prep.dataset.data_version,
            split_hash=prep.split.split_hash,
            metrics={},
            gate_cfg=cfg.gate,
            validation_gate_result=GateResult("validation", False, ["not evaluated yet"]),
            extra={"targets": cfg.targets, "ci_family": family},
        )
        bundle = save_bundle(out / "bundle", model, prep.builder, card)
        evaluation = _evaluation(cfg, prep, out)
        card.update(
            {
                "metrics": {"val": evaluation.val, "test": evaluation.test},
                "validation_gate": evaluation.validation_gate.to_dict(),
                "bias_gate": evaluation.bias_gate.to_dict(),
                "slice_summary_val": _gaps(evaluation),
            }
        )
        write_json(bundle / "model_card.json", card)
        tracking.log_metrics({**tracking.flatten_metrics(evaluation.val, "val"), "train_seconds": seconds})
        registration = tracking.register_bundle(
            cfg,
            bundle,
            alias=cfg.ci.get("local_alias", "ci-candidate"),
            description=f"CI candidate ({family}, tuned parameters)",
        )
        registration["run_id"] = active.info.run_id
    shutil.copy2(bundle / "model_card.json", out / "model_card.json")
    write_json(out / "metrics.json", {"val": evaluation.val, "test": evaluation.test})
    per_dim = pd.DataFrame({family: {d: v["mae"] for d, v in evaluation.val["per_dimension"].items()}})
    plot_per_dimension(per_dim, out / "val_mae_per_dimension.png", f"CI candidate ({family}): validation MAE")
    state.update(
        {
            "family": family,
            "params": params,
            "split_hash": prep.split.split_hash,
            "data_version": prep.dataset.data_version,
            "registration": registration,
            "sample_size": cfg.sample_size,
            "event": "training_completed",
            "val": evaluation.val["overall"],
            "test": evaluation.test["overall"],
        }
    )
    _mark(state, "train", "passed", started, train_seconds=round(seconds, 1))
    return EXIT_OK


def _gaps(evaluation: Evaluation) -> list[dict[str, Any]]:
    from vibespace_model.bias import gap_by_column

    return gap_by_column(evaluation.slice_table).to_dict(orient="records")


def step_validate(cfg: ModelConfig, args: argparse.Namespace, out: Path, state: dict[str, Any]) -> int:
    """Validation gate on the shared split."""
    started = time.perf_counter()
    _require(state, "train")
    evaluation = _evaluation(cfg, prepare_ci(cfg, args.skip_hash_check), out)
    gate = evaluation.validation_gate
    write_json(out / "validation_gate.json", gate.to_dict())
    state["validation_gate"] = gate.to_dict()
    _mark(state, "validate", "passed" if gate.passed else "failed", started)
    if not gate.passed:
        state["event"] = "validation_failed"
        return EXIT_VALIDATION
    return EXIT_OK


def step_bias(cfg: ModelConfig, args: argparse.Namespace, out: Path, state: dict[str, Any]) -> int:
    """Bias gate on validation slices; writes the slice table and a short Markdown report."""
    started = time.perf_counter()
    _require(state, "train")
    evaluation = _evaluation(cfg, prepare_ci(cfg, args.skip_hash_check), out)
    gate = evaluation.bias_gate
    evaluation.slice_table.to_csv(out / "bias_slices_val.csv", index=False, lineterminator="\n")
    write_json(out / "bias_gate.json", gate.to_dict())
    (out / "bias_report.md").write_text(_bias_markdown(evaluation, cfg), encoding="utf-8")
    state["bias_gate"] = gate.to_dict()
    _mark(state, "bias", "passed" if gate.passed else "failed", started)
    if not gate.passed:
        state["event"] = "bias_failed"
        return EXIT_BIAS
    return EXIT_OK


def step_promote(cfg: ModelConfig, args: argparse.Namespace, out: Path, state: dict[str, Any]) -> int:
    """Registry push + champion rule; refuses to run unless both gates passed."""
    started = time.perf_counter()
    _require(state, "train")
    for gate in ("validation_gate", "bias_gate"):
        if not (state.get(gate) or {}).get("passed"):
            _mark(state, "promote", "blocked", started, reason=f"{gate} has not passed")
            state["event"] = "validation_failed" if gate == "validation_gate" else "bias_failed"
            return EXIT_VALIDATION if gate == "validation_gate" else EXIT_BIAS
    registry = build_registry(cfg, dry_run=args.dry_run)
    reg = state.get("registration") or {}
    extra = {
        "mlflow_run_id": reg.get("run_id"),
        "local_registry_version": reg.get("version"),
        "source": "ci-run",
        "dry_run": bool(args.dry_run),
        "sample_size": cfg.sample_size,
    }
    result = promote(cfg, registry, out / "bundle", prepare_ci(cfg, args.skip_hash_check), extra=extra)
    write_json(out / "decision.json", result)
    decision = result["decision"]
    state.update(
        {
            "decision": decision,
            "version": result["version"],
            "backend": result["backend"],
            "champion_after": result["champion_after"],
            "champion_before": result["champion_before"],
            "champion_evaluation": result["champion_evaluation"],
            "event": "promoted" if decision["promote"] else "not_promoted",
        }
    )
    _mark(state, "promote", "passed", started, label=decision["label"])
    return EXIT_OK


def _require(state: dict[str, Any], step: str) -> None:
    if (state["steps"].get(step) or {}).get("status") != "passed":
        raise ConfigError(f"ci-run step '{step}' has not completed in this output folder; run it first")


STEP_FUNCS = {"train": step_train, "validate": step_validate, "bias": step_bias, "promote": step_promote}


def main(cfg: ModelConfig, args: argparse.Namespace) -> int:
    """Run one step or the whole sequence; always leaves ``state.json`` and ``job_summary.md`` behind."""
    out = out_dir(cfg, args)
    step = getattr(args, "ci_step", None) or "all"
    if step in ("all", "train"):
        for stale in ("state.json", "decision.json", "validation_gate.json", "bias_gate.json"):
            (out / stale).unlink(missing_ok=True)
    state = load_state(out)
    state.setdefault("started_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    state["dry_run"] = bool(args.dry_run)
    code = EXIT_OK
    try:
        for name in STEPS if step == "all" else (step,):
            code = STEP_FUNCS[name](cfg, args, out, state)
            save_state(out, state)
            if code != EXIT_OK:
                break
    except Exception as exc:
        state["event"] = "pipeline_error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save_state(out, state)
        LOGGER.error("ci-run failed: %s", state["error"])
        return EXIT_ERROR
    save_state(out, state)
    print(
        json.dumps(
            {
                "event": state.get("event"),
                "exit_code": code,
                "decision": (state.get("decision") or {}).get("label"),
                "version": state.get("version"),
                "out": str(out),
            },
            indent=2,
        )
    )
    return code


# ---------------------------------------------------------------------------- reports


def _bias_markdown(evaluation: Evaluation, cfg: ModelConfig) -> str:
    gate = evaluation.bias_gate
    table = evaluation.slice_table
    rows = [
        f"| {r.slice_column} | {r.slice} | {r.n} | {r.mae:.4f} | {r.mae_ratio:.3f} | {r.status} |"
        for r in table.itertuples()
    ]
    return "\n".join(
        [
            "# CI bias check (validation split)",
            "",
            f"Gate: slices with at least {cfg.bias['min_gated_slice_size']} movies must have MAE <= "
            f"{cfg.bias['max_mae_ratio']} x overall MAE. Result: **{'passed' if gate.passed else 'failed'}** "
            f"({'; '.join(gate.reasons)}).",
            "",
            "| Slice column | Slice | n | MAE | Ratio | Status |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
        ]
    )


NOT_PROMOTED_BANNER = "NOT PROMOTED: champion unchanged (rollback)"


def headline(state: dict[str, Any]) -> str | None:
    """One-line outcome shown at the top of the summary (None while the run has no outcome yet)."""
    event = state.get("event")
    champion = (state.get("champion_after") or {}).get("version")
    return {
        "not_promoted": f"{NOT_PROMOTED_BANNER}: `{champion}` stays champion",
        "promoted": f"PROMOTED: `{state.get('version')}` is the new champion",
        "validation_failed": "VALIDATION GATE FAILED: nothing was registered",
        "bias_failed": "BIAS GATE FAILED: registry push blocked",
        "pipeline_error": "PIPELINE ERROR: see the failed step",
    }.get(event or "")


def job_summary(state: dict[str, Any]) -> str:
    """Markdown for ``$GITHUB_STEP_SUMMARY``: outcome first, then metrics, gates, promotion decision."""
    lines = []
    top = headline(state)
    if top:
        lines += [f"> **{top}**", ""]
        if state.get("event") == "not_promoted":
            lines += [f"> - {reason}" for reason in (state.get("decision") or {}).get("reasons", [])] + [""]
    lines += ["## VibeSpace Movie DNA training run", ""]
    mode = "dry run (local registry folder)" if state.get("dry_run") else "GCP registry"
    lines.append(
        f"Mode: {mode}" + (f" · sample of {state['sample_size']} movies" if state.get("sample_size") else "")
    )
    if state.get("family"):
        lines.append(
            f"Model: `{state['family']}` with the tuned Part 2 parameters · split hash "
            f"`{str(state.get('split_hash', ''))[:12]}`"
        )
    lines += ["", "| Step | Status | Seconds |", "|---|---|---|"]
    for name in STEPS:
        info = state["steps"].get(name)
        lines.append(
            f"| {name} | {info['status'] if info else 'not run'} | {info['seconds'] if info else '–'} |"
        )
    if state.get("val"):
        lines += ["", "| Split | MAE | RMSE | R² | Spearman |", "|---|---|---|---|---|"]
        for split in ("val", "test"):
            m = state[split]
            lines.append(
                f"| {split} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.3f} | {m['spearman']:.3f} |"
            )
    for gate in ("validation_gate", "bias_gate"):
        if state.get(gate):
            g = state[gate]
            lines += [
                "",
                f"**{gate.replace('_', ' ').title()}: {'passed' if g['passed'] else 'FAILED'}** – "
                + "; ".join(g["reasons"]),
            ]
    if state.get("decision"):
        d = state["decision"]
        champ = (state.get("champion_after") or {}).get("version")
        lines += [
            "",
            f"**Promotion: {d['label']}** (candidate `{state.get('version')}`, champion now `{champ}`)",
            "",
            *[f"- {r}" for r in d["reasons"]],
        ]
    if state.get("error"):
        lines += ["", f"**Pipeline error:** {state['error']}"]
    lines += ["", f"Notification event: `{state.get('event', 'none')}`", ""]
    return "\n".join(lines)
