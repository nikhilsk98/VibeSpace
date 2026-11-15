"""Part 2: Optuna hyperparameter tuning (TPE sampler, median pruner) on the saved Part 1 split.

The objective is the mean validation MAE. Only training and validation rows are ever handed to the
objective (:class:`TuningData` asserts that no test movie is present). Studies are stored in a local
SQLite file so an interrupted run resumes where it stopped; every trial is a nested MLflow run.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from vibespace_model import tracking
from vibespace_model.config import ModelConfig
from vibespace_model.errors import ConfigError
from vibespace_model.evaluate import compute_metrics
from vibespace_model.models import create_model
from vibespace_model.splits import Split, assert_no_test_rows

LOGGER = logging.getLogger(__name__)
TUNABLE: tuple[str, ...] = ("ridge", "lightgbm", "mlp")
_TYPES = {"float", "int", "categorical"}


# ---------------------------------------------------------------------------- search spaces


@dataclass(frozen=True)
class ParamSpec:
    """One hyperparameter's search range."""

    name: str
    type: str
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: tuple[Any, ...] = ()

    def suggest(self, trial: optuna.Trial) -> Any:
        """Ask the trial for a value of this parameter."""
        if self.type == "float":
            return trial.suggest_float(self.name, self.low, self.high, log=self.log, step=self.step)
        if self.type == "int":
            return trial.suggest_int(
                self.name, int(self.low), int(self.high), log=self.log, step=int(self.step or 1)
            )
        return trial.suggest_categorical(self.name, list(self.choices))


def parse_search_space(space: dict[str, Any]) -> dict[str, ParamSpec]:
    """Validate a YAML search space and turn it into :class:`ParamSpec` objects."""
    if not space:
        raise ConfigError("search space is empty")
    specs = {}
    for name, raw in space.items():
        kind = str(raw.get("type", "")).lower()
        if kind not in _TYPES:
            raise ConfigError(f"{name}: type must be one of {sorted(_TYPES)}, got '{kind}'")
        if kind == "categorical":
            choices = tuple(raw.get("choices") or ())
            if not choices:
                raise ConfigError(f"{name}: categorical parameters need a non-empty 'choices' list")
            specs[name] = ParamSpec(name, kind, choices=choices)
            continue
        if "low" not in raw or "high" not in raw:
            raise ConfigError(f"{name}: numeric parameters need 'low' and 'high'")
        low, high, log = float(raw["low"]), float(raw["high"]), bool(raw.get("log", False))
        if low >= high:
            raise ConfigError(f"{name}: low ({low}) must be below high ({high})")
        if log and low <= 0:
            raise ConfigError(f"{name}: log scale needs low > 0")
        if log and raw.get("step"):
            raise ConfigError(f"{name}: 'step' cannot be combined with log scale")
        specs[name] = ParamSpec(name, kind, low, high, log, raw.get("step"))
    return specs


def suggest_params(trial: optuna.Trial, specs: dict[str, ParamSpec]) -> dict[str, Any]:
    """One value per parameter for this trial."""
    return {name: spec.suggest(trial) for name, spec in specs.items()}


# ---------------------------------------------------------------------------- data guard


@dataclass
class TuningData:
    """Train and validation rows only; constructing it asserts that no test movie is included."""

    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    split: Split

    def __post_init__(self) -> None:
        assert_no_test_rows(np.concatenate([self.train_ids, self.val_ids]), self.split, "tuning")

    @classmethod
    def from_prepared(cls, prep: Any) -> TuningData:
        """Take the train / val parts of a :class:`vibespace_model.train.Prepared`."""
        return cls(
            prep.X["train"],
            prep.Y["train"],
            prep.X["val"],
            prep.Y["val"],
            prep.ids["train"],
            prep.ids["val"],
            prep.split,
        )


# ---------------------------------------------------------------------------- studies


def make_sampler(tuning_cfg: dict[str, Any], seed: int) -> optuna.samplers.BaseSampler:
    """TPE sampler with a fixed seed (deterministic sequence of suggestions)."""
    return optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=int(tuning_cfg.get("tpe_n_startup_trials", 10))
    )


def make_pruner(tuning_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    """Median pruner on per-epoch validation MAE (only the MLP reports intermediate values)."""
    cfg = tuning_cfg.get("pruner") or {}
    if cfg.get("type", "median") == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(cfg.get("n_startup_trials", 5)),
        n_warmup_steps=int(cfg.get("n_warmup_steps", 10)),
    )


def storage_url(cfg: ModelConfig) -> str:
    """SQLite file holding all studies (git-ignored, under the artifacts folder)."""
    path = cfg.artifacts_subdir("part2") / cfg.tuning.get("storage_file", "optuna/optuna.db")
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def study_name(model_name: str, split_hash: str, seed: int) -> str:
    """Studies are tied to the split and the seed, so a resumed study never mixes splits."""
    return f"{model_name}-{split_hash[:12]}-seed{seed}"


def resolve_params(model_name: str, fixed: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    """Merge fixed and suggested values (e.g. MLP ``hidden_sizes`` "256-128" -> [256, 128])."""
    params = {**fixed, **suggested}
    if model_name == "mlp" and isinstance(params.get("hidden_sizes"), str):
        params["hidden_sizes"] = [int(v) for v in params["hidden_sizes"].split("-")]
    return params


def evaluate_params(
    model_name: str,
    params: dict[str, Any],
    data: TuningData,
    seed: int,
    targets: list[str],
    epoch_callback: Any = None,
) -> tuple[float, Any]:
    """Fit on train (validation for early stopping) and return (mean validation MAE, model)."""
    model = create_model(model_name, params, seed)
    model.fit(data.X_train, data.Y_train, data.X_val, data.Y_val, epoch_callback=epoch_callback)
    mae = compute_metrics(data.Y_val, model.predict(data.X_val), targets)["overall"]["mae"]
    return float(mae), model


def make_objective(
    model_name: str,
    specs: dict[str, ParamSpec],
    fixed: dict[str, Any],
    data: TuningData,
    seed: int,
    targets: list[str],
    log_to_mlflow: bool = True,
) -> Any:
    """Optuna objective: one trial = one fit + validation MAE, logged as a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        suggested = suggest_params(trial, specs)
        params = resolve_params(model_name, fixed, suggested)
        start = time.perf_counter()

        def report(epoch: int, val_mae: float) -> None:
            trial.report(val_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch}")

        state, value = "complete", None
        try:
            value, _ = evaluate_params(
                model_name,
                params,
                data,
                seed,
                targets,
                epoch_callback=report if model_name == "mlp" else None,
            )
            if not np.isfinite(value):
                raise optuna.TrialPruned("non-finite validation MAE")
            return value
        except optuna.TrialPruned:
            state = "pruned"
            raise
        except Exception:
            state = "failed"
            raise
        finally:
            duration = time.perf_counter() - start
            trial.set_user_attr("duration_seconds", duration)
            if log_to_mlflow:
                with tracking.run(
                    f"{model_name}-trial-{trial.number:03d}",
                    nested=True,
                    tags={"model_type": model_name, "trial_state": state, "trial": trial.number},
                ):
                    tracking.log_params(params)
                    metrics = {"duration_seconds": duration, "pruned": float(state == "pruned")}
                    if value is not None:
                        metrics["val_mae"] = value
                    tracking.log_metrics(metrics)

    return objective


def finished_trials(study: optuna.Study) -> int:
    """Completed + pruned trials (failed trials are retried on resume)."""
    done = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(1 for t in study.trials if t.state in done)


def run_study(
    cfg: ModelConfig,
    model_name: str,
    data: TuningData,
    n_trials: int,
    timeout_seconds: float | None,
    log_to_mlflow: bool = True,
    storage: str | None = None,
) -> optuna.Study:
    """Create or resume the study of one model and run the remaining trials within the timeout."""
    model_cfg = cfg.tuning["models"][model_name]
    specs = parse_search_space(model_cfg["search_space"])
    fixed = {**cfg.model_params(model_name), **(model_cfg.get("fixed") or {})}
    study = optuna.create_study(
        study_name=study_name(model_name, data.split.split_hash, cfg.random_seed),
        storage=storage or storage_url(cfg),
        sampler=make_sampler(cfg.tuning, cfg.random_seed),
        pruner=make_pruner(cfg.tuning),
        direction="minimize",
        load_if_exists=True,
    )
    study.set_user_attr("fixed_params", json.loads(json.dumps(fixed, default=str)))
    remaining = max(0, n_trials - finished_trials(study))
    if remaining and (timeout_seconds is None or timeout_seconds > 0):
        LOGGER.info(
            "Tuning %s: %d trials (%d already done), timeout %s s",
            model_name,
            remaining,
            finished_trials(study),
            None if timeout_seconds is None else round(timeout_seconds),
        )
        study.optimize(
            make_objective(model_name, specs, fixed, data, cfg.random_seed, cfg.targets, log_to_mlflow),
            n_trials=remaining,
            timeout=timeout_seconds,
            gc_after_trial=True,
            catch=(ValueError,),
        )
    else:
        LOGGER.info("Tuning %s: nothing to run (%d trials done)", model_name, finished_trials(study))
    return study


def best_params(study: optuna.Study, model_name: str) -> dict[str, Any]:
    """Fixed + best suggested parameters of a study."""
    return resolve_params(model_name, dict(study.user_attrs.get("fixed_params", {})), dict(study.best_params))


def study_summary(study: optuna.Study) -> dict[str, Any]:
    """Counts, best value and total duration of a study."""
    states = [t.state.name for t in study.trials]
    durations = [t.user_attrs.get("duration_seconds", 0.0) for t in study.trials]
    return {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "n_complete": states.count("COMPLETE"),
        "n_pruned": states.count("PRUNED"),
        "n_failed": states.count("FAIL"),
        "best_value": study.best_value if states.count("COMPLETE") else None,
        "best_trial": study.best_trial.number if states.count("COMPLETE") else None,
        "best_params": dict(study.best_params) if states.count("COMPLETE") else {},
        "total_trial_seconds": float(np.sum(durations)),
    }


def run_tuning(
    cfg: ModelConfig, data: TuningData, log_to_mlflow: bool = True, storage: str | None = None
) -> dict[str, optuna.Study]:
    """Tune every model in ``model_order`` within the per-model and total time caps."""
    total_cap = float(cfg.tuning.get("total_time_cap_minutes", 90)) * 60
    model_caps = cfg.tuning.get("model_time_cap_minutes", {})
    order = [m for m in cfg.tuning.get("model_order", TUNABLE) if m in cfg.tuning.get("models", {})]
    start, studies = time.perf_counter(), {}
    for model_name in order:
        remaining = total_cap - (time.perf_counter() - start)
        cap = float(model_caps.get(model_name, total_cap / 60)) * 60
        n_trials = int(cfg.tuning["models"][model_name]["n_trials"])
        studies[model_name] = run_study(
            cfg, model_name, data, n_trials, max(0.0, min(remaining, cap)), log_to_mlflow, storage
        )
    return studies


def save_best_params(path: Path, params: dict[str, dict[str, Any]]) -> None:
    """Best parameters per model as JSON (also the input of ``--tuned-params``)."""
    path.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")


def load_best_params(path: Path) -> dict[str, dict[str, Any]]:
    """Read :func:`save_best_params` output."""
    if not path.is_file():
        raise ConfigError(f"tuned-params file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
