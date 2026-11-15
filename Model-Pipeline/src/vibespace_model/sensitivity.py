"""Part 2: hyperparameter sensitivity.

1. Optuna parameter importance (fANOVA, fixed seed) for every tuned model.
2. One-at-a-time sweeps of the final candidate's most important hyperparameters: each sweep changes one
   value and keeps the others at their best values, then records the validation MAE.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from vibespace_model.tuning import ParamSpec, TuningData, evaluate_params, resolve_params  # noqa: E402

LOGGER = logging.getLogger(__name__)


def param_importances(study: optuna.Study, seed: int, evaluator: str = "fanova") -> dict[str, float]:
    """Normalised importance per hyperparameter (empty if the study has fewer than 2 complete trials)."""
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < 2 or not study.best_params:
        return {}
    chosen = (
        optuna.importance.FanovaImportanceEvaluator(seed=seed)
        if evaluator == "fanova"
        else optuna.importance.MeanDecreaseImpurityImportanceEvaluator(seed=seed)
    )
    try:
        return {
            k: float(v) for k, v in optuna.importance.get_param_importances(study, evaluator=chosen).items()
        }
    except (RuntimeError, ValueError) as exc:  # e.g. a parameter with a single observed value
        LOGGER.warning("Parameter importance failed for %s: %s", study.study_name, exc)
        return {}


def sweep_values(spec: ParamSpec, best: Any, points: int) -> list[Any]:
    """Values to try for one parameter: a grid over its range plus the best value."""
    if spec.type == "categorical":
        return list(spec.choices)
    grid = np.geomspace(spec.low, spec.high, points) if spec.log else np.linspace(spec.low, spec.high, points)
    values = [*grid.tolist(), float(best)]
    if spec.type == "int":
        values = [int(round(v)) for v in values]
    return sorted(set(values), key=float)


def run_sweeps(
    model_name: str,
    best_suggested: dict[str, Any],
    fixed: dict[str, Any],
    specs: dict[str, ParamSpec],
    params_to_sweep: list[str],
    data: TuningData,
    seed: int,
    targets: list[str],
    points: int = 7,
) -> pd.DataFrame:
    """Validation MAE for each value of each swept parameter (others at their best values)."""
    rows = []
    for name in params_to_sweep:
        for value in sweep_values(specs[name], best_suggested[name], points):
            suggested = {**best_suggested, name: value}
            mae, _ = evaluate_params(
                model_name, resolve_params(model_name, fixed, suggested), data, seed, targets
            )
            rows.append(
                {"param": name, "value": value, "val_mae": mae, "is_best": value == best_suggested[name]}
            )
            LOGGER.info("Sweep %s %s=%s -> val MAE %.5f", model_name, name, value, mae)
    return pd.DataFrame(rows)


def sweep_summary(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Per parameter: best / worst MAE over the sweep and the spread (how sensitive the model is)."""
    rows = []
    for name, group in sweeps.groupby("param", sort=False):
        best_row = group.loc[group["val_mae"].idxmin()]
        rows.append(
            {
                "param": name,
                "min_val_mae": float(group["val_mae"].min()),
                "max_val_mae": float(group["val_mae"].max()),
                "spread": float(group["val_mae"].max() - group["val_mae"].min()),
                "relative_spread": float(
                    (group["val_mae"].max() - group["val_mae"].min()) / group["val_mae"].min()
                ),
                "best_value_in_sweep": best_row["value"],
            }
        )
    return pd.DataFrame(rows)


def plot_sweeps(sweeps: pd.DataFrame, specs: dict[str, ParamSpec], path: Path, title: str) -> Path:
    """One panel per swept parameter: validation MAE vs value (log x-axis for log-scale parameters)."""
    params = list(dict.fromkeys(sweeps["param"]))
    fig, axes = plt.subplots(1, max(len(params), 1), figsize=(4.5 * max(len(params), 1), 3.8), squeeze=False)
    for ax, name in zip(axes[0], params, strict=False):
        group = sweeps[sweeps["param"] == name]
        if specs[name].type == "categorical":
            labels = [str(v) for v in group["value"]]
            ax.bar(labels, group["val_mae"], color=["#DD8452" if b else "#4C72B0" for b in group["is_best"]])
            ax.tick_params(axis="x", rotation=30)
            ax.set_ylim(group["val_mae"].min() * 0.98, group["val_mae"].max() * 1.01)
        else:
            ax.plot(group["value"].astype(float), group["val_mae"], marker="o")
            best = group[group["is_best"]]
            ax.scatter(best["value"].astype(float), best["val_mae"], color="#DD8452", zorder=3, label="best")
            if specs[name].log:
                ax.set_xscale("log")
            ax.legend(fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_ylabel("validation MAE")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_importances(importances: dict[str, dict[str, float]], path: Path) -> Path:
    """Horizontal bars of fANOVA importance per tuned model."""
    models = [m for m, imp in importances.items() if imp]
    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(5 * max(len(models), 1), 3.8), squeeze=False)
    for ax, model in zip(axes[0], models, strict=False):
        items = sorted(importances[model].items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color="#55A868")
        ax.set_title(f"{model}: hyperparameter importance")
        ax.set_xlabel("fANOVA importance (normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
