"""Regression metrics, comparison tables and plots.

Per dimension: MAE, RMSE, R² and Spearman. Overall values are the unweighted mean over the 12 dimensions
(for MAE this equals the MAE over all cells). Spearman is 0.0 when a prediction is constant (e.g. the
mean baseline), where the coefficient is undefined.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

METRICS: tuple[str, ...] = ("mae", "rmse", "r2", "spearman")


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    total = float(((y - y.mean()) ** 2).sum())
    return 0.0 if total == 0 else 1.0 - float(((y - p) ** 2).sum()) / total


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    if np.ptp(p) == 0 or np.ptp(y) == 0:
        return 0.0
    value = spearmanr(y, p).statistic
    return 0.0 if np.isnan(value) else float(value)


def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, targets: list[str]) -> dict[str, Any]:
    """``{"overall": {...}, "per_dimension": {dim: {...}}}`` for the 4 metrics."""
    Y_true, Y_pred = np.asarray(Y_true, dtype=float), np.asarray(Y_pred, dtype=float)
    if Y_true.shape != Y_pred.shape or Y_true.shape[1] != len(targets):
        raise ValueError(f"shape mismatch: {Y_true.shape} vs {Y_pred.shape} for {len(targets)} targets")
    per_dim = {}
    for k, dim in enumerate(targets):
        y, p = Y_true[:, k], Y_pred[:, k]
        per_dim[dim] = {
            "mae": float(np.abs(y - p).mean()),
            "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "r2": _r2(y, p),
            "spearman": _spearman(y, p),
        }
    overall = {m: float(np.mean([per_dim[d][m] for d in targets])) for m in METRICS}
    overall["n"] = len(Y_true)
    return {"overall": overall, "per_dimension": per_dim}


def improvement_over_baseline(mae: float, baseline_mae: float) -> float:
    """Relative MAE reduction vs the baseline (0.15 = 15% lower MAE)."""
    return 0.0 if baseline_mae <= 0 else (baseline_mae - mae) / baseline_mae


def comparison_table(results: dict[str, dict[str, Any]], baseline: str = "baseline") -> pd.DataFrame:
    """One row per model: overall validation metrics, improvement over the baseline, train time."""
    base_mae = results[baseline]["val"]["overall"]["mae"] if baseline in results else None
    rows = []
    for name, res in results.items():
        overall = res["val"]["overall"]
        row = {"model": name, **{f"val_{m}": overall[m] for m in METRICS}}
        if base_mae is not None:
            row["improvement_vs_baseline"] = improvement_over_baseline(overall["mae"], base_mae)
        if "test" in res:
            row["test_mae"] = res["test"]["overall"]["mae"]
        row["train_seconds"] = res.get("train_seconds")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)


def per_dimension_table(
    results: dict[str, dict[str, Any]], split: str = "val", metric: str = "mae"
) -> pd.DataFrame:
    """Dimensions x models table of one metric."""
    return pd.DataFrame(
        {
            name: {d: v[metric] for d, v in res[split]["per_dimension"].items()}
            for name, res in results.items()
        }
    )


def write_json(path: Path, payload: Any) -> None:
    """Pretty JSON with numpy types converted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ---------------------------------------------------------------------------- plots


def plot_model_comparison(table: pd.DataFrame, path: Path, title: str = "Validation MAE by model") -> Path:
    """Bar chart of mean validation MAE per model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(table["model"], table["val_mae"], color="#4C72B0")
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_ylabel("mean validation MAE (lower is better)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_per_dimension(
    table: pd.DataFrame, path: Path, title: str = "Validation MAE per DNA dimension"
) -> Path:
    """Grouped bars: one group per dimension, one bar per model."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    n_models = table.shape[1]
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(table.index))
    for i, model in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[model], width, label=model)
    ax.set_xticks(x, table.index, rotation=30, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pred_vs_true(
    Y_true: np.ndarray, Y_pred: np.ndarray, targets: list[str], dims: list[str], path: Path, title: str
) -> Path:
    """Scatter of predicted vs true scores for a few dimensions."""
    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4), squeeze=False)
    for ax, dim in zip(axes[0], dims, strict=True):
        k = targets.index(dim)
        ax.scatter(Y_true[:, k], Y_pred[:, k], s=4, alpha=0.3)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(dim)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
