"""Part 3: bias detection (slice metrics with Fairlearn), the bias gate and mitigation helpers.

Slices are movie attributes (language group, era, primary genre, popularity bucket, short overview):
MovieLens and the app hold no user demographics, so fairness here means "the DNA predictor is not much
worse for some kinds of movies". Slices with at least ``min_gated_slice_size`` movies must have
MAE <= ``max_mae_ratio`` x overall MAE; smaller slices are reported with a bootstrap confidence
interval and marked "insufficient data" without gating.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fairlearn.metrics import MetricFrame  # noqa: E402

from vibespace_model.select import GateResult  # noqa: E402


def _mae(y_true: Any, y_pred: Any) -> float:
    return float(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)).mean())


def _rmse(y_true: Any, y_pred: Any) -> float:
    return float(np.sqrt(((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2).mean()))


def _count(y_true: Any, y_pred: Any) -> int:
    return len(y_true)


def slice_values(catalog: pd.DataFrame, column: str) -> np.ndarray:
    """Slice labels of one column as strings (missing -> "unknown")."""
    return catalog[column].astype("object").where(catalog[column].notna(), "unknown").astype(str).to_numpy()


def metric_frame(Y_true: np.ndarray, Y_pred: np.ndarray, values: np.ndarray) -> MetricFrame:
    """Fairlearn MetricFrame with MAE, RMSE and count per slice (over all 12 dimensions)."""
    return MetricFrame(
        metrics={"mae": _mae, "rmse": _rmse, "n": _count},
        y_true=Y_true,
        y_pred=Y_pred,
        sensitive_features=pd.Series(values, name="slice"),
    )


def bootstrap_ci(
    row_errors: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of per-movie errors."""
    row_errors = np.asarray(row_errors, dtype=float)
    if row_errors.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(row_errors, size=(n_resamples, row_errors.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def evaluate_slices(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    catalog: pd.DataFrame,
    columns: list[str],
    cfg: dict[str, Any],
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """One row per (slice column, slice value): n, MAE, RMSE, ratio to overall, CI, gate status."""
    overall_mae = _mae(Y_true, Y_pred)
    row_mae = np.abs(Y_true - Y_pred).mean(axis=1)
    min_n, max_ratio = int(cfg["min_gated_slice_size"]), float(cfg["max_mae_ratio"])
    boot = cfg.get("bootstrap", {})
    rows = []
    for column in columns:
        if column not in catalog.columns:
            continue
        values = slice_values(catalog, column)
        frame = metric_frame(Y_true, Y_pred, values).by_group
        for value, metrics in frame.iterrows():
            n = int(metrics["n"])
            ratio = float(metrics["mae"]) / overall_mae if overall_mae > 0 else 1.0
            lo, hi = bootstrap_ci(
                row_mae[values == value],
                int(boot.get("n_resamples", 1000)),
                float(boot.get("confidence", 0.95)),
                seed,
            )
            gated = n >= min_n
            status = ("pass" if ratio <= max_ratio else "fail") if gated else "insufficient data"
            rows.append(
                {
                    "slice_column": column,
                    "slice": str(value),
                    "n": n,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "mae_ratio": ratio,
                    "ci_low": lo,
                    "ci_high": hi,
                    "gated": gated,
                    "status": status,
                }
            )
    return pd.DataFrame(rows), overall_mae


def bias_gate(table: pd.DataFrame, overall_mae: float, cfg: dict[str, Any]) -> GateResult:
    """Pass when every gated slice has MAE <= max ratio x overall MAE."""
    gated = table[table["gated"]]
    failing = gated[gated["status"] == "fail"]
    worst = gated.loc[gated["mae_ratio"].idxmax()] if not gated.empty else None
    checks = {
        "overall_mae": overall_mae,
        "max_mae_ratio": float(cfg["max_mae_ratio"]),
        "min_gated_slice_size": int(cfg["min_gated_slice_size"]),
        "n_gated_slices": len(gated),
        "n_small_slices": int((~table["gated"]).sum()),
        "worst_slice": None if worst is None else f"{worst['slice_column']}={worst['slice']}",
        "worst_ratio": None if worst is None else float(worst["mae_ratio"]),
        "failing_slices": [
            f"{r.slice_column}={r.slice} (ratio {r.mae_ratio:.3f}, n={r.n})" for r in failing.itertuples()
        ],
    }
    if failing.empty:
        reasons = [
            f"all {len(gated)} gated slices have MAE <= {cfg['max_mae_ratio']} x overall"
            + (f" (worst {checks['worst_slice']}: {checks['worst_ratio']:.3f})" if worst is not None else "")
        ]
    else:
        reasons = [
            f"slice {s} exceeds {cfg['max_mae_ratio']} x overall MAE" for s in checks["failing_slices"]
        ]
    return GateResult("bias", failing.empty, reasons, checks)


def worst_ratio(table: pd.DataFrame) -> float:
    """Largest MAE ratio among gated slices (the bias gap used for ranking)."""
    gated = table[table["gated"]]
    return float(gated["mae_ratio"].max()) if not gated.empty else 1.0


def gap_by_column(table: pd.DataFrame) -> pd.DataFrame:
    """Per slice column: worst / best gated slice MAE and their ratio."""
    rows = []
    for column, group in table[table["gated"]].groupby("slice_column"):
        rows.append(
            {
                "slice_column": column,
                "worst_mae": group["mae"].max(),
                "best_mae": group["mae"].min(),
                "worst_over_best": group["mae"].max() / group["mae"].min(),
                "worst_ratio_to_overall": group["mae_ratio"].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("worst_ratio_to_overall", ascending=False).reset_index(drop=True)


def choose_mitigation_column(table: pd.DataFrame, configured: str = "auto") -> str:
    """Configured column, or the one with the worst gated ratio (largest gap when all pass)."""
    if configured != "auto":
        return configured
    gaps = gap_by_column(table)
    if gaps.empty:
        return str(table["slice_column"].iloc[0])
    failing = table[(table["status"] == "fail")]
    if not failing.empty:
        return str(failing.sort_values("mae_ratio", ascending=False)["slice_column"].iloc[0])
    return str(gaps.sort_values("worst_over_best", ascending=False)["slice_column"].iloc[0])


def reweighting_weights(values: np.ndarray, power: float = 1.0, max_weight: float = 20.0) -> np.ndarray:
    """Inverse-frequency sample weights ``(1 / freq) ** power``, capped, normalised to mean 1."""
    labels, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    raw = (1.0 / (counts / counts.sum())) ** power
    weights = raw[inverse]
    weights = weights / weights.mean()
    weights = np.minimum(weights, max_weight)
    return weights / weights.mean()


def resample_indices(values: np.ndarray, min_share_of_largest: float = 0.1, seed: int = 42) -> np.ndarray:
    """Training-row indices with small slices oversampled (with replacement) to a minimum size.

    Every original row is kept once; slices smaller than ``min_share_of_largest`` x the largest slice
    get extra copies up to that size. Only ever applied to the training split.
    """
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(values, return_counts=True)
    target = int(np.ceil(counts.max() * min_share_of_largest))
    extra = []
    for label, count in zip(labels, counts, strict=True):
        if count < target:
            members = np.flatnonzero(values == label)
            extra.append(rng.choice(members, size=target - count, replace=True))
    base = np.arange(len(values))
    return np.sort(np.concatenate([base, *extra])) if extra else base


def plot_slice_mae(table: pd.DataFrame, overall_mae: float, max_ratio: float, path: Path, title: str) -> Path:
    """Horizontal bars of slice MAE per slice column with the gate line; small slices hatched."""
    columns = list(dict.fromkeys(table["slice_column"]))
    heights = [max(2.0, 0.28 * (table["slice_column"] == c).sum() + 1) for c in columns]
    fig, axes = plt.subplots(
        len(columns), 1, figsize=(9, sum(heights)), squeeze=False, gridspec_kw={"height_ratios": heights}
    )
    for ax, column in zip(axes[:, 0], columns, strict=True):
        group = table[table["slice_column"] == column].sort_values("mae")
        colors = [
            "#C44E52" if s == "fail" else "#4C72B0" if s == "pass" else "#BBBBBB" for s in group["status"]
        ]
        labels = [f"{s} (n={n})" for s, n in zip(group["slice"], group["n"], strict=True)]
        errors = np.clip([group["mae"] - group["ci_low"], group["ci_high"] - group["mae"]], 0.0, None)
        ax.barh(labels, group["mae"], color=colors, xerr=errors, ecolor="#555555")
        ax.axvline(overall_mae, color="black", linewidth=1, label="overall MAE")
        ax.axvline(
            overall_mae * max_ratio,
            color="#C44E52",
            linestyle="--",
            linewidth=1,
            label=f"gate ({max_ratio} x overall)",
        )
        ax.set_title(column, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
