"""Part 3: feature importance with SHAP and permutation importance, grouped by feature type.

SHAP values per model family:
- LightGBM: ``shap.TreeExplainer`` on each of the 12 boosters (exact TreeSHAP).
- MLP: ``shap.GradientExplainer`` (expected gradients) with a training-set background.
- Ridge: exact linear SHAP, ``coef * (x - mean(background))`` (what ``shap.LinearExplainer`` computes
  for independent features).

Grouping sums |SHAP| over the members of a group (the 384 embedding dimensions become one
``plot_embedding`` value), then averages over movies. Permutation importance shuffles all columns of a
group together on the validation set and measures the MAE increase (model-agnostic cross-check).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vibespace_model.models import DnaModel  # noqa: E402

LOGGER = logging.getLogger(__name__)


def sample_rows(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Sorted random row positions (all rows when ``size >= n_rows``)."""
    if size >= n_rows:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=size, replace=False))


def _tree_shap(model: Any, X: np.ndarray) -> np.ndarray:
    import shap

    values = []
    for booster in model.boosters:
        explainer = shap.TreeExplainer(booster)
        values.append(np.asarray(explainer.shap_values(X, check_additivity=False)))
    return np.stack(values, axis=2)


def _gradient_shap(model: Any, X: np.ndarray, background: np.ndarray, seed: int, nsamples: int) -> np.ndarray:
    import shap
    import torch

    torch.manual_seed(seed)
    explainer = shap.GradientExplainer(model.torch_module(), torch.as_tensor(background, dtype=torch.float32))
    raw = explainer.shap_values(torch.as_tensor(X, dtype=torch.float32), nsamples=nsamples, rseed=seed)
    values = np.stack(raw, axis=2) if isinstance(raw, list) else np.asarray(raw)
    return values.reshape(len(X), X.shape[1], -1)


def linear_shap(coef: np.ndarray, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Exact SHAP of a linear model with independent features: ``coef[k] * (x - E[x])``."""
    centred = X - background.mean(axis=0)
    return centred[:, :, None] * coef.T[None, :, :]


def shap_values(
    model: DnaModel, X: np.ndarray, background: np.ndarray, seed: int, nsamples: int = 100
) -> np.ndarray:
    """SHAP values of shape (n_rows, n_features, n_outputs) for any supported family."""
    if model.name == "lightgbm":
        return _tree_shap(model, X)
    if model.name == "mlp":
        return _gradient_shap(model, X, background, seed, nsamples)
    if model.name == "ridge":
        return linear_shap(model.coef_, X, background)
    raise ValueError(f"no SHAP explainer for model type '{model.name}'")


def group_importance(values: np.ndarray, groups: dict[str, list[int]], targets: list[str]) -> pd.DataFrame:
    """Groups x dimensions: mean over movies of the summed |SHAP| of each group's features."""
    absolute = np.abs(values)
    table = {
        group: absolute[:, idx, :].sum(axis=1).mean(axis=0) if idx else np.zeros(values.shape[2])
        for group, idx in groups.items()
    }
    return pd.DataFrame(table, index=targets).T


def importance_shares(table: pd.DataFrame) -> pd.Series:
    """Overall importance per group (mean over dimensions), normalised to sum to 1."""
    overall = table.mean(axis=1)
    total = overall.sum()
    return overall / total if total > 0 else overall


def permutation_importance_by_group(
    model: DnaModel,
    X: np.ndarray,
    Y: np.ndarray,
    groups: dict[str, list[int]],
    seed: int,
    repeats: int = 5,
) -> pd.DataFrame:
    """Mean and std of the MAE increase when one group's columns are shuffled together."""
    base = float(np.abs(model.predict(X) - Y).mean())
    rng = np.random.default_rng(seed)
    rows = []
    for group, idx in groups.items():
        if not idx:
            rows.append({"group": group, "mae_increase": 0.0, "std": 0.0, "base_mae": base})
            continue
        increases = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, idx] = X[rng.permutation(len(X))][:, idx]
            increases.append(float(np.abs(model.predict(shuffled) - Y).mean()) - base)
        rows.append(
            {
                "group": group,
                "mae_increase": float(np.mean(increases)),
                "std": float(np.std(increases)),
                "base_mae": base,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def top_features(
    values: np.ndarray, names: list[str], exclude_prefix: str = "emb_", k: int = 15
) -> pd.DataFrame:
    """Metadata features ranked by mean |SHAP| over movies and dimensions."""
    keep = [i for i, n in enumerate(names) if not n.startswith(exclude_prefix)]
    mean_abs = np.abs(values[:, keep, :]).mean(axis=(0, 2))
    order = np.argsort(mean_abs)[::-1][:k]
    return pd.DataFrame({"feature": [names[keep[i]] for i in order], "mean_abs_shap": mean_abs[order]})


# ---------------------------------------------------------------------------- plots


def plot_group_importance(
    shares: dict[str, pd.Series], permutation: dict[str, pd.DataFrame], path: Path
) -> Path:
    """Side-by-side bars: SHAP share per group and permutation MAE increase per group, per model."""
    models = list(shares)
    fig, axes = plt.subplots(len(models), 2, figsize=(12, 3.6 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        s = shares[model].sort_values()
        axes[row, 0].barh(s.index, s.values, color="#4C72B0")
        axes[row, 0].set_title(f"{model}: share of mean |SHAP| by feature group (test sample)")
        p = permutation[model]["mae_increase"].reindex(s.index)
        axes[row, 1].barh(
            p.index, p.values, xerr=permutation[model]["std"].reindex(s.index).values, color="#DD8452"
        )
        axes[row, 1].set_title(f"{model}: permutation importance (validation MAE increase)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_heatmap(table: pd.DataFrame, path: Path, title: str) -> Path:
    """Feature groups x DNA dimensions heatmap of mean |SHAP| (each column normalised to sum 1)."""
    normalised = table / table.sum(axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(normalised.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(normalised.shape[0]):
        for j in range(normalised.shape[1]):
            ax.text(
                j, i, f"{normalised.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white"
            )
    fig.colorbar(image, ax=ax, label="share of |SHAP| within the dimension")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_metadata_summary(
    values: np.ndarray, X: np.ndarray, names: list[str], targets: list[str], dims: list[str], path: Path
) -> Path:
    """SHAP beeswarm of the metadata features (embedding excluded) for a few DNA dimensions."""
    import shap

    keep = [i for i, n in enumerate(names) if not n.startswith("emb_")]
    fig = plt.figure(figsize=(7 * len(dims), 7))
    for j, dim in enumerate(dims):
        fig.add_subplot(1, len(dims), j + 1)
        k = targets.index(dim)
        shap.summary_plot(
            values[:, keep, k],
            X[:, keep],
            feature_names=[names[i] for i in keep],
            max_display=15,
            show=False,
            plot_size=None,
            color_bar=j == len(dims) - 1,
        )
        plt.title(f"{dim}: metadata SHAP values")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close("all")
    return path
