"""Part 1: load data, build features, split, train all candidates, evaluate, select, register.

:func:`prepare` is shared by every part: it loads the verified data, creates (Part 1) or reloads
(Parts 2 and 3) the saved split, fits the feature builder on the training rows only and returns the
train / val / test matrices.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vibespace_model import tracking
from vibespace_model.config import ModelConfig
from vibespace_model.data_loader import Dataset, load_dataset
from vibespace_model.errors import SplitError
from vibespace_model.evaluate import (
    comparison_table,
    compute_metrics,
    per_dimension_table,
    plot_model_comparison,
    plot_per_dimension,
    plot_pred_vs_true,
    write_json,
)
from vibespace_model.features import FeatureBuilder
from vibespace_model.models import CANDIDATES, DnaModel, create_model, load_model
from vibespace_model.select import GateResult, build_model_card, save_bundle, select_best, validation_gate
from vibespace_model.splits import SPLITS, Split, load_split, make_split, save_split, slice_counts

LOGGER = logging.getLogger(__name__)
SCATTER_DIMS = ["energy", "humor", "darkness", "romance"]


@dataclass
class Prepared:
    """Everything a training step needs, aligned by split."""

    dataset: Dataset
    split: Split
    builder: FeatureBuilder
    X: dict[str, np.ndarray]
    Y: dict[str, np.ndarray]
    ids: dict[str, np.ndarray]
    catalog: dict[str, pd.DataFrame]
    slice_table: pd.DataFrame


def split_dir(cfg: ModelConfig) -> Path:
    """Where the Part 1 split assignment lives."""
    return cfg.artifacts_subdir("split")


def prepare(cfg: ModelConfig, skip_hash_check: bool = False, create_split: bool = False) -> Prepared:
    """Load data, create or reload the split, fit features on train only."""
    dataset = load_dataset(cfg, skip_hash_check=skip_hash_check)
    if create_split:
        split = make_split(
            dataset.movie_ids,
            dataset.catalog[cfg.split["stratify_on"]],
            {k: float(cfg.split[k]) for k in SPLITS},
            cfg.random_seed,
            int(cfg.split.get("min_stratum_size", 20)),
        )
        save_split(split, split_dir(cfg))
    else:
        split = load_split(split_dir(cfg))
        if set(split.assignment["movieId"]) != set(dataset.movie_ids):
            raise SplitError(
                "the saved split does not cover exactly the current movies; rerun `train` (Part 1)"
            )
    masks = split.masks(dataset.movie_ids)
    parts = {name: dataset.subset(mask) for name, mask in masks.items()}

    builder = FeatureBuilder(cfg.features, int(cfg.data["embedding_dim"]))
    builder.fit(parts["train"].catalog, parts["train"].embeddings)  # train rows only
    X = {name: builder.transform(p.catalog, p.embeddings) for name, p in parts.items()}
    slice_cols = [c for c in cfg.slices if c in dataset.catalog.columns]
    return Prepared(
        dataset=dataset,
        split=split,
        builder=builder,
        X=X,
        Y={name: p.labels for name, p in parts.items()},
        ids={name: p.movie_ids for name, p in parts.items()},
        catalog={name: p.catalog for name, p in parts.items()},
        slice_table=slice_counts(dataset.catalog, masks, slice_cols),
    )


def fit_model(
    name: str,
    params: dict[str, Any],
    prep: Prepared,
    seed: int,
    sample_weight: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    Y_train: np.ndarray | None = None,
) -> tuple[DnaModel, float]:
    """Train one candidate on train rows (validation only for early stopping); return model and seconds."""
    model = create_model(name, params, seed)
    start = time.perf_counter()
    model.fit(
        prep.X["train"] if X_train is None else X_train,
        prep.Y["train"] if Y_train is None else Y_train,
        prep.X["val"],
        prep.Y["val"],
        sample_weight=sample_weight,
    )
    return model, time.perf_counter() - start


def train_candidates(
    cfg: ModelConfig,
    prep: Prepared,
    params_by_model: dict[str, dict[str, Any]],
    log_to_mlflow: bool = True,
    run_prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Fit every candidate and compute its validation metrics (one nested MLflow run each)."""
    results: dict[str, dict[str, Any]] = {}
    for name, params in params_by_model.items():
        LOGGER.info("Training %s", name)
        model, seconds = fit_model(name, params, prep, cfg.random_seed)
        val = compute_metrics(prep.Y["val"], model.predict(prep.X["val"]), cfg.targets)
        results[name] = {"model": model, "val": val, "train_seconds": seconds, "params": model.get_params()}
        LOGGER.info(
            "%s: val MAE %.4f, R² %.3f (%.1fs)", name, val["overall"]["mae"], val["overall"]["r2"], seconds
        )
        if log_to_mlflow:
            with tracking.run(f"{run_prefix}{name}", nested=True, tags={"model_type": name}):
                tracking.log_params(model.get_params())
                tracking.log_params(model.fit_info, prefix="fit_")
                tracking.log_metrics({**tracking.flatten_metrics(val, "val"), "train_seconds": seconds})
    return results


def default_params(cfg: ModelConfig) -> dict[str, dict[str, Any]]:
    """Config defaults of every candidate, in the fixed candidate order."""
    return {name: cfg.model_params(name) for name in CANDIDATES if name in cfg.models}


def write_comparison_reports(
    cfg: ModelConfig, results: dict[str, dict[str, Any]], out: Path, title_suffix: str = ""
) -> dict[str, Path]:
    """Comparison CSV/JSON + bar chart + per-dimension chart for a set of results."""
    table = comparison_table(results)
    table.to_csv(out / "model_comparison.csv", index=False, lineterminator="\n")
    per_dim = per_dimension_table(results)
    per_dim.to_csv(out / "per_dimension_val_mae.csv", lineterminator="\n")
    return {
        "comparison_csv": out / "model_comparison.csv",
        "per_dimension_csv": out / "per_dimension_val_mae.csv",
        "comparison_png": plot_model_comparison(
            table, out / "val_mae_by_model.png", f"Validation MAE by model{title_suffix}"
        ),
        "per_dimension_png": plot_per_dimension(
            per_dim, out / "val_mae_per_dimension.png", f"Validation MAE per DNA dimension{title_suffix}"
        ),
    }


def serialisable(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Results without the model objects (for JSON)."""
    return {name: {k: v for k, v in res.items() if k != "model"} for name, res in results.items()}


def run_part1(cfg: ModelConfig, skip_hash_check: bool = False) -> dict[str, Any]:
    """Train all candidates with default parameters, select by validation MAE, gate, save and register."""
    reports, artifacts = cfg.reports_subdir("part1"), cfg.artifacts_subdir("part1")
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=True)
    tracking.setup_tracking(cfg)
    tags = tracking.session_tags(cfg, prep.dataset.data_version, prep.split.split_hash, "part1")

    with tracking.run("part1-train", tags=tags) as parent:
        tracking.log_params({"n_" + k: len(v) for k, v in prep.ids.items()})
        tracking.log_params(cfg.gate, prefix="gate_")
        results = train_candidates(cfg, prep, default_params(cfg))

        selected = select_best({n: r["val"]["overall"] for n, r in results.items()})
        gate = validation_gate(
            results[selected]["val"]["overall"], results["baseline"]["val"]["overall"], cfg.gate
        )
        model = results[selected]["model"]
        test = compute_metrics(prep.Y["test"], model.predict(prep.X["test"]), cfg.targets)  # report only

        for name, res in results.items():
            res["model"].save(artifacts / "models" / name)
        prep.builder.save(artifacts / "models" / "features.json")
        card = build_model_card(
            model=model,
            stage="part1-candidate",
            data_version=prep.dataset.data_version,
            split_hash=prep.split.split_hash,
            metrics={"val": results[selected]["val"], "test": test},
            gate_cfg=cfg.gate,
            validation_gate_result=gate,
            extra={
                "targets": cfg.targets,
                "feature_groups": {g: len(i) for g, i in prep.builder.groups.items()},
            },
        )
        bundle = save_bundle(artifacts / "selected", model, prep.builder, card)

        files = write_comparison_reports(cfg, results, reports)
        files["scatter_png"] = plot_pred_vs_true(
            prep.Y["val"],
            model.predict(prep.X["val"]),
            cfg.targets,
            SCATTER_DIMS,
            reports / "pred_vs_true_val.png",
            f"{selected}: predicted vs true (validation)",
        )
        prep.slice_table.to_csv(reports / "split_slice_counts.csv", index=False, lineterminator="\n")
        summary = {
            "part": 1,
            "join_report": prep.dataset.join_report,
            "data_version": prep.dataset.data_version,
            "split_hash": prep.split.split_hash,
            "split_counts": {k: int(len(v)) for k, v in prep.ids.items()},
            "results": serialisable(results),
            "selected": selected,
            "selected_test": test,
            "validation_gate": gate.to_dict(),
            "mlflow_parent_run": parent.info.run_id,
        }
        write_json(reports / "part1_summary.json", summary)
        tracking.log_metrics(
            {
                "selected_val_mae": results[selected]["val"]["overall"]["mae"],
                "selected_test_mae": test["overall"]["mae"],
                "gate_passed": float(gate.passed),
            }
        )
        tracking.log_artifacts(
            [reports / "part1_summary.json", *files.values(), reports / "split_slice_counts.csv"],
            artifact_path="reports",
        )
        registration = tracking.register_bundle(
            cfg, bundle, alias="candidate", description=f"Part 1 selection ({selected}, default params)"
        )
        summary["registration"] = registration
        write_json(reports / "part1_summary.json", summary)
    LOGGER.info("Part 1: selected %s, gate %s", selected, "PASSED" if gate.passed else "FAILED")
    return summary


def load_part1_models(cfg: ModelConfig) -> dict[str, DnaModel]:
    """Saved Part 1 candidates (for the ``evaluate`` command)."""
    base = cfg.artifacts_subdir("part1") / "models"
    models = {name: load_model(base / name) for name in CANDIDATES if (base / name).is_dir()}
    if not models:
        raise FileNotFoundError(f"no saved Part 1 models in {base}; run `train` first")
    return models


def run_evaluate(cfg: ModelConfig, skip_hash_check: bool = False) -> dict[str, Any]:
    """Re-evaluate the saved Part 1 candidates on validation and rewrite the comparison reports."""
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=False)
    models = load_part1_models(cfg)
    results = {
        name: {
            "model": m,
            "val": compute_metrics(prep.Y["val"], m.predict(prep.X["val"]), cfg.targets),
            "train_seconds": None,
        }
        for name, m in models.items()
    }
    out = cfg.reports_subdir("part1") / "evaluate"
    out.mkdir(parents=True, exist_ok=True)
    write_comparison_reports(cfg, results, out)
    summary = {"results": serialisable(results), "split_hash": prep.split.split_hash}
    write_json(out / "evaluate_summary.json", summary)
    return summary


def gate_from_summary(summary: dict[str, Any]) -> GateResult:
    """Rebuild the gate result stored in a summary JSON."""
    gate = summary["validation_gate"]
    return GateResult(gate["name"], gate["passed"], gate["reasons"], gate["checks"])
