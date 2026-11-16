"""Orchestration of Parts 2 and 3 (Part 1 lives in :mod:`vibespace_model.train`).

Each workflow reloads the saved Part 1 split, writes its reports under ``reports/part<N>/``, logs a
parent MLflow run and returns a JSON-serialisable summary. Selection always uses validation metrics;
the test split is used only after a decision, for reporting.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from vibespace_model import bias as bias_mod
from vibespace_model import explain as explain_mod
from vibespace_model import tracking
from vibespace_model.config import ModelConfig
from vibespace_model.errors import VibeSpaceModelError
from vibespace_model.evaluate import (
    compute_metrics,
    improvement_over_baseline,
    plot_model_comparison,
    plot_per_dimension,
    write_json,
)
from vibespace_model.models import DnaModel, load_model
from vibespace_model.select import (
    GateResult,
    build_model_card,
    final_selection,
    load_bundle,
    save_bundle,
    select_best,
    validation_gate,
)
from vibespace_model.sensitivity import (
    param_importances,
    plot_importances,
    plot_sweeps,
    run_sweeps,
    sweep_summary,
)
from vibespace_model.train import Prepared, fit_model, prepare
from vibespace_model.tuning import (
    TUNABLE,
    TuningData,
    best_params,
    load_best_params,
    parse_search_space,
    run_tuning,
    save_best_params,
    storage_url,
    study_name,
    study_summary,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- shared helpers


def read_summary(cfg: ModelConfig, part: str) -> dict[str, Any]:
    """Summary JSON written by an earlier part (clear error if that part has not run)."""
    path = cfg.reports_subdir(part) / f"{part}_summary.json"
    if not path.is_file():
        command = {"part1": "train", "part2": "tune", "part3": "bias"}.get(part, part)
        raise VibeSpaceModelError(f"{path} not found; run the `{command}` command first")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_from_dict(data: dict[str, Any]) -> GateResult:
    return GateResult(data["name"], data["passed"], data["reasons"], data["checks"])


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Small Markdown table writer (no extra dependency)."""

    def cell(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return format(float(value), floatfmt)
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def _optuna_plots(study: optuna.Study, out: Path, model_name: str) -> list[Path]:
    """Optimisation history, parameter importance and parallel-coordinate plots of one study."""
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as optuna_plt

    paths = []
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    plots = [("optimization_history", optuna_plt.plot_optimization_history)]
    if len(complete) >= 2:
        plots += [
            ("param_importances", optuna_plt.plot_param_importances),
            ("parallel_coordinate", optuna_plt.plot_parallel_coordinate),
        ]
    for label, func in plots:
        try:
            ax = func(study)
            fig = ax if hasattr(ax, "savefig") else np.ravel(ax)[0].figure
            fig.set_size_inches(9, 5)
            fig.suptitle(f"{model_name}: {label.replace('_', ' ')}")
            path = out / f"optuna_{model_name}_{label}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            paths.append(path)
        except (ValueError, RuntimeError) as exc:  # pragma: no cover - plotting edge cases
            LOGGER.warning("Optuna plot %s for %s failed: %s", label, model_name, exc)
    return paths


# ---------------------------------------------------------------------------- Part 2: tuning


def run_tune(
    cfg: ModelConfig, skip_hash_check: bool = False, tuned_params_path: Path | None = None
) -> dict[str, Any]:
    """Tune (or reuse tuned params), retrain best configs, compare with defaults, select, gate, register."""
    part1 = read_summary(cfg, "part1")
    reports, artifacts = cfg.reports_subdir("part2"), cfg.artifacts_subdir("part2")
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=False)
    if prep.split.split_hash != part1["split_hash"]:
        raise VibeSpaceModelError(
            "split hash differs from Part 1; rerun `train` so all parts share one split"
        )
    data = TuningData.from_prepared(prep)  # asserts: no test movie reaches tuning
    tracking.setup_tracking(cfg)
    tags = tracking.session_tags(cfg, prep.dataset.data_version, prep.split.split_hash, "part2")
    models = [m for m in cfg.tuning.get("model_order", TUNABLE) if m in cfg.tuning.get("models", {})]

    with tracking.run(
        "part2-tune", tags={**tags, "mode": "reuse" if tuned_params_path else "search"}
    ) as parent:
        start = time.perf_counter()
        studies: dict[str, optuna.Study] = {}
        if tuned_params_path:
            tuned = {m: p for m, p in load_best_params(tuned_params_path).items() if m in models}
            LOGGER.info("Reusing tuned parameters from %s (no Optuna search)", tuned_params_path)
        else:
            studies = run_tuning(cfg, data)
            tuned = {
                m: best_params(s, m)
                for m, s in studies.items()
                if any(t.state == optuna.trial.TrialState.COMPLETE for t in s.trials)
            }
        tuning_seconds = time.perf_counter() - start
        save_best_params(reports / "best_params.json", tuned)

        tuned_results: dict[str, dict[str, Any]] = {}
        for name, params in tuned.items():
            model, seconds = fit_model(name, params, prep, cfg.random_seed)
            val = compute_metrics(prep.Y["val"], model.predict(prep.X["val"]), cfg.targets)
            tuned_results[name] = {
                "model": model,
                "val": val,
                "train_seconds": seconds,
                "params": model.get_params(),
            }
            model.save(artifacts / "tuned" / name)
            with tracking.run(f"tuned-{name}", nested=True, tags={"model_type": name, "variant": "tuned"}):
                tracking.log_params(model.get_params())
                tracking.log_metrics({**tracking.flatten_metrics(val, "val"), "train_seconds": seconds})

        variants: dict[str, dict[str, Any]] = {}
        for name, res in part1["results"].items():
            variants[f"{name}_default"] = {"family": name, "variant": "default", "val": res["val"]}
        for name, res in tuned_results.items():
            variants[f"{name}_tuned"] = {"family": name, "variant": "tuned", "val": res["val"]}
        baseline_val = part1["results"]["baseline"]["val"]["overall"]
        selected = select_best({k: v["val"]["overall"] for k, v in variants.items()})
        family, variant = variants[selected]["family"], variants[selected]["variant"]
        model = (
            tuned_results[family]["model"]
            if variant == "tuned"
            else load_model(cfg.artifacts_subdir("part1") / "models" / family)
        )
        gate = validation_gate(variants[selected]["val"]["overall"], baseline_val, cfg.gate)

        test = compute_metrics(prep.Y["test"], model.predict(prep.X["test"]), cfg.targets)  # after selection
        card = build_model_card(
            model=model,
            stage="part2-candidate",
            data_version=prep.dataset.data_version,
            split_hash=prep.split.split_hash,
            metrics={"val": variants[selected]["val"], "test": test},
            gate_cfg=cfg.gate,
            validation_gate_result=gate,
            extra={"targets": cfg.targets, "variant": variant, "selected_from": sorted(variants)},
        )
        bundle = save_bundle(artifacts / "final_candidate", model, prep.builder, card)

        files = _part2_reports(cfg, reports, variants, selected, prep, model, test, studies)
        summary = {
            "part": 2,
            "split_hash": prep.split.split_hash,
            "mode": "reuse" if tuned_params_path else "search",
            "tuning_seconds": tuning_seconds,
            "studies": {m: study_summary(s) for m, s in studies.items()},
            "best_params": tuned,
            "variants": variants,
            "tuned_train_seconds": {m: r["train_seconds"] for m, r in tuned_results.items()},
            "selected": selected,
            "selected_family": family,
            "selected_variant": variant,
            "validation_gate": gate.to_dict(),
            "selected_test": test,
            "test_split_used_for_tuning": False,
            "mlflow_parent_run": parent.info.run_id,
        }
        tracking.log_metrics(
            {
                "tuning_seconds": tuning_seconds,
                "selected_val_mae": variants[selected]["val"]["overall"]["mae"],
                "selected_test_mae": test["overall"]["mae"],
                "gate_passed": float(gate.passed),
            }
        )
        write_json(reports / "part2_summary.json", summary)
        tracking.log_artifacts(
            [reports / "part2_summary.json", reports / "best_params.json", *files], artifact_path="reports"
        )
        summary["registration"] = tracking.register_bundle(
            cfg, bundle, alias="candidate", description=f"Part 2 selection ({selected})"
        )
        write_json(reports / "part2_summary.json", summary)
    LOGGER.info("Part 2: final candidate %s, gate %s", selected, "PASSED" if gate.passed else "FAILED")
    return summary


def _part2_reports(
    cfg: ModelConfig,
    out: Path,
    variants: dict[str, dict[str, Any]],
    selected: str,
    prep: Prepared,
    model: DnaModel,
    test: dict[str, Any],
    studies: dict[str, optuna.Study],
) -> list[Path]:
    import matplotlib.pyplot as plt

    files: list[Path] = []
    for name, study in studies.items():
        files += _optuna_plots(study, out, name)

    baseline_mae = variants["baseline_default"]["val"]["overall"]["mae"]
    rows = []
    for key, res in variants.items():
        overall = res["val"]["overall"]
        rows.append(
            {
                "variant": key,
                "family": res["family"],
                "kind": res["variant"],
                "val_mae": overall["mae"],
                "val_rmse": overall["rmse"],
                "val_r2": overall["r2"],
                "val_spearman": overall["spearman"],
                "improvement_vs_baseline": improvement_over_baseline(overall["mae"], baseline_mae),
            }
        )
    table = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    table.to_csv(out / "default_vs_tuned.csv", index=False, lineterminator="\n")
    files.append(out / "default_vs_tuned.csv")

    families = [f for f in ("ridge", "lightgbm", "mlp") if f"{f}_default" in variants]
    fig, ax = plt.subplots(figsize=(8, 4))
    width, x = 0.38, np.arange(len(families))
    default = [variants[f"{f}_default"]["val"]["overall"]["mae"] for f in families]
    tuned = [variants.get(f"{f}_tuned", variants[f"{f}_default"])["val"]["overall"]["mae"] for f in families]
    bars1 = ax.bar(x - width / 2, default, width, label="default (Part 1)")
    bars2 = ax.bar(x + width / 2, tuned, width, label="tuned (Part 2)")
    ax.bar_label(bars1, fmt="%.4f", fontsize=7)
    ax.bar_label(bars2, fmt="%.4f", fontsize=7)
    ax.axhline(baseline_mae, color="black", linestyle=":", label=f"mean baseline ({baseline_mae:.4f})")
    ax.set_xticks(x, families)
    ax.set_ylabel("mean validation MAE")
    ax.set_ylim(min(default + tuned) * 0.95, baseline_mae * 1.02)
    ax.set_title("Default vs tuned validation MAE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "default_vs_tuned.png", dpi=120)
    plt.close(fig)
    files.append(out / "default_vs_tuned.png")

    per_dim = pd.DataFrame(
        {
            "baseline": {
                d: v["mae"] for d, v in variants["baseline_default"]["val"]["per_dimension"].items()
            },
            selected: {d: v["mae"] for d, v in variants[selected]["val"]["per_dimension"].items()},
        }
    )
    per_dim.to_csv(out / "final_candidate_per_dimension_val_mae.csv", lineterminator="\n")
    files.append(
        plot_per_dimension(
            per_dim,
            out / "final_candidate_per_dimension_val_mae.png",
            f"Per-dimension validation MAE: {selected} vs baseline",
        )
    )
    files.append(out / "final_candidate_per_dimension_val_mae.csv")

    val_test = pd.DataFrame(
        [
            {
                "split": "validation",
                **{m: variants[selected]["val"]["overall"][m] for m in ("mae", "rmse", "r2", "spearman")},
            },
            {"split": "test", **{m: test["overall"][m] for m in ("mae", "rmse", "r2", "spearman")}},
        ]
    )
    val_test.to_csv(out / "final_candidate_val_vs_test.csv", index=False, lineterminator="\n")
    files.append(out / "final_candidate_val_vs_test.csv")
    files.append(
        plot_model_comparison(
            table.rename(columns={"variant": "model"})[["model", "val_mae"]].head(8),
            out / "variants_val_mae.png",
            "Validation MAE of all variants (best 8)",
        )
    )
    return files


# ---------------------------------------------------------------------------- Part 2: sensitivity


def run_sensitivity(cfg: ModelConfig, skip_hash_check: bool = False) -> dict[str, Any]:
    """fANOVA importance per tuned model + one-at-a-time sweeps for the final candidate's family."""
    part2 = read_summary(cfg, "part2")
    out = cfg.reports_subdir("part2") / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    if part2["mode"] == "reuse":
        LOGGER.warning("Part 2 ran with --tuned-params (no studies); sensitivity analysis skipped")
        summary = {"skipped": True, "reason": "no Optuna studies (tuned parameters were reused)"}
        write_json(out / "sensitivity_summary.json", summary)
        return summary

    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=False)
    data = TuningData.from_prepared(prep)
    storage = storage_url(cfg)
    studies = {
        m: optuna.load_study(
            study_name=study_name(m, prep.split.split_hash, cfg.random_seed), storage=storage
        )
        for m in part2["studies"]
    }
    sens_cfg = cfg.tuning.get("sensitivity", {})
    importances = {
        m: param_importances(s, cfg.random_seed, sens_cfg.get("importance_evaluator", "fanova"))
        for m, s in studies.items()
    }
    pd.DataFrame(importances).to_csv(out / "param_importances.csv", lineterminator="\n")
    files = [out / "param_importances.csv", plot_importances(importances, out / "param_importances.png")]

    family = part2["selected_family"]
    summary: dict[str, Any] = {"importances": importances, "final_family": family}
    if family not in studies:
        summary["sweeps"] = None
        summary["reason"] = f"final candidate family '{family}' was not tuned; no sweep"
    else:
        specs = parse_search_space(cfg.tuning["models"][family]["search_space"])
        ranked = sorted(importances[family], key=importances[family].get, reverse=True) or list(specs)
        top = ranked[: int(sens_cfg.get("top_k", 3))]
        fixed = dict(studies[family].user_attrs.get("fixed_params", {}))
        tracking.setup_tracking(cfg)
        with tracking.run(
            "part2-sensitivity",
            tags={"part": "part2", "model_type": family, "split_hash": prep.split.split_hash},
        ):
            sweeps = run_sweeps(
                family,
                dict(studies[family].best_params),
                fixed,
                specs,
                top,
                data,
                cfg.random_seed,
                cfg.targets,
                int(sens_cfg.get("points", 7)),
            )
            sweeps.to_csv(out / f"sweeps_{family}.csv", index=False, lineterminator="\n")
            spread = sweep_summary(sweeps)
            spread.to_csv(out / f"sweep_summary_{family}.csv", index=False, lineterminator="\n")
            files += [
                out / f"sweeps_{family}.csv",
                out / f"sweep_summary_{family}.csv",
                plot_sweeps(
                    sweeps,
                    specs,
                    out / f"sweeps_{family}.png",
                    f"{family}: validation MAE when one hyperparameter changes",
                ),
            ]
            tracking.log_params({"swept": ",".join(top)})
            tracking.log_metrics({f"spread_{r.param}": r.spread for r in spread.itertuples()})
            tracking.log_artifacts(files, artifact_path="sensitivity")
        summary.update(
            {
                "swept_params": top,
                "sweeps": sweeps.to_dict(orient="records"),
                "sweep_summary": spread.to_dict(orient="records"),
            }
        )
    write_json(out / "sensitivity_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------- Part 3: explainability


def _variant_model(cfg: ModelConfig, family: str, variant: str) -> DnaModel:
    base = (
        cfg.artifacts_subdir("part2") / "tuned"
        if variant == "tuned"
        else cfg.artifacts_subdir("part1") / "models"
    )
    return load_model(base / family)


def _ranked_variants(part2: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return sorted(part2["variants"].items(), key=lambda kv: kv[1]["val"]["overall"]["mae"])


def run_explain(cfg: ModelConfig, skip_hash_check: bool = False) -> dict[str, Any]:
    """SHAP (grouped) for the final candidate and the next-best other family, plus permutation importance."""
    part2 = read_summary(cfg, "part2")
    out = cfg.reports_subdir("part3")
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=False)
    final_model, builder, _ = load_bundle(cfg.artifacts_subdir("part2") / "final_candidate")
    if builder.state != json.loads(json.dumps(prep.builder.state)):
        raise VibeSpaceModelError("saved preprocessing differs from the one rebuilt on the Part 1 split")

    explain_targets = {part2["selected_family"]: final_model}
    for _, res in _ranked_variants(part2):
        if res["family"] not in explain_targets and res["family"] != "baseline":
            explain_targets[res["family"]] = _variant_model(cfg, res["family"], res["variant"])
            break

    n_sample = int(cfg.explain.get("sample_size", 1000))
    n_background = int(cfg.explain.get("background_size", 200))
    rows = explain_mod.sample_rows(len(prep.X["test"]), n_sample, cfg.random_seed)
    background = prep.X["train"][explain_mod.sample_rows(len(prep.X["train"]), n_background, cfg.random_seed)]
    X_explain = prep.X["test"][rows]
    names, groups = prep.builder.feature_names, prep.builder.groups

    summary: dict[str, Any] = {"n_explained": len(rows), "models": {}}
    shares, permutation, files = {}, {}, []
    tracking.setup_tracking(cfg)
    with tracking.run("part3-explain", tags={"part": "part3", "split_hash": prep.split.split_hash}):
        for family, model in explain_targets.items():
            start = time.perf_counter()
            values = explain_mod.shap_values(
                model, X_explain, background, cfg.random_seed, int(cfg.explain.get("gradient_nsamples", 100))
            )
            table = explain_mod.group_importance(values, groups, cfg.targets)
            share = explain_mod.importance_shares(table)
            perm = explain_mod.permutation_importance_by_group(
                model,
                prep.X["val"],
                prep.Y["val"],
                groups,
                cfg.random_seed,
                int(cfg.explain.get("permutation_repeats", 5)),
            )
            top_meta = explain_mod.top_features(values, names)
            table.to_csv(out / f"shap_group_by_dimension_{family}.csv", lineterminator="\n")
            perm.to_csv(out / f"permutation_importance_{family}.csv", lineterminator="\n")
            top_meta.to_csv(out / f"shap_top_metadata_{family}.csv", index=False, lineterminator="\n")
            files += [
                out / f"shap_group_by_dimension_{family}.csv",
                out / f"permutation_importance_{family}.csv",
                out / f"shap_top_metadata_{family}.csv",
                explain_mod.plot_heatmap(
                    table,
                    out / f"shap_heatmap_{family}.png",
                    f"{family}: share of |SHAP| per feature group and DNA dimension",
                ),
                explain_mod.plot_metadata_summary(
                    values,
                    X_explain,
                    names,
                    cfg.targets,
                    ["humor", "darkness"],
                    out / f"shap_metadata_{family}.png",
                ),
            ]
            shares[family], permutation[family] = share, perm
            summary["models"][family] = {
                "shap_share": share.to_dict(),
                "shap_by_dimension": table.to_dict(),
                "permutation_mae_increase": perm["mae_increase"].to_dict(),
                "top_metadata_features": top_meta.head(10).to_dict(orient="records"),
                "top_groups": share.sort_values(ascending=False).index.tolist()[:3],
                "seconds": time.perf_counter() - start,
            }
            tracking.log_metrics({f"{family}_shap_share_{g}": v for g, v in share.items()})
        files.append(explain_mod.plot_group_importance(shares, permutation, out / "grouped_importance.png"))
        write_json(out / "explain_summary.json", summary)
        tracking.log_artifacts([*files, out / "explain_summary.json"], artifact_path="explain")
    return summary


# ---------------------------------------------------------------------------- Part 3: bias + final selection


def _slice_entry(
    cfg: ModelConfig,
    name: str,
    family: str,
    model: DnaModel,
    prep: Prepared,
    baseline_val: dict[str, float],
    kind: str,
) -> dict[str, Any]:
    val = compute_metrics(prep.Y["val"], model.predict(prep.X["val"]), cfg.targets)
    columns = list(cfg.bias.get("slice_columns", cfg.slices))
    table, overall = bias_mod.evaluate_slices(
        prep.Y["val"], model.predict(prep.X["val"]), prep.catalog["val"], columns, cfg.bias, cfg.random_seed
    )
    return {
        "name": name,
        "family": family,
        "kind": kind,
        "model": model,
        "val": val,
        "val_mae": val["overall"]["mae"],
        "val_spearman": val["overall"]["spearman"],
        "validation_gate": validation_gate(val["overall"], baseline_val, cfg.gate),
        "bias_gate": bias_mod.bias_gate(table, overall, cfg.bias),
        "bias_gap": bias_mod.worst_ratio(table),
        "slice_table": table,
    }


def run_bias(cfg: ModelConfig, skip_hash_check: bool = False) -> dict[str, Any]:
    """Slice metrics + bias gate for every tuned family, mitigation experiments, final selection, registry."""
    part1, part2 = read_summary(cfg, "part1"), read_summary(cfg, "part2")
    out, artifacts = cfg.reports_subdir("part3"), cfg.artifacts_subdir("part3")
    prep = prepare(cfg, skip_hash_check=skip_hash_check, create_split=False)
    baseline_val = part1["results"]["baseline"]["val"]["overall"]
    tracking.setup_tracking(cfg)
    tags = tracking.session_tags(cfg, prep.dataset.data_version, prep.split.split_hash, "part3")

    with tracking.run("part3-bias", tags=tags) as parent:
        best_by_family: dict[str, tuple[str, dict[str, Any]]] = {}
        for key, res in _ranked_variants(part2):
            if res["family"] != "baseline" and res["family"] not in best_by_family:
                best_by_family[res["family"]] = (key, res)
        entries = [
            _slice_entry(
                cfg, key, fam, _variant_model(cfg, fam, res["variant"]), prep, baseline_val, res["variant"]
            )
            for fam, (key, res) in best_by_family.items()
        ]

        family = part2["selected_family"]
        candidate = next(e for e in entries if e["family"] == family)
        column = bias_mod.choose_mitigation_column(
            candidate["slice_table"], cfg.bias.get("mitigation", {}).get("column", "auto")
        )
        params = {k: v for k, v in candidate["model"].get_params().items() if k != "seed"}
        train_values = bias_mod.slice_values(prep.catalog["train"], column)
        mit = cfg.bias.get("mitigation", {})
        weights = bias_mod.reweighting_weights(
            train_values,
            float(mit.get("reweighting", {}).get("power", 1.0)),
            float(mit.get("reweighting", {}).get("max_weight", 20.0)),
        )
        reweighted, _ = fit_model(family, params, prep, cfg.random_seed, sample_weight=weights)
        idx = bias_mod.resample_indices(
            train_values, float(mit.get("resampling", {}).get("min_share_of_largest", 0.1)), cfg.random_seed
        )
        resampled, _ = fit_model(
            family, params, prep, cfg.random_seed, X_train=prep.X["train"][idx], Y_train=prep.Y["train"][idx]
        )
        entries.append(
            _slice_entry(cfg, f"{family}_reweighted", family, reweighted, prep, baseline_val, "reweighted")
        )
        entries.append(
            _slice_entry(cfg, f"{family}_resampled", family, resampled, prep, baseline_val, "resampled")
        )

        for entry in entries:
            with tracking.run(
                f"bias-{entry['name']}",
                nested=True,
                tags={"model_type": entry["family"], "variant": entry["kind"]},
            ):
                tracking.log_metrics(
                    {
                        "val_mae": entry["val_mae"],
                        "bias_gap": entry["bias_gap"],
                        "validation_gate_passed": float(entry["validation_gate"].passed),
                        "bias_gate_passed": float(entry["bias_gate"].passed),
                    }
                )
            entry["slice_table"].to_csv(
                out / f"slices_val_{entry['name']}.csv", index=False, lineterminator="\n"
            )

        decision = final_selection(entries)
        final = next(e for e in entries if e["name"] == decision["name"])
        model = final["model"]
        test_pred = model.predict(prep.X["test"])
        test = compute_metrics(prep.Y["test"], test_pred, cfg.targets)
        test_slices, test_overall = bias_mod.evaluate_slices(
            prep.Y["test"],
            test_pred,
            prep.catalog["test"],
            list(cfg.bias.get("slice_columns", cfg.slices)),
            cfg.bias,
            cfg.random_seed,
        )
        test_gate = bias_mod.bias_gate(test_slices, test_overall, cfg.bias)
        test_slices.to_csv(out / "slices_test_final.csv", index=False, lineterminator="\n")
        final["slice_table"].to_csv(out / "slices_val_final.csv", index=False, lineterminator="\n")

        explain_path = out / "explain_summary.json"
        shap_top = {}
        if explain_path.is_file():
            explain = json.loads(explain_path.read_text(encoding="utf-8"))
            shap_top = {m: v["top_groups"] for m, v in explain.get("models", {}).items()}
        mitigation = _mitigation_table(entries, family)
        mitigation.to_csv(out / "mitigation_comparison.csv", index=False, lineterminator="\n")
        card = build_model_card(
            model=model,
            stage="part3-final",
            data_version=prep.dataset.data_version,
            split_hash=prep.split.split_hash,
            metrics={"val": final["val"], "test": test},
            gate_cfg=cfg.gate,
            validation_gate_result=final["validation_gate"],
            bias_gate_result=final["bias_gate"],
            extra={
                "targets": cfg.targets,
                "variant": final["kind"],
                "selection": decision,
                "thresholds": {
                    "validation_gate": cfg.gate,
                    "bias_gate": {k: cfg.bias[k] for k in ("min_gated_slice_size", "max_mae_ratio")},
                },
                "slice_summary_val": bias_mod.gap_by_column(final["slice_table"]).to_dict(orient="records"),
                "bias_gate_test": test_gate.to_dict(),
                "shap_top_groups": shap_top,
                "mitigation_column": column,
            },
        )
        bundle = save_bundle(artifacts / "final_model", model, prep.builder, card)
        files = _part3_reports(cfg, out, entries, final, test, test_slices, test_overall, column, decision)

        summary = {
            "part": 3,
            "split_hash": prep.split.split_hash,
            "entries": [
                {
                    k: (v.to_dict() if isinstance(v, GateResult) else v)
                    for k, v in e.items()
                    if k not in ("model", "slice_table")
                }
                for e in entries
            ],
            "mitigation_column": column,
            "decision": decision,
            "final": {
                "name": final["name"],
                "family": final["family"],
                "variant": final["kind"],
                "params": model.get_params(),
                "val": final["val"],
                "test": test,
                "validation_gate": final["validation_gate"].to_dict(),
                "bias_gate": final["bias_gate"].to_dict(),
                "bias_gate_test": test_gate.to_dict(),
            },
            "mlflow_parent_run": parent.info.run_id,
        }
        write_json(out / "part3_summary.json", summary)
        tracking.log_metrics(
            {
                "final_val_mae": final["val_mae"],
                "final_test_mae": test["overall"]["mae"],
                "final_bias_gap": final["bias_gap"],
            }
        )
        tracking.log_artifacts([out / "part3_summary.json", *files], artifact_path="reports")
        summary["registration"] = tracking.register_bundle(
            cfg, bundle, alias="champion-candidate", description=f"Part 3 final model ({final['name']})"
        )
        write_json(out / "part3_summary.json", summary)
    LOGGER.info("Part 3: final model %s (%s)", final["name"], decision["reason"])
    return summary


def _mitigation_table(entries: list[dict[str, Any]], family: str) -> pd.DataFrame:
    rows = []
    for e in entries:
        if e["family"] != family:
            continue
        gaps = bias_mod.gap_by_column(e["slice_table"])
        rows.append(
            {
                "model": e["name"],
                "kind": e["kind"],
                "val_mae": e["val_mae"],
                "val_r2": e["val"]["overall"]["r2"],
                "worst_slice_ratio": e["bias_gap"],
                "worst_slice": e["bias_gate"].checks["worst_slice"],
                "max_worst_over_best": float(gaps["worst_over_best"].max()) if not gaps.empty else 1.0,
                "validation_gate": "pass" if e["validation_gate"].passed else "fail",
                "bias_gate": "pass" if e["bias_gate"].passed else "fail",
            }
        )
    return pd.DataFrame(rows)


def _part3_reports(
    cfg: ModelConfig,
    out: Path,
    entries: list[dict[str, Any]],
    final: dict[str, Any],
    test: dict[str, Any],
    test_slices: pd.DataFrame,
    test_overall: float,
    column: str,
    decision: dict[str, Any],
) -> list[Path]:
    max_ratio = float(cfg.bias["max_mae_ratio"])
    files = [
        bias_mod.plot_slice_mae(
            final["slice_table"],
            final["val_mae"],
            max_ratio,
            out / "slice_mae_val_final.png",
            f"{final['name']}: validation MAE per slice",
        ),
        bias_mod.plot_slice_mae(
            test_slices,
            test_overall,
            max_ratio,
            out / "slice_mae_test_final.png",
            f"{final['name']}: test MAE per slice",
        ),
        out / "mitigation_comparison.csv",
        out / "slices_val_final.csv",
        out / "slices_test_final.csv",
    ]
    family = final["family"]
    mitigation = _mitigation_table(entries, family)
    by_column = pd.concat(
        [
            e["slice_table"]
            .loc[e["slice_table"]["slice_column"] == column, ["slice", "n", "mae"]]
            .rename(columns={"mae": e["kind"]})
            .set_index(["slice", "n"])
            for e in entries
            if e["family"] == family
        ],
        axis=1,
    ).reset_index()

    gate_rows = pd.DataFrame(
        [
            {
                "model": e["name"],
                "val_mae": e["val_mae"],
                "validation_gate": "pass" if e["validation_gate"].passed else "fail",
                "bias_gate": "pass" if e["bias_gate"].passed else "fail",
                "worst_slice": e["bias_gate"].checks["worst_slice"],
                "worst_ratio": e["bias_gap"],
            }
            for e in entries
        ]
    )
    small = final["slice_table"][~final["slice_table"]["gated"]][
        ["slice_column", "slice", "n", "mae", "ci_low", "ci_high"]
    ]
    gated = final["slice_table"][final["slice_table"]["gated"]][
        ["slice_column", "slice", "n", "mae", "mae_ratio", "status"]
    ]
    test_view = test_slices[["slice_column", "slice", "n", "mae", "mae_ratio", "status"]]
    lines = [
        "# Bias report: Movie DNA predictor (Part 3)",
        "",
        "Written by `python -m vibespace_model.cli bias`. Slices are movie attributes; MovieLens and "
        "the app hold no user demographics. Gate: slices with at least "
        f"{cfg.bias['min_gated_slice_size']} movies must have MAE <= {max_ratio} x overall MAE (validation "
        "split decides; test split is reported). Smaller slices are reported with a bootstrap "
        f"{int(float(cfg.bias['bootstrap']['confidence']) * 100)}% CI and marked 'insufficient data'.",
        "",
        "## Candidates and gates (validation)",
        "",
        _markdown_table(gate_rows),
        "",
        f"## Mitigation on `{family}` (slice column `{column}`)",
        "",
        _markdown_table(mitigation),
        "",
        f"Validation MAE per `{column}` slice and variant:",
        "",
        _markdown_table(by_column),
        "",
        "## Final decision",
        "",
        f"- Final model: **{final['name']}** ({decision['reason']}).",
        f"- Flagged for review: {'yes' if decision['flagged'] else 'no'}.",
        f"- Validation MAE {final['val_mae']:.4f}; test MAE {test['overall']['mae']:.4f}, "
        f"test R² {test['overall']['r2']:.3f}, test Spearman {test['overall']['spearman']:.3f}.",
        "",
        "## Final model: gated slices (validation)",
        "",
        _markdown_table(gated),
        "",
        "## Final model: small slices (validation, not gated)",
        "",
        _markdown_table(small) if not small.empty else "None.",
        "",
        "## Final model: all slices (test, reporting only)",
        "",
        _markdown_table(test_view),
        "",
    ]
    (out / "bias_report.md").write_text("\n".join(lines), encoding="utf-8")
    files.append(out / "bias_report.md")
    return files


# ---------------------------------------------------------------------------- select command


def stage_gate(cfg: ModelConfig, stage: str) -> tuple[dict[str, Any], list[GateResult]]:
    """Gates recorded for a stage (``part1``, ``part2`` or ``part3``) from its summary."""
    summary = read_summary(cfg, stage)
    if stage == "part3":
        final = summary["final"]
        return final, [_gate_from_dict(final["validation_gate"]), _gate_from_dict(final["bias_gate"])]
    return summary, [_gate_from_dict(summary["validation_gate"])]


__all__ = [
    "read_summary",
    "run_bias",
    "run_explain",
    "run_sensitivity",
    "run_tune",
    "select_best",
    "stage_gate",
]
