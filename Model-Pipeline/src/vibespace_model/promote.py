"""Part 4: champion / challenger promotion and rollback.

Promotion rule (``registry.promotion_tolerance`` in the config, default 0.002): a candidate becomes
champion only if it passes the validation gate **and** the bias gate **and** its mean validation MAE is no
worse than the current champion's by more than the tolerance, both evaluated on the same split. Otherwise
the champion is unchanged ("not promoted (rollback)"). With no champion yet, the first candidate passing
both gates becomes champion. The champion is mirrored as alias ``champion`` in the local MLflow registry.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vibespace_model import bias as bias_mod
from vibespace_model.config import ModelConfig
from vibespace_model.evaluate import compute_metrics
from vibespace_model.features import FeatureBuilder
from vibespace_model.models import DnaModel, create_model
from vibespace_model.registry import (
    DEFAULT_CHAMPION_PACKAGE,
    GcpBackend,
    LocalBackend,
    ModelRegistry,
    RegistryError,
    create_bundle,
    make_version,
)
from vibespace_model.select import GateResult, load_bundle, validation_gate
from vibespace_model.train import Prepared

LOGGER = logging.getLogger(__name__)
CHAMPION_ALIAS = "champion"


@dataclass
class Evaluation:
    """Metrics and gate results of one model on the shared split."""

    name: str
    val: dict[str, Any]
    test: dict[str, Any]
    validation_gate: GateResult
    bias_gate: GateResult
    bias_gap: float
    slice_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def val_mae(self) -> float:
        """Mean validation MAE."""
        return float(self.val["overall"]["mae"])

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (overall metrics and gates)."""
        return {
            "name": self.name,
            "val": self.val["overall"],
            "test": self.test["overall"],
            "validation_gate": self.validation_gate.to_dict(),
            "bias_gate": self.bias_gate.to_dict(),
            "bias_gap": self.bias_gap,
        }


@dataclass
class Decision:
    """Outcome of the promotion rule."""

    promote: bool
    reasons: list[str]
    candidate_val_mae: float
    champion_val_mae: float | None
    champion_version: str | None
    tolerance: float

    @property
    def label(self) -> str:
        """``promoted`` or ``not promoted (rollback)``."""
        return "promoted" if self.promote else "not promoted (rollback)"

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {**asdict(self), "label": self.label}


# ---------------------------------------------------------------------------- registry construction


def build_registry(cfg: ModelConfig, dry_run: bool = False) -> ModelRegistry:
    """GCP registry from ``cfg.registry``, or a local folder when ``dry_run`` is set."""
    reg = cfg.registry
    champions = reg.get("champion_package", DEFAULT_CHAMPION_PACKAGE)
    if dry_run:
        backend = LocalBackend(
            cfg.root / reg.get("dry_run_dir", "artifacts/registry-dry-run"),
            package=reg.get("package", "movie-dna"),
        )
        return ModelRegistry(backend, champion_package=champions)
    return ModelRegistry(
        GcpBackend(reg["project"], reg["location"], reg["repository"], reg["package"], reg["bucket"]),
        champion_package=champions,
    )


# ---------------------------------------------------------------------------- evaluation


def split_inputs(prep: Prepared) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Raw catalog rows and embeddings per split (so each bundle applies its own preprocessing)."""
    masks = prep.split.masks(prep.dataset.movie_ids)
    return {
        name: (prep.dataset.subset(mask).catalog, prep.dataset.subset(mask).embeddings)
        for name, mask in masks.items()
    }


def baseline_val_overall(cfg: ModelConfig, prep: Prepared) -> dict[str, float]:
    """Mean-baseline validation metrics on the shared split (reference of the validation gate)."""
    baseline = create_model("baseline", seed=cfg.random_seed).fit(prep.X["train"], prep.Y["train"])
    return compute_metrics(prep.Y["val"], baseline.predict(prep.X["val"]), cfg.targets)["overall"]


def evaluate_model(
    cfg: ModelConfig,
    prep: Prepared,
    model: DnaModel,
    builder: FeatureBuilder,
    name: str,
    baseline: dict[str, float] | None = None,
) -> Evaluation:
    """Validation + test metrics, validation gate and bias gate (validation slices) for one model."""
    inputs = split_inputs(prep)
    preds = {split: model.predict(builder.transform(*inputs[split])) for split in ("val", "test")}
    val = compute_metrics(prep.Y["val"], preds["val"], cfg.targets)
    test = compute_metrics(prep.Y["test"], preds["test"], cfg.targets)
    table, overall = bias_mod.evaluate_slices(
        prep.Y["val"],
        preds["val"],
        prep.catalog["val"],
        list(cfg.bias.get("slice_columns", cfg.slices)),
        cfg.bias,
        cfg.random_seed,
    )
    return Evaluation(
        name=name,
        val=val,
        test=test,
        validation_gate=validation_gate(
            val["overall"], baseline or baseline_val_overall(cfg, prep), cfg.gate
        ),
        bias_gate=bias_mod.bias_gate(table, overall, cfg.bias),
        bias_gap=bias_mod.worst_ratio(table),
        slice_table=table,
    )


def evaluate_bundle_dir(
    cfg: ModelConfig, prep: Prepared, bundle_dir: Path, name: str, baseline: dict[str, float] | None = None
) -> Evaluation:
    """Load a saved or unpacked bundle and evaluate it."""
    model, builder, _ = load_bundle(bundle_dir)
    return evaluate_model(cfg, prep, model, builder, name, baseline)


def decide(
    candidate: Evaluation, champion: Evaluation | None, champion_version: str | None, tolerance: float
) -> Decision:
    """Apply the promotion rule."""
    champ_mae = champion.val_mae if champion else None
    reasons: list[str] = []
    if not candidate.validation_gate.passed:
        reasons.append("validation gate failed: " + "; ".join(candidate.validation_gate.reasons))
    if not candidate.bias_gate.passed:
        reasons.append("bias gate failed: " + "; ".join(candidate.bias_gate.reasons))
    if reasons:
        reasons.append(
            "champion unchanged" + (f" ({champion_version})" if champion_version else " (none yet)")
        )
        return Decision(False, reasons, candidate.val_mae, champ_mae, champion_version, tolerance)
    if champion is None:
        return Decision(
            True,
            ["no champion yet: first model passing the validation and bias gates"],
            candidate.val_mae,
            None,
            None,
            tolerance,
        )
    delta = candidate.val_mae - champion.val_mae
    if delta <= tolerance:
        how = (
            f"better than champion {champion_version} by {-delta:.5f}"
            if delta < 0
            else f"within tolerance: {delta:.5f} worse than champion {champion_version} (<= {tolerance})"
        )
        return Decision(
            True,
            [f"validation MAE {candidate.val_mae:.5f} vs {champion.val_mae:.5f}: {how}"],
            candidate.val_mae,
            champ_mae,
            champion_version,
            tolerance,
        )
    return Decision(
        False,
        [
            f"validation MAE {candidate.val_mae:.5f} vs champion {champion_version} "
            f"({champion.val_mae:.5f}): difference {delta:+.5f}, more than the tolerance {tolerance:+.4f}",
            f"champion unchanged ({champion_version})",
        ],
        candidate.val_mae,
        champ_mae,
        champion_version,
        tolerance,
    )


def evaluate_against_champion(
    cfg: ModelConfig, registry: ModelRegistry, candidate_dir: Path, prep: Prepared, workdir: Path
) -> tuple[Decision, Evaluation, Evaluation | None, dict | None]:
    """Pull the champion, evaluate candidate and champion on the same split, apply the rule."""
    baseline = baseline_val_overall(cfg, prep)
    candidate = evaluate_bundle_dir(cfg, prep, candidate_dir, "candidate", baseline)
    champion_meta = registry.get_champion()
    champion = None
    if champion_meta:
        root = registry.pull(champion_meta["version"], workdir / "champion")
        champion = evaluate_bundle_dir(cfg, prep, root, f"champion {champion_meta['version']}", baseline)
    tolerance = float(cfg.registry.get("promotion_tolerance", 0.002))
    decision = decide(candidate, champion, champion_meta["version"] if champion_meta else None, tolerance)
    return decision, candidate, champion, champion_meta


# ---------------------------------------------------------------------------- promote / rollback


def promote(
    cfg: ModelConfig,
    registry: ModelRegistry,
    candidate_dir: Path,
    prep: Prepared,
    extra: dict[str, Any] | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Push the candidate bundle, then set it champion if the rule allows (otherwise keep the champion)."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        decision, cand_eval, champ_eval, champion_meta = evaluate_against_champion(
            cfg, registry, candidate_dir, prep, work
        )
        _, _, card = load_bundle(candidate_dir)
        version = version or make_version((card.get("data_version") or {}).get("git_commit"))
        bundle = create_bundle(candidate_dir, work / "packed", version, extra)
        meta = registry.push(
            bundle,
            {
                "candidate_evaluation": cand_eval.to_dict(),
                "champion_evaluation": champ_eval.to_dict() if champ_eval else None,
                "decision": decision.to_dict(),
                "status": decision.label,
                **(extra or {}),
            },
        )
    mirrored = None
    if decision.promote:
        registry.set_champion(
            version,
            "; ".join(decision.reasons),
            event="promote",
            details={
                "val_mae": cand_eval.val_mae,
                "test_mae": cand_eval.test["overall"]["mae"],
                "local_registry_version": meta.get("local_registry_version"),
            },
        )
        mirrored = mirror_champion_alias(cfg, meta)
    else:
        LOGGER.info("Not promoted (rollback): %s", "; ".join(decision.reasons))
    return {
        "version": version,
        "decision": decision.to_dict(),
        "candidate": cand_eval.to_dict(),
        "champion_before": champion_meta,
        "champion_evaluation": champ_eval.to_dict() if champ_eval else None,
        "champion_after": registry.get_champion(),
        "local_mlflow_alias_version": mirrored,
        "backend": registry.backend.description,
    }


def rollback(cfg: ModelConfig, registry: ModelRegistry, to_version: str | None = None) -> dict[str, Any]:
    """Reset the champion to the previous champion (or ``to_version``) after verifying its bundle."""
    current = registry.get_champion()
    target = to_version or (current or {}).get("previous_version")
    if not target:
        raise RegistryError("nothing to roll back to: no previous champion recorded (pass --to-version)")
    if current and target == current["version"]:
        raise RegistryError(f"{target} is already the champion")
    registry.verify_remote(target)  # raises if the bundle is missing or its checksum does not match
    champion = registry.set_champion(
        target, f"manual rollback from {current['version'] if current else 'none'}", event="rollback"
    )
    meta = registry.version_meta(target) or {}
    return {"champion": champion, "local_mlflow_alias_version": mirror_champion_alias(cfg, meta)}


def mirror_champion_alias(cfg: ModelConfig, meta: dict[str, Any]) -> str | None:
    """Set alias ``champion`` on the matching local MLflow version (None if it is not in this store)."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from vibespace_model import tracking

    local_version = meta.get("local_registry_version")
    if not local_version:
        LOGGER.warning(
            "Champion %s has no local MLflow version recorded; alias not mirrored", meta.get("version")
        )
        return None
    tracking.setup_tracking(cfg)
    client = MlflowClient()
    try:
        found = client.get_model_version(cfg.registered_model_name, str(local_version))
    except MlflowException:
        LOGGER.warning(
            "Local MLflow has no version %s of %s; alias not mirrored",
            local_version,
            cfg.registered_model_name,
        )
        return None
    if meta.get("mlflow_run_id") and found.run_id != meta["mlflow_run_id"]:
        LOGGER.warning("Local MLflow version %s belongs to another run; alias not mirrored", local_version)
        return None
    client.set_registered_model_alias(cfg.registered_model_name, CHAMPION_ALIAS, str(local_version))
    client.set_model_version_tag(
        cfg.registered_model_name, str(local_version), "gcp_registry_version", meta.get("version", "")
    )
    return str(local_version)


def local_registration(cfg: ModelConfig, alias: str) -> dict[str, Any]:
    """MLflow run id and version behind a local alias (empty dict if the alias does not exist)."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from vibespace_model import tracking

    tracking.setup_tracking(cfg)
    try:
        version = MlflowClient().get_model_version_by_alias(cfg.registered_model_name, alias)
    except MlflowException:
        return {}
    return {
        "local_registry_version": str(version.version),
        "mlflow_run_id": version.run_id,
        "local_registry_alias": alias,
    }
