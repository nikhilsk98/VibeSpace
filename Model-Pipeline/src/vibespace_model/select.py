"""Model selection, the validation gate and the final selection with the bias gate.

Selection only ever looks at validation metrics. The validation gate is reused by CI (Part 5): the
selected model must beat the mean baseline's validation MAE by ``min_mae_improvement`` and reach a mean
validation R² of ``min_r2``. Part 3 adds the bias gate through :func:`final_selection`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vibespace_model.evaluate import improvement_over_baseline, write_json
from vibespace_model.features import FeatureBuilder
from vibespace_model.models import DnaModel, load_model

MODEL_DIR, FEATURES_FILE, CARD_FILE = "model", "features.json", "model_card.json"


@dataclass
class GateResult:
    """Outcome of a gate with human-readable reasons."""

    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON / MLflow."""
        return asdict(self)


def validation_gate(
    val_overall: dict[str, float], baseline_overall: dict[str, float], gate: dict[str, float]
) -> GateResult:
    """Pass when MAE improvement >= ``min_mae_improvement`` and mean R² >= ``min_r2``."""
    improvement = improvement_over_baseline(val_overall["mae"], baseline_overall["mae"])
    checks = {
        "val_mae": val_overall["mae"],
        "baseline_val_mae": baseline_overall["mae"],
        "mae_improvement": improvement,
        "min_mae_improvement": gate["min_mae_improvement"],
        "val_r2": val_overall["r2"],
        "min_r2": gate["min_r2"],
    }
    reasons = []
    if improvement < gate["min_mae_improvement"]:
        reasons.append(
            f"MAE improvement over baseline {improvement:.1%} < required {gate['min_mae_improvement']:.1%}"
        )
    if val_overall["r2"] < gate["min_r2"]:
        reasons.append(f"mean validation R² {val_overall['r2']:.3f} < required {gate['min_r2']:.3f}")
    passed = not reasons
    if passed:
        reasons.append(
            f"MAE improvement {improvement:.1%} >= {gate['min_mae_improvement']:.1%} and "
            f"R² {val_overall['r2']:.3f} >= {gate['min_r2']:.3f}"
        )
    return GateResult("validation", passed, reasons, checks)


def select_best(val_overall: dict[str, dict[str, float]], tolerance: float = 1e-9) -> str:
    """Lowest mean validation MAE; ties (within ``tolerance``) go to the higher mean Spearman."""
    if not val_overall:
        raise ValueError("no candidates to select from")
    best_mae = min(m["mae"] for m in val_overall.values())
    tied = [name for name, m in val_overall.items() if m["mae"] - best_mae <= tolerance]
    return max(tied, key=lambda name: (val_overall[name]["spearman"], -list(val_overall).index(name)))


def final_selection(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the final model after both gates.

    Each entry: ``name``, ``val_mae``, ``val_spearman``, ``validation_gate`` and ``bias_gate``
    (:class:`GateResult`), ``bias_gap`` (worst gated slice MAE / overall MAE).
    Rule: best validation MAE among entries passing both gates. If none passes the bias gate, the entry
    with the smallest bias gap among those passing the validation gate is chosen and flagged.
    """
    both = [e for e in entries if e["validation_gate"].passed and e["bias_gate"].passed]
    if both:
        chosen = min(both, key=lambda e: (e["val_mae"], -e["val_spearman"]))
        return {
            "name": chosen["name"],
            "flagged": False,
            "reason": "best validation MAE among models passing the validation and bias gates",
        }
    valid = [e for e in entries if e["validation_gate"].passed]
    if valid:
        chosen = min(valid, key=lambda e: (e["bias_gap"], e["val_mae"]))
        return {
            "name": chosen["name"],
            "flagged": True,
            "reason": "no model passes the bias gate; chose the smallest slice gap among models "
            "passing the validation gate (flagged for review)",
        }
    chosen = min(entries, key=lambda e: (e["val_mae"], -e["val_spearman"]))
    return {
        "name": chosen["name"],
        "flagged": True,
        "reason": "no model passes the validation gate; best validation MAE chosen (flagged, not deployable)",
    }


# ---------------------------------------------------------------------------- bundles + model card


def build_model_card(
    *,
    model: DnaModel,
    stage: str,
    data_version: dict[str, Any],
    split_hash: str,
    metrics: dict[str, Any],
    gate_cfg: dict[str, float],
    validation_gate_result: GateResult,
    bias_gate_result: GateResult | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Model card stored next to the saved model (and logged to MLflow)."""
    return {
        "model_name": "vibespace-movie-dna",
        "stage": stage,
        "model_type": model.name,
        "params": model.get_params(),
        "fit_info": model.fit_info,
        "data_version": data_version,
        "split_hash": split_hash,
        "metrics": metrics,
        "thresholds": {"validation_gate": gate_cfg},
        "validation_gate": validation_gate_result.to_dict(),
        "bias_gate": bias_gate_result.to_dict() if bias_gate_result else {"status": "pending (Part 3)"},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def save_bundle(directory: Path, model: DnaModel, builder: FeatureBuilder, card: dict[str, Any]) -> Path:
    """Model + preprocessing + model card in one folder (replaces an older bundle)."""
    if directory.exists():
        shutil.rmtree(directory)
    model.save(directory / MODEL_DIR)
    builder.save(directory / FEATURES_FILE)
    write_json(directory / CARD_FILE, card)
    return directory


def load_bundle(directory: Path) -> tuple[DnaModel, FeatureBuilder, dict[str, Any]]:
    """Reverse of :func:`save_bundle`."""
    model = load_model(directory / MODEL_DIR)
    builder = FeatureBuilder.load(directory / FEATURES_FILE)
    card = json.loads((directory / CARD_FILE).read_text(encoding="utf-8"))
    return model, builder, card
