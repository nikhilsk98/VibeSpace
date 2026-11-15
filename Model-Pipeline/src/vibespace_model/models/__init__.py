"""Candidate models with one interface: ``fit``, ``predict``, ``save``, ``load``, ``get_params``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vibespace_model.models.base import PARAMS_FILE, DnaModel
from vibespace_model.models.baseline import MeanBaseline
from vibespace_model.models.lightgbm_model import LightGBMModel
from vibespace_model.models.mlp import MLPModel
from vibespace_model.models.ridge import RidgeModel

MODEL_CLASSES: dict[str, type[DnaModel]] = {
    "baseline": MeanBaseline,
    "ridge": RidgeModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}
CANDIDATES: tuple[str, ...] = tuple(MODEL_CLASSES)


def create_model(name: str, params: dict[str, Any] | None = None, seed: int = 42) -> DnaModel:
    """New, unfitted model of type ``name``."""
    if name not in MODEL_CLASSES:
        raise ValueError(f"unknown model type '{name}'; choose from {sorted(MODEL_CLASSES)}")
    return MODEL_CLASSES[name](params, seed)


def load_model(directory: Path) -> DnaModel:
    """Load any saved model; the type is read from its ``params.json``."""
    meta = json.loads((directory / PARAMS_FILE).read_text(encoding="utf-8"))
    return MODEL_CLASSES[meta["model_type"]].load(directory)


__all__ = ["CANDIDATES", "MODEL_CLASSES", "DnaModel", "create_model", "load_model"]
