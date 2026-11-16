"""Common interface of all Movie DNA candidate models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

EpochCallback = Callable[[int, float], None]
PARAMS_FILE = "params.json"


class DnaModel(ABC):
    """Multi-output regressor: feature matrix -> 12 DNA scores in [0, 1]."""

    name: ClassVar[str] = "base"

    def __init__(self, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.seed = int(seed)
        self.n_outputs: int | None = None
        self.fit_info: dict[str, Any] = {}

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> DnaModel:
        """Train on (X, Y); X_val / Y_val are only used for early stopping."""

    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Unclipped predictions, shape (n, n_outputs)."""

    @abstractmethod
    def _save_weights(self, directory: Path) -> None:
        """Write the fitted weights into ``directory``."""

    @abstractmethod
    def _load_weights(self, directory: Path) -> None:
        """Read weights written by :meth:`_save_weights`."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions clipped to [0, 1]."""
        if self.n_outputs is None:
            raise RuntimeError(f"{self.name} model used before fit")
        return np.clip(self._predict_raw(np.asarray(X, dtype=np.float32)), 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        """Hyperparameters (and the seed) of this model."""
        return {**self.params, "seed": self.seed}

    def save(self, directory: Path) -> None:
        """Save parameters, fit info and weights into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": self.name,
            "params": self.params,
            "seed": self.seed,
            "n_outputs": self.n_outputs,
            "fit_info": self.fit_info,
        }
        (directory / PARAMS_FILE).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        self._save_weights(directory)

    @classmethod
    def load(cls, directory: Path) -> DnaModel:
        """Rebuild a saved model of this class."""
        meta = json.loads((directory / PARAMS_FILE).read_text(encoding="utf-8"))
        model = cls(meta["params"], meta["seed"])
        model.n_outputs = meta["n_outputs"]
        model.fit_info = meta.get("fit_info", {})
        model._load_weights(directory)
        return model

    @staticmethod
    def _check_xy(X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
            raise ValueError(f"expected X (n, f) and Y (n, k), got {X.shape} and {Y.shape}")
