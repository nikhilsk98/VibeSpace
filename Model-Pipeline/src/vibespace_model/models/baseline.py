"""Mean baseline: predicts the training mean of every DNA dimension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vibespace_model.models.base import DnaModel, EpochCallback


class MeanBaseline(DnaModel):
    """Reference point for the validation gate (MAE improvement is measured against it)."""

    name = "baseline"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        super().__init__(params, seed)
        self.means: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> MeanBaseline:
        """Store the (weighted) mean of each output."""
        self._check_xy(X, Y)
        self.means = np.average(Y, axis=0, weights=sample_weight)
        self.n_outputs = Y.shape[1]
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.means, (len(X), 1))

    def _save_weights(self, directory: Path) -> None:
        (directory / "means.json").write_text(json.dumps(self.means.tolist()), encoding="utf-8")

    def _load_weights(self, directory: Path) -> None:
        self.means = np.asarray(json.loads((directory / "means.json").read_text(encoding="utf-8")))
