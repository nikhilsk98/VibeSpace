"""Multi-output Ridge regression (one shared regularisation strength)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from vibespace_model.models.base import DnaModel, EpochCallback


class RidgeModel(DnaModel):
    """Linear model on the standardised metadata and the raw plot embedding."""

    name = "ridge"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        super().__init__(params, seed)
        self.model: Ridge | None = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> RidgeModel:
        """Closed-form fit; ``alpha`` comes from the params (default 1.0)."""
        self._check_xy(X, Y)
        self.model = Ridge(alpha=float(self.params.get("alpha", 1.0)), random_state=self.seed)
        self.model.fit(X.astype(np.float64), Y, sample_weight=sample_weight)
        self.n_outputs = Y.shape[1]
        return self

    @property
    def coef_(self) -> np.ndarray:
        """Coefficients, shape (n_outputs, n_features)."""
        return self.model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Intercepts, shape (n_outputs,)."""
        return self.model.intercept_

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X.astype(np.float64))

    def _save_weights(self, directory: Path) -> None:
        joblib.dump(self.model, directory / "ridge.joblib")

    def _load_weights(self, directory: Path) -> None:
        self.model = joblib.load(directory / "ridge.joblib")
