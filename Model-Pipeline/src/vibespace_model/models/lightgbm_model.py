"""LightGBM: one gradient-boosted regressor per DNA dimension, predictions clipped to [0, 1]."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from vibespace_model.models.base import DnaModel, EpochCallback

_BOOSTER_KEYS = (
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "n_jobs",
)


class LightGBMModel(DnaModel):
    """12 independent boosters; the number of trees is chosen by early stopping on validation."""

    name = "lightgbm"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        super().__init__(params, seed)
        self.boosters: list[lgb.Booster] = []

    def _regressor_params(self) -> dict[str, Any]:
        params = {k: self.params[k] for k in _BOOSTER_KEYS if k in self.params}
        params.setdefault("n_estimators", 500)
        return {
            **params,
            "objective": self.params.get("objective", "regression"),
            "random_state": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "metric": "l1",  # early stopping watches validation MAE only
            "verbosity": -1,
        }

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> LightGBMModel:
        """Fit one regressor per output; early stopping when validation data is given."""
        self._check_xy(X, Y)
        stop_rounds = int(self.params.get("early_stopping_rounds", 0) or 0)
        self.boosters, best_iterations = [], []
        for k in range(Y.shape[1]):
            regressor = lgb.LGBMRegressor(**self._regressor_params())
            fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
            if X_val is not None and Y_val is not None and stop_rounds > 0:
                fit_kwargs["eval_X"] = (X_val,)
                fit_kwargs["eval_y"] = (Y_val[:, k],)
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stop_rounds, first_metric_only=False, verbose=False)
                ]
            regressor.fit(X, Y[:, k], **fit_kwargs)
            best = regressor.best_iteration_ or regressor.n_estimators
            best_iterations.append(int(best))
            booster = regressor.booster_
            # keep only the trees up to the best iteration, so save/load/predict all agree
            self.boosters.append(lgb.Booster(model_str=booster.model_to_string(num_iteration=best)))
        self.n_outputs = Y.shape[1]
        self.fit_info = {"best_iterations": best_iterations}
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([booster.predict(X) for booster in self.boosters])

    def _save_weights(self, directory: Path) -> None:
        for k, booster in enumerate(self.boosters):
            booster.save_model(str(directory / f"booster_{k:02d}.txt"))
        (directory / "boosters.json").write_text(json.dumps({"n": len(self.boosters)}), encoding="utf-8")

    def _load_weights(self, directory: Path) -> None:
        n = json.loads((directory / "boosters.json").read_text(encoding="utf-8"))["n"]
        self.boosters = [lgb.Booster(model_file=str(directory / f"booster_{k:02d}.txt")) for k in range(n)]
