"""PyTorch MLP on CPU: input -> hidden layers (ReLU + dropout) -> 12 sigmoid outputs.

Huber loss, Adam, early stopping on the validation loss (best weights restored), fixed seeds and a fixed
number of CPU threads so runs are reproducible.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from vibespace_model.models.base import DnaModel, EpochCallback


def parse_hidden_sizes(value: Any) -> list[int]:
    """Accept ``[256, 128]``, ``"256-128"`` or ``256``."""
    if isinstance(value, str):
        return [int(v) for v in value.split("-") if v]
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


class _Network(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], n_out: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_in
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers += [nn.Linear(width, n_out), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _seed_everything(seed: int, threads: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (legacy global seed for any library that uses it)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


class MLPModel(DnaModel):
    """Feed-forward network; ``epoch_callback(epoch, val_mae)`` lets Optuna prune trials."""

    name = "mlp"

    def __init__(self, params: dict | None = None, seed: int = 42) -> None:
        super().__init__(params, seed)
        self.network: _Network | None = None
        self.n_inputs: int | None = None

    def _p(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    def _build(self, n_in: int, n_out: int) -> _Network:
        hidden = parse_hidden_sizes(self._p("hidden_sizes", [256, 128]))
        return _Network(n_in, hidden, n_out, float(self._p("dropout", 0.2)))

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        epoch_callback: EpochCallback | None = None,
    ) -> MLPModel:
        """Mini-batch training with early stopping on validation loss (fixed epochs without val data)."""
        self._check_xy(X, Y)
        _seed_everything(self.seed, int(self._p("num_threads", 4)))
        generator = torch.Generator().manual_seed(self.seed)
        self.n_inputs, self.n_outputs = X.shape[1], Y.shape[1]
        self.network = self._build(self.n_inputs, self.n_outputs)

        x = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(Y, dtype=torch.float32)
        w = torch.as_tensor(
            sample_weight if sample_weight is not None else np.ones(len(X)), dtype=torch.float32
        )
        has_val = X_val is not None and Y_val is not None
        xv = torch.as_tensor(X_val, dtype=torch.float32) if has_val else None
        yv = torch.as_tensor(Y_val, dtype=torch.float32) if has_val else None

        loss_fn = nn.HuberLoss(reduction="none", delta=float(self._p("huber_delta", 0.1)))
        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=float(self._p("learning_rate", 1e-3)),
            weight_decay=float(self._p("weight_decay", 1e-4)),
        )
        batch_size = int(self._p("batch_size", 128))
        max_epochs, patience = int(self._p("max_epochs", 200)), int(self._p("patience", 15))

        best_loss, best_state, best_epoch, stale, history = np.inf, None, 0, 0, []
        for epoch in range(1, max_epochs + 1):
            self.network.train()
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                per_row = loss_fn(self.network(x[idx]), y[idx]).mean(dim=1)
                loss = (per_row * w[idx]).sum() / w[idx].sum()
                loss.backward()
                optimizer.step()
            if not has_val:
                continue
            self.network.eval()
            with torch.no_grad():
                pred = self.network(xv)
                val_loss = float(loss_fn(pred, yv).mean())
                val_mae = float((pred - yv).abs().mean())
            history.append({"epoch": epoch, "val_loss": val_loss, "val_mae": val_mae})
            if val_loss < best_loss - 1e-7:
                best_loss, best_state, best_epoch, stale = (
                    val_loss,
                    copy.deepcopy(self.network.state_dict()),
                    epoch,
                    0,
                )
            else:
                stale += 1
            if epoch_callback is not None:
                epoch_callback(epoch, val_mae)
            if stale >= patience:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        self.fit_info = {
            "best_epoch": best_epoch if has_val else max_epochs,
            "epochs_run": history[-1]["epoch"] if history else max_epochs,
            "best_val_loss": None if best_state is None else best_loss,
        }
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            return self.network(torch.as_tensor(X, dtype=torch.float32)).numpy().astype(np.float64)

    def torch_module(self) -> nn.Module:
        """The trained network (used by SHAP's GradientExplainer)."""
        return self.network

    def _save_weights(self, directory: Path) -> None:
        torch.save(self.network.state_dict(), directory / "mlp_state.pt")
        (directory / "mlp_shape.json").write_text(json.dumps({"n_inputs": self.n_inputs}), encoding="utf-8")

    def _load_weights(self, directory: Path) -> None:
        self.n_inputs = json.loads((directory / "mlp_shape.json").read_text(encoding="utf-8"))["n_inputs"]
        self.network = self._build(self.n_inputs, self.n_outputs)
        self.network.load_state_dict(torch.load(directory / "mlp_state.pt", weights_only=True))
        self.network.eval()
