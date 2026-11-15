"""MLflow logging to a temporary store and registration in the local Model Registry."""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from vibespace_model import tracking
from vibespace_model.data_loader import load_dataset
from vibespace_model.features import FeatureBuilder
from vibespace_model.models import create_model
from vibespace_model.select import GateResult, build_model_card, save_bundle


def test_flatten_metrics() -> None:
    metrics = {"overall": {"mae": 0.1, "n": 5}, "per_dimension": {"energy": {"mae": 0.2}}}
    assert tracking.flatten_metrics(metrics, "val") == {"val_mae": 0.1, "val_mae_energy": 0.2}


def test_logging_and_registration(cfg, tmp_path) -> None:
    experiment_id = tracking.setup_tracking(cfg)
    assert (cfg.paths.mlflow_dir / "mlflow.db").is_file()
    assert tracking.setup_tracking(cfg) == experiment_id  # idempotent
    assert tracking.registry_summary(cfg) == []

    data = load_dataset(cfg)
    builder = FeatureBuilder(cfg.features, 16).fit(data.catalog, data.embeddings)
    X = builder.transform(data.catalog, data.embeddings)
    model = create_model("ridge", {"alpha": 1.0}).fit(X, data.labels)
    card = build_model_card(
        model=model,
        stage="test",
        data_version=data.data_version,
        split_hash="abc",
        metrics={},
        gate_cfg=cfg.gate,
        validation_gate_result=GateResult("validation", True),
        extra={"targets": cfg.targets},
    )
    bundle = save_bundle(tmp_path / "bundle", model, builder, card)

    with tracking.run("parent", tags={"part": "test"}) as parent:
        with tracking.run("child", nested=True, tags={"model_type": "ridge"}):
            tracking.log_params({"alpha": 1.0, "long": "x" * 900})
            tracking.log_metrics({"val_mae": 0.12, "bad": float("nan"), "none": None})
        tracking.log_artifacts([tmp_path / "bundle" / "model_card.json", tmp_path / "missing.txt"], "cards")
        first = tracking.register_bundle(cfg, bundle, alias="candidate", description="first")
        second = tracking.register_bundle(cfg, bundle, alias="champion-candidate", description="second")

    client = MlflowClient()
    runs = client.search_runs([experiment_id])
    child = next(r for r in runs if r.info.run_name == "child")
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.data.metrics == {"val_mae": 0.12}
    assert len(child.data.params["long"]) == 500
    assert first["version"] == "1" and second["version"] == "2"
    assert client.get_model_version_by_alias(cfg.registered_model_name, "candidate").version == 1
    summary = tracking.registry_summary(cfg)
    assert [s["aliases"] for s in summary] == [["candidate"], ["champion-candidate"]]

    loaded = mlflow.pyfunc.load_model(f"models:/{cfg.registered_model_name}@champion-candidate")
    frame = data.catalog.iloc[:4].copy()
    emb = pd.DataFrame(data.embeddings[:4], columns=[f"emb_{i:03d}" for i in range(16)])
    scores = loaded.predict(pd.concat([frame.reset_index(drop=True), emb], axis=1))
    assert list(scores.columns) == cfg.targets
    np.testing.assert_allclose(scores.to_numpy(), model.predict(X[:4]), rtol=1e-6)


def test_session_tags(cfg) -> None:
    tags = tracking.session_tags(cfg, {"data_version_id": "d", "git_commit": "g"}, "s", "part1")
    assert tags["split_hash"] == "s" and tags["git_commit"] == "g" and tags["sample_size"] == "full"
