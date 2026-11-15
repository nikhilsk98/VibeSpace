"""Selection, tie-break, validation gate, final selection with the bias gate, bundles and model card."""

from __future__ import annotations

import numpy as np
import pytest

from vibespace_model.features import FeatureBuilder
from vibespace_model.models import create_model
from vibespace_model.select import (
    GateResult,
    build_model_card,
    final_selection,
    load_bundle,
    save_bundle,
    select_best,
    validation_gate,
)

GATE = {"min_mae_improvement": 0.15, "min_r2": 0.30}
BASE = {"mae": 0.25, "r2": 0.0, "spearman": 0.0}


def test_select_best_by_mae() -> None:
    results = {
        "baseline": BASE,
        "ridge": {"mae": 0.17, "spearman": 0.6},
        "lightgbm": {"mae": 0.16, "spearman": 0.5},
    }
    assert select_best(results) == "lightgbm"


def test_tie_break_on_spearman() -> None:
    results = {"a": {"mae": 0.16, "spearman": 0.60}, "b": {"mae": 0.16, "spearman": 0.70}}
    assert select_best(results) == "b"
    same = {"a": {"mae": 0.16, "spearman": 0.7}, "b": {"mae": 0.16, "spearman": 0.7}}
    assert select_best(same) == "a"  # full tie: first in candidate order
    with pytest.raises(ValueError):
        select_best({})


def test_validation_gate_pass() -> None:
    gate = validation_gate({"mae": 0.20, "r2": 0.45}, BASE, GATE)
    assert gate.passed
    assert gate.checks["mae_improvement"] == pytest.approx(0.2)
    assert "20.0%" in gate.reasons[0]


def test_validation_gate_fail_reasons() -> None:
    gate = validation_gate({"mae": 0.23, "r2": 0.10}, BASE, GATE)
    assert not gate.passed
    assert len(gate.reasons) == 2
    assert any("MAE improvement" in r for r in gate.reasons)
    assert any("R²" in r for r in gate.reasons)
    only_r2 = validation_gate({"mae": 0.15, "r2": 0.1}, BASE, GATE)
    assert not only_r2.passed and len(only_r2.reasons) == 1
    assert only_r2.to_dict()["name"] == "validation"


def _entry(name, mae, val_ok, bias_ok, gap, spearman=0.5):
    return {
        "name": name,
        "val_mae": mae,
        "val_spearman": spearman,
        "bias_gap": gap,
        "validation_gate": GateResult("validation", val_ok),
        "bias_gate": GateResult("bias", bias_ok),
    }


def test_final_selection_prefers_models_passing_both_gates() -> None:
    entries = [
        _entry("best_but_biased", 0.15, True, False, 1.4),
        _entry("fair", 0.16, True, True, 1.1),
        _entry("fairer", 0.17, True, True, 1.0),
    ]
    decision = final_selection(entries)
    assert decision == {
        "name": "fair",
        "flagged": False,
        "reason": "best validation MAE among models passing the validation and bias gates",
    }


def test_final_selection_none_pass_bias_gate() -> None:
    entries = [
        _entry("a", 0.15, True, False, 1.40),
        _entry("b", 0.16, True, False, 1.30),
        _entry("c", 0.14, False, False, 1.10),
    ]
    decision = final_selection(entries)
    assert decision["name"] == "b"  # smallest gap among models passing the validation gate
    assert decision["flagged"]


def test_final_selection_none_pass_validation_gate() -> None:
    decision = final_selection([_entry("a", 0.2, False, True, 1.0), _entry("b", 0.19, False, True, 1.0)])
    assert decision["name"] == "b" and decision["flagged"]


def test_bundle_and_model_card(tmp_path, rng) -> None:
    from tests.fixtures.build_fixtures import make_tables, model_config

    tables = make_tables(60, seed=1)
    features_cfg = model_config(tmp_path)["features"]
    catalog = tables["catalog"].iloc[:50].reset_index(drop=True)
    emb = tables["embeddings"][:50]
    builder = FeatureBuilder(features_cfg, 16).fit(catalog, emb)
    X = builder.transform(catalog, emb)
    Y = rng.random((50, 12))
    model = create_model("ridge", {"alpha": 2.0}).fit(X, Y)
    gate = validation_gate({"mae": 0.2, "r2": 0.4}, BASE, GATE)
    card = build_model_card(
        model=model,
        stage="test",
        data_version={"git_commit": "x"},
        split_hash="h",
        metrics={"val": {"overall": {"mae": 0.2}}},
        gate_cfg=GATE,
        validation_gate_result=gate,
        extra={"targets": ["t"] * 12},
    )
    for key in (
        "model_type",
        "params",
        "data_version",
        "split_hash",
        "metrics",
        "thresholds",
        "validation_gate",
        "bias_gate",
        "created_at",
    ):
        assert key in card
    assert card["bias_gate"] == {"status": "pending (Part 3)"}
    save_bundle(tmp_path / "bundle", model, builder, card)
    save_bundle(tmp_path / "bundle", model, builder, card)  # overwriting an old bundle works
    loaded_model, loaded_builder, loaded_card = load_bundle(tmp_path / "bundle")
    np.testing.assert_array_equal(
        loaded_model.predict(loaded_builder.transform(catalog, emb)), model.predict(X)
    )
    assert loaded_card["split_hash"] == "h"
    with_bias = build_model_card(
        model=model,
        stage="final",
        data_version={},
        split_hash="h",
        metrics={},
        gate_cfg=GATE,
        validation_gate_result=gate,
        bias_gate_result=GateResult("bias", True, ["ok"]),
    )
    assert with_bias["bias_gate"]["passed"] is True
