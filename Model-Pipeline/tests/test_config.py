"""Configuration loading, path resolution and environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from vibespace_model.config import DNA_DIMENSIONS, load_config
from vibespace_model.errors import ConfigError

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"


def _edit(path: Path, **changes: object) -> None:
    raw = yaml.safe_load(path.read_text())
    for dotted, value in changes.items():
        node = raw
        *parents, leaf = dotted.split("__")
        for key in parents:
            node = node[key]
        if value is None:
            node.pop(leaf)
        else:
            node[leaf] = value
    path.write_text(yaml.safe_dump(raw))


def test_repository_config_loads_with_defaults() -> None:
    cfg = load_config(REPO_CONFIG)
    assert cfg.random_seed == 42
    assert tuple(cfg.targets) == DNA_DIMENSIONS
    assert cfg.paths.dvc_lock.name == "dvc.lock"
    assert cfg.paths.processed_dir.parts[-3:] == ("Data-Pipeline", "data", "processed")
    assert cfg.gate == {"min_mae_improvement": 0.15, "min_r2": 0.30}
    assert cfg.tuning["models"]["lightgbm"]["n_trials"] == 50
    assert cfg.tuning["models"]["mlp"]["n_trials"] == 40
    assert cfg.bias["min_gated_slice_size"] == 30


def test_environment_overrides(tree, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("VIBESPACE_MODEL_REPORTS_DIR", str(tmp_path / "elsewhere"))
    cfg = load_config(tree["config"])
    assert cfg.random_seed == 7
    assert cfg.paths.reports_dir == (tmp_path / "elsewhere").resolve()


def test_relative_paths_resolve_against_model_pipeline_folder(tree) -> None:
    _edit(tree["config"], paths__reports_dir="my_reports")
    cfg = load_config(tree["config"])
    assert cfg.paths.reports_dir == (tree["config"].parent.parent / "my_reports").resolve()


def test_sample_runs_use_separate_folders(tree) -> None:
    cfg = load_config(tree["config"], sample_size=50)
    assert cfg.reports_subdir("part1").parts[-2:] == ("sample", "part1")
    assert cfg.artifacts_subdir("part1").parts[-2:] == ("sample", "part1")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"split__train": 0.9}, "sum to 1"),
        ({"targets": ["energy"]}, "12 DNA dimensions"),
        ({"gate__min_r2": None}, "gate.min_r2"),
        ({"models__xgboost": {}}, "unknown candidate"),
        ({"data": None}, "missing key 'data'"),
    ],
)
def test_invalid_configs_are_rejected(tree, changes, message) -> None:
    _edit(tree["config"], **changes)
    with pytest.raises(ConfigError, match=message):
        load_config(tree["config"])


def test_missing_file_and_non_mapping(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "configs" / "model_config.yaml"
    bad.parent.mkdir()
    bad.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_model_params_are_copies(cfg) -> None:
    params = cfg.model_params("mlp")
    params["hidden_sizes"].append(999)
    assert 999 not in cfg.model_params("mlp")["hidden_sizes"]
    with pytest.raises(ConfigError):
        cfg.model_params("xgboost")
