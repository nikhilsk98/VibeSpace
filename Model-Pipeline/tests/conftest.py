"""Shared fixtures: synthetic pipeline outputs in a temporary folder, configs pointing at it."""

# macOS: pytest.ini loads vibespace_model.pytest_openmp first, so LightGBM and PyTorch share one OpenMP.
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.build_fixtures import build_fixture_tree
from vibespace_model.config import ModelConfig, load_config

for _name in list(os.environ):
    if _name.startswith("VIBESPACE_MODEL_") or _name in ("RANDOM_SEED", "GIT_COMMIT"):
        os.environ.pop(_name)


@pytest.fixture
def tree(tmp_path: Path) -> dict[str, Path]:
    """A fresh fixture tree per test."""
    return build_fixture_tree(tmp_path / "fixture")


@pytest.fixture
def cfg(tree: dict[str, Path]) -> ModelConfig:
    """Config of the per-test fixture tree."""
    return load_config(tree["config"])


@pytest.fixture(scope="session")
def part1_tree(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One fixture tree on which Part 1 (train) has run; shared read-only by several tests."""
    from vibespace_model.train import run_part1

    paths = build_fixture_tree(tmp_path_factory.mktemp("part1") / "fixture")
    config = load_config(paths["config"])
    summary = run_part1(config)
    return {"paths": paths, "cfg": config, "summary": summary}


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator."""
    return np.random.default_rng(0)
