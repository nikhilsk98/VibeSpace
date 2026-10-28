"""Tests for scripts.build_dna_labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import build_dna_labels as bd
from scripts.config import PIPELINE_ROOT
from scripts.exceptions import MappingError

REAL_MAPPING = PIPELINE_ROOT / "configs" / "movie_dna_mapping_v1.yaml"


def mapping_yaml(tmp_path, dims=None, allowed=None):
    """Write a mapping file; by default a valid one where each dimension uses its own tags."""
    if dims is None:
        dims = {d: {"positive": [f"{d}_a", f"{d}_b", f"{d}_c"], "inverse": [f"{d}_inv"]} for d in bd.DNA_DIMENSIONS}
    path = tmp_path / "mapping.yaml"
    path.write_text(yaml.safe_dump({"version": "vtest", "status": "test", "allowed_shared_tags": allowed or {},
                                    "dimensions": dims}))
    return path


def tags_for(mapping):
    tags = sorted(mapping.all_tags())
    return pd.DataFrame({"tagId": range(1, len(tags) + 1), "tag": tags})


def test_real_mapping_loads_and_has_all_dimensions():
    mapping = bd.load_mapping(REAL_MAPPING)
    assert mapping.version == "v1"
    assert set(mapping.positive) == set(bd.DNA_DIMENSIONS)
    for dim in bd.DNA_DIMENSIONS:
        assert 3 <= len(mapping.positive[dim]) <= 6


def test_real_mapping_tags_exist_in_fixture_genome(ml_cfg):
    genome_tags = pd.read_csv(ml_cfg.movielens_csv_dir / "genome-tags.csv")
    ids = bd.validate_mapping_tags(bd.load_mapping(REAL_MAPPING), genome_tags)
    assert len(ids) == len(bd.load_mapping(REAL_MAPPING).all_tags())


def test_unknown_tag_fails_loudly(tmp_path):
    mapping = bd.load_mapping(mapping_yaml(tmp_path))
    tags = tags_for(mapping)
    tags.loc[0, "tag"] = "something else"
    with pytest.raises(MappingError, match="do not exist"):
        bd.validate_mapping_tags(mapping, tags)


def test_tag_match_is_case_sensitive(tmp_path):
    mapping = bd.load_mapping(mapping_yaml(tmp_path))
    tags = tags_for(mapping)
    tags["tag"] = tags["tag"].str.upper()
    with pytest.raises(MappingError):
        bd.validate_mapping_tags(mapping, tags)


def test_missing_file_and_sections(tmp_path):
    with pytest.raises(MappingError, match="not found"):
        bd.load_mapping(tmp_path / "nope.yaml")
    (tmp_path / "empty.yaml").write_text("version: v1\n")
    with pytest.raises(MappingError, match="dimensions"):
        bd.load_mapping(tmp_path / "empty.yaml")


def test_missing_dimension(tmp_path):
    dims = {d: {"positive": [f"{d}1", f"{d}2", f"{d}3"]} for d in bd.DNA_DIMENSIONS[:-1]}
    with pytest.raises(MappingError, match="missing"):
        bd.load_mapping(mapping_yaml(tmp_path, dims))


@pytest.mark.parametrize("positive", [["a", "b"], ["a", "b", "c", "d", "e", "f", "g"], []])
def test_positive_tag_count_enforced(tmp_path, positive):
    dims = {d: {"positive": [f"{d}{i}" for i in range(3)]} for d in bd.DNA_DIMENSIONS}
    dims["humor"]["positive"] = positive
    with pytest.raises(MappingError, match="positive tags"):
        bd.load_mapping(mapping_yaml(tmp_path, dims))


def test_tag_both_positive_and_inverse(tmp_path):
    dims = {d: {"positive": [f"{d}{i}" for i in range(3)]} for d in bd.DNA_DIMENSIONS}
    dims["humor"]["inverse"] = ["humor0"]
    with pytest.raises(MappingError, match="positive and inverse"):
        bd.load_mapping(mapping_yaml(tmp_path, dims))


def test_shared_tag_needs_justification(tmp_path):
    dims = {d: {"positive": [f"{d}{i}" for i in range(3)]} for d in bd.DNA_DIMENSIONS}
    dims["pace"]["positive"][0] = "energy0"
    with pytest.raises(MappingError, match="without justification"):
        bd.load_mapping(mapping_yaml(tmp_path, dims))
    ok = bd.load_mapping(mapping_yaml(tmp_path, dims, allowed={"energy0": "fast action drives both"}))
    assert "energy0" in ok.positive["pace"]


def test_inverse_tag_math():
    """realistic 0.2 + fantasy (inverse) 0.9 -> mean(0.2, 0.1) = 0.15 before percentiles."""
    dims_pos = {d: [f"{d}_p"] for d in bd.DNA_DIMENSIONS}
    dims_pos["realism"] = ["realistic"]
    inverse = {d: [] for d in bd.DNA_DIMENSIONS}
    inverse["realism"] = ["fantasy"]
    mapping = bd.DnaMapping("v", "", dims_pos, inverse)
    ids = {t: i for i, t in enumerate(sorted(mapping.all_tags()), start=1)}
    relevance = pd.DataFrame({tid: [0.5] for tid in ids.values()}, index=[42])
    relevance[ids["realistic"]] = 0.2
    relevance[ids["fantasy"]] = 0.9
    raw = bd.compute_raw_scores(relevance, mapping, ids)
    assert raw.loc[42, "realism"] == pytest.approx(0.15)


def test_missing_tag_scores_detected():
    mapping = bd.DnaMapping("v", "", {d: ["a"] for d in bd.DNA_DIMENSIONS}, {d: [] for d in bd.DNA_DIMENSIONS})
    with pytest.raises(MappingError, match="have scores"):
        bd.compute_raw_scores(pd.DataFrame({2: [0.1]}), mapping, {"a": 1})


def test_percentiles_in_unit_interval_with_ties():
    raw = pd.DataFrame({"x": [0.1, 0.5, 0.5, 0.9]})
    pct = bd.to_percentiles(raw)["x"].tolist()
    assert pct[0] == 0.0 and pct[-1] == 1.0 and pct[1] == pct[2] == pytest.approx(0.5)
    assert bd.to_percentiles(pd.DataFrame({"x": [0.3]}))["x"].iloc[0] == 0.5
    assert bd.to_percentiles(pd.DataFrame({"x": []})).empty


def test_load_genome_relevance_filters_tags(ml_cfg):
    path = ml_cfg.movielens_csv_dir / "genome-scores.csv"
    wide = bd.load_genome_relevance(path, [1, 2, 3], block_size=4096)
    assert list(wide.columns) == [1, 2, 3]
    full = pd.read_csv(path)
    assert wide.shape[0] == full["movieId"].nunique()
    expected = full[(full.movieId == wide.index[0]) & (full.tagId == 2)]["relevance"].iloc[0]
    assert wide.iloc[0][2] == pytest.approx(expected)


def test_no_scores_for_tags(ml_cfg):
    with pytest.raises(MappingError, match="None of the mapped tags"):
        bd.load_genome_relevance(ml_cfg.movielens_csv_dir / "genome-scores.csv", [99999])


def test_run_build_dna_labels(ml_cfg):
    summary = bd.run_build_dna_labels(ml_cfg)
    labels = pd.read_parquet(ml_cfg.processed_dir / bd.DNA_LABELS_FILE)
    assert summary["movies_labelled"] == len(labels) == 58
    values = labels[list(bd.DNA_DIMENSIONS)].to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    assert (labels["mapping_version"] == "v1").all()
    assert labels["movieId"].is_unique
    report = (ml_cfg.reports_dir / "dna_sanity_report.md").read_text()
    assert "reviewed by the team" in report and "## humor" in report


def test_run_fails_on_unknown_tag(ml_cfg):
    tags = ml_cfg.movielens_csv_dir / "genome-tags.csv"
    tags.write_text(tags.read_text().replace("funny", "funy"))
    with pytest.raises(MappingError, match="funny"):
        bd.run_build_dna_labels(ml_cfg)
