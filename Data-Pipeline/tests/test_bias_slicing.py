"""Tests for scripts.bias_slicing (synthetic data with a built-in slice gap)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts import bias_slicing as bs
from scripts.build_dna_labels import DNA_DIMENSIONS, DNA_LABELS_FILE
from scripts.clean import CLEAN_FILE
from scripts.encode_plots import EMBEDDINGS_FILE, EMBEDDINGS_INDEX_FILE
from scripts.exceptions import PipelineError
from scripts.io_utils import save_numpy, write_parquet


def synthetic(n: int = 1000, seed: int = 0, dim: int = 16):
    """English movies follow the embedding signal; 'other' movies are noisier (harder)."""
    rng = np.random.default_rng(seed)
    lang = np.where(rng.uniform(size=n) < 0.7, "english", "other_international")
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    w = rng.normal(size=(dim, len(DNA_DIMENSIONS)))
    signal = 1 / (1 + np.exp(-emb @ w * 0.3))
    noise = np.where(lang[:, None] == "english", 0.03, 0.25) * rng.normal(size=signal.shape)
    y = np.clip(signal + noise, 0, 1)
    catalog = pd.DataFrame({
        "movieId": np.arange(n), "language_group": lang,
        "era": rng.choice(["pre_1980", "1980_1999", "2000_plus"], size=n),
        "primary_genre": rng.choice(["drama", "comedy", "horror"], size=n),
        "popularity_bucket": rng.choice(["low", "mid", "high"], size=n),
        "runtime": rng.uniform(80, 150, size=n), "year": pd.array(rng.integers(1950, 2020, size=n), dtype="Int64"),
        "vote_count": rng.integers(0, 10000, size=n).astype(float), "vote_average": rng.uniform(3, 9, size=n),
    })
    labels = pd.DataFrame(y, columns=list(DNA_DIMENSIONS))
    labels.insert(0, "movieId", np.arange(n))
    index = pd.DataFrame({"row": np.arange(n), "movieId": np.arange(n)})
    return catalog, labels, emb, index


def test_representation_and_coverage():
    catalog, labels, _, _ = synthetic(100)
    rep = bs.representation(catalog, labels.iloc[:50])
    lang = rep["language_group"]
    assert lang["movies"].sum() == 100
    assert lang["with_labels"].sum() == 50
    assert ((lang["label_coverage"] >= 0) & (lang["label_coverage"] <= 1)).all()


def test_inverse_frequency_weights():
    groups = pd.DataFrame({"a": ["x"] * 9 + ["y"]})
    w = bs.inverse_frequency_weights(groups)
    assert w.mean() == pytest.approx(1.0) and w[-1] > w[0]
    assert w.max() / w.min() <= bs.WEIGHT_CLIP[1] / bs.WEIGHT_CLIP[0]


def test_resample_indices_oversamples_small_groups():
    groups = pd.Series(["a"] * 50 + ["b"] * 40 + ["c"] * 2)
    idx = bs.resample_indices(groups, seed=0)
    counts = groups.iloc[idx].value_counts()
    assert counts["c"] >= 20 and counts["a"] == 50


def test_split_is_70_15_15_and_disjoint():
    tr, va, te = bs.split_by_movie(200, seed=1)
    assert len(tr) == 140 and len(va) == 30 and len(te) == 30
    assert not (set(tr) & set(va)) and not (set(va) & set(te))


def test_build_features_excludes_tag_genome():
    catalog, labels, emb, _ = synthetic(20)
    x = bs.build_features(catalog.assign(runtime=np.nan), emb)
    assert x.shape[0] == 20 and np.isfinite(x).all()
    assert x.shape[1] == emb.shape[1] + 4 + catalog["language_group"].nunique()


def test_analysis_detects_gap_and_reports():
    catalog, labels, emb, index = synthetic()
    result = bs.run_bias_analysis(catalog, labels, emb, index, seed=0)
    base = result["runs"]["baseline"]
    lang = base.by_slice["language_group"]
    assert lang.loc["other_international", "mae"] > lang.loc["english", "mae"]
    assert "other_international" in bs.significant_slices(base).get("language_group", [])
    assert set(result["runs"]) == {"baseline", "reweighted", "resampled"}
    payload = bs.metrics_payload(result)
    assert set(payload["overall_mae"]) == {"baseline", "reweighted", "resampled"}
    text = bs.render_report(result)
    assert "Why slices are movie attributes" in text and "Trade-off" in text and "| baseline |" in text


def test_report_on_small_sample_has_no_judged_slices():
    catalog, labels, emb, index = synthetic(60)
    text = bs.render_report(bs.run_bias_analysis(catalog, labels, emb, index))
    assert "too small to compare" in text


def test_too_few_movies():
    catalog, labels, emb, index = synthetic(30)
    with pytest.raises(PipelineError, match="at least"):
        bs.run_bias_analysis(catalog, labels, emb, index)


def test_prepare_dataset_aligns_rows():
    catalog, labels, emb, index = synthetic(60)
    index = index.iloc[::-1].reset_index(drop=True)  # rows in a different order
    data, aligned = bs.prepare_dataset(catalog, labels.iloc[:50], emb, index)
    assert len(data) == 50
    assert np.allclose(aligned[0], emb[int(data["movieId"].iloc[0])])


def test_run_bias_slicing_writes_outputs(cfg):
    catalog, labels, emb, index = synthetic(200)
    write_parquet(cfg.processed_dir / CLEAN_FILE, catalog)
    write_parquet(cfg.processed_dir / DNA_LABELS_FILE, labels)
    save_numpy(cfg.processed_dir / EMBEDDINGS_FILE, emb)
    write_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE, index)
    out = bs.run_bias_slicing(cfg)
    assert "baseline" in out["overall_mae"]
    assert (cfg.reports_dir / bs.BIAS_REPORT_FILE).is_file()
    assert (cfg.metrics_dir / "bias" / bs.BIAS_METRICS_FILE).is_file()


def test_sample_run_keeps_committed_report(cfg):
    sample_cfg = replace(cfg, sample_size=50)
    catalog, labels, emb, index = synthetic(200)
    # a sample run reads and writes the sample tree (data/sample/...), never the full outputs
    write_parquet(sample_cfg.processed_dir / CLEAN_FILE, catalog)
    write_parquet(sample_cfg.processed_dir / DNA_LABELS_FILE, labels)
    save_numpy(sample_cfg.processed_dir / EMBEDDINGS_FILE, emb)
    write_parquet(sample_cfg.processed_dir / EMBEDDINGS_INDEX_FILE, index)
    bs.run_bias_slicing(sample_cfg)
    assert (cfg.reports_dir / "sample" / bs.BIAS_REPORT_FILE).is_file()
    assert not (cfg.reports_dir / bs.BIAS_REPORT_FILE).exists()
    assert (sample_cfg.metrics_dir / "bias" / bs.BIAS_METRICS_FILE).is_file()
    assert not (cfg.metrics_dir / "bias").exists()
