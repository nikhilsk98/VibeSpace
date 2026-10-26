"""Tests for scripts.encode_plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import encode_plots as ep
from scripts.clean import CLEAN_FILE
from scripts.exceptions import SchemaError
from scripts.io_utils import write_parquet
from tests.helpers import FakeEncoder


def catalog(overviews):
    return pd.DataFrame({"movieId": range(1, len(overviews) + 1), "overview": overviews})


def test_build_plot_texts_skips_missing():
    texts, skipped = ep.build_plot_texts(catalog(["A plot.", None, "  ", "Another plot."]))
    assert texts["movieId"].tolist() == [1, 4] and skipped == 2
    assert texts["text_sha1"].str.len().eq(40).all()


def test_encoder_version_is_path_safe():
    assert ep.encoder_version("sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers__all-MiniLM-L6-v2"


def test_encode_texts_shapes_and_norms():
    vecs = ep.encode_texts(FakeEncoder(), ["a", "b"], batch_size=1, dim=384)
    assert vecs.shape == (2, 384) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert ep.encode_texts(FakeEncoder(), [], 8, 384).shape == (0, 384)


def test_encode_texts_wrong_dim():
    with pytest.raises(SchemaError):
        ep.encode_texts(FakeEncoder(dim=10), ["a"], 8, 384)


def test_cache_reuse_and_invalidation(tmp_path):
    cache = tmp_path / "c.npz"
    enc = FakeEncoder()
    factory_calls = []

    def factory(name):
        factory_calls.append(name)
        return enc

    emb1, index1, s1 = ep.encode_plots(catalog(["one", "two", None]), factory, "m", cache)
    assert s1["newly_encoded"] == 2 and s1["skipped_missing_overview"] == 1
    emb2, _, s2 = ep.encode_plots(catalog(["one", "two", None]), factory, "m", cache)
    assert s2["newly_encoded"] == 0 and len(factory_calls) == 1  # encoder not even loaded
    assert np.allclose(emb1, emb2)
    _, _, s3 = ep.encode_plots(catalog(["one", "changed", None]), factory, "m", cache)
    assert s3["newly_encoded"] == 1  # changed overview -> new hash -> re-encoded
    _, _, s4 = ep.encode_plots(catalog(["one", "two", None]), factory, "m", cache, use_cache=False)
    assert s4["newly_encoded"] == 2
    assert index1["row"].tolist() == [0, 1] and index1["movieId"].tolist() == [1, 2]


def test_no_cache_run_does_not_shrink_cache(tmp_path):
    cache = tmp_path / "c.npz"
    ep.encode_plots(catalog(["a", "b", "c"]), lambda n: FakeEncoder(), "m", cache)
    ep.encode_plots(catalog(["a"]), lambda n: FakeEncoder(), "m", cache, use_cache=False)
    assert len(ep.load_cache(cache, 384)) == 3
    _, _, warm = ep.encode_plots(catalog(["a", "b", "c"]), lambda n: FakeEncoder(), "m", cache)
    assert warm["newly_encoded"] == 0


def test_bad_cache_is_ignored(tmp_path):
    bad = tmp_path / "c.npz"
    bad.write_bytes(b"garbage")
    assert ep.load_cache(bad, 384) == {}
    np.savez(tmp_path / "d.npz", movie_ids=np.array([1]), text_sha1=np.array(["x"]), vectors=np.zeros((1, 5)))
    assert ep.load_cache(tmp_path / "d.npz", 384) == {}
    assert ep.load_cache(tmp_path / "missing.npz", 384) == {}


def test_empty_catalog(tmp_path):
    emb, index, summary = ep.encode_plots(catalog([]).astype({"overview": object}), lambda n: FakeEncoder(), "m",
                                          tmp_path / "c.npz")
    assert emb.shape == (0, 384) and index.empty and summary["encoded_rows"] == 0


def test_run_encode_plots(cfg):
    write_parquet(cfg.processed_dir / CLEAN_FILE, catalog(["plot a", None, "plot c"]))
    summary = ep.run_encode_plots(cfg, encoder_factory=lambda name: FakeEncoder())
    emb = np.load(cfg.processed_dir / ep.EMBEDDINGS_FILE)
    index = pd.read_parquet(cfg.processed_dir / ep.EMBEDDINGS_INDEX_FILE)
    assert emb.shape == (2, 384) and index["movieId"].tolist() == [1, 3]
    assert summary["skipped_missing_overview"] == 1
    assert list((cfg.cache_dir / "embeddings").glob("*.npz"))
