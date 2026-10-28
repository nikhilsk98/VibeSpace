"""Encode plot overviews into 384-dim vectors with ``all-MiniLM-L6-v2``.

The encoder is a small local text encoder used only to turn plot text into
vectors. Vectors are cached per (movieId, overview hash) and encoder version,
so reruns only encode new or changed overviews. Movies without an overview are
skipped and logged. Outputs:

* ``plot_embeddings.npy``           float32 array (n_movies x 384), L2-normalised
* ``plot_embeddings_index.parquet`` row -> movieId mapping (+ text hash)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from scripts.clean import CLEAN_FILE
from scripts.config import PipelineConfig
from scripts.exceptions import SchemaError
from scripts.io_utils import ensure_dir, read_parquet, save_numpy, text_sha1, write_json, write_parquet
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

EMBEDDINGS_FILE = "plot_embeddings.npy"
EMBEDDINGS_INDEX_FILE = "plot_embeddings_index.parquet"
ENCODE_SUMMARY_FILE = "encode_plots_summary.json"


class TextEncoder(Protocol):
    """Anything with a SentenceTransformer-style ``encode`` method."""

    def encode(self, sentences: list[str], batch_size: int = ..., **kwargs: Any) -> np.ndarray:
        """Return one vector per sentence."""


def encoder_version(model_name: str) -> str:
    """Filesystem-safe identifier of the encoder, used to key the cache."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name)


def load_encoder(model_name: str) -> TextEncoder:
    """Load the sentence-transformer on CPU (downloads weights on first use)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def build_plot_texts(catalog: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (movieId/text/text_sha1 for movies with an overview, number skipped)."""
    text = catalog["overview"].fillna("").astype(str).str.strip()
    has_text = text.ne("")
    out = pd.DataFrame({"movieId": catalog.loc[has_text, "movieId"].astype("int64"),
                        "text": text[has_text]}).reset_index(drop=True)
    out["text_sha1"] = out["text"].map(text_sha1)
    return out, int((~has_text).sum())


def load_cache(path: Path, dim: int) -> dict[tuple[int, str], np.ndarray]:
    """Read cached vectors keyed by (movieId, text hash); bad caches are ignored."""
    if not Path(path).is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            ids, hashes, vectors = data["movie_ids"], data["text_sha1"], data["vectors"]
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Embedding cache %s unreadable (%s); starting fresh", Path(path).name, exc)
        return {}
    if vectors.ndim != 2 or vectors.shape[1] != dim:
        logger.warning("Embedding cache has dim %s, expected %d; ignoring it", vectors.shape, dim)
        return {}
    return {(int(i), str(h)): v for i, h, v in zip(ids, hashes, vectors)}


def save_cache(path: Path, cache: dict[tuple[int, str], np.ndarray], dim: int) -> None:
    """Write the cache as a compressed ``.npz``."""
    ensure_dir(Path(path).parent)
    keys = sorted(cache)
    vectors = np.stack([cache[k] for k in keys]).astype(np.float32) if keys else np.zeros((0, dim), np.float32)
    tmp = Path(path).with_name(Path(path).stem + ".tmp.npz")
    np.savez_compressed(tmp, movie_ids=np.array([k[0] for k in keys], dtype=np.int64),
                        text_sha1=np.array([k[1] for k in keys], dtype="<U40"), vectors=vectors)
    tmp.replace(path)


def encode_texts(encoder: TextEncoder, texts: list[str], batch_size: int, dim: int) -> np.ndarray:
    """Encode texts in batches and return L2-normalised float32 vectors."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    vectors = np.asarray(encoder.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                        convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(texts), dim):
        raise SchemaError(f"Encoder returned shape {vectors.shape}, expected ({len(texts)}, {dim})")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def encode_plots(catalog: pd.DataFrame, encoder_factory: Any, model_name: str, cache_path: Path | None,
                 batch_size: int = 64, dim: int = 384, use_cache: bool = True
                 ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Return (embeddings, index, summary). ``encoder_factory`` is only called if needed."""
    texts, skipped = build_plot_texts(catalog)
    if skipped:
        logger.info("Skipping %d movies without an overview", skipped)
    # ``use_cache=False`` re-encodes everything but still merges into (never shrinks) the stored cache.
    stored = load_cache(cache_path, dim) if cache_path else {}
    cache = dict(stored) if use_cache else {}
    keys = list(zip(texts["movieId"].tolist(), texts["text_sha1"].tolist()))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    logger.info("Plot encoding: %d texts, %d cached, %d to encode (batch=%d)",
                len(keys), len(keys) - len(todo), len(todo), batch_size)
    if todo:
        encoder = encoder_factory(model_name)
        new_vectors = encode_texts(encoder, [texts["text"].iat[i] for i in todo], batch_size, dim)
        for i, vec in zip(todo, new_vectors):
            cache[keys[i]] = vec
            stored[keys[i]] = vec
        if cache_path:
            save_cache(cache_path, stored, dim)
    embeddings = (np.stack([cache[k] for k in keys]).astype(np.float32) if keys
                  else np.zeros((0, dim), dtype=np.float32))
    index = texts[["movieId", "text_sha1"]].copy()
    index.insert(0, "row", np.arange(len(index), dtype=np.int64))
    index["encoder"] = model_name
    summary = {"movies": int(len(catalog)), "encoded_rows": int(len(keys)), "newly_encoded": len(todo),
               "from_cache": len(keys) - len(todo), "skipped_missing_overview": skipped,
               "dim": dim, "encoder": model_name}
    return embeddings, index, summary


def run_encode_plots(cfg: PipelineConfig, encoder_factory: Any = None) -> dict[str, Any]:
    """Encode the clean catalog's overviews and write embeddings + index."""
    encoder_factory = encoder_factory or load_encoder
    catalog = read_parquet(cfg.processed_dir / CLEAN_FILE, columns=["movieId", "overview"])
    cache_path = cfg.cache_dir / "embeddings" / f"{encoder_version(cfg.encoder_model)}.npz"
    embeddings, index, summary = encode_plots(catalog, encoder_factory, cfg.encoder_model, cache_path,
                                              cfg.encoder_batch_size, cfg.embedding_dim, cfg.use_cache)
    save_numpy(cfg.processed_dir / EMBEDDINGS_FILE, embeddings)
    write_parquet(cfg.processed_dir / EMBEDDINGS_INDEX_FILE, index)
    write_json(cfg.processed_dir / ENCODE_SUMMARY_FILE, summary)
    logger.info("Plot embeddings: %s", summary)
    return summary
