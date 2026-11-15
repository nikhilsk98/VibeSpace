"""Feature matrix for the Movie DNA predictor.

Inputs are the plot embedding plus metadata known at prediction time. Every scaler and encoder is fitted
on the training split only and saved with the model (as plain JSON). The Tag Genome is only the source of
the labels: :func:`assert_no_leakage` fails loudly if a genome, relevance or label column reaches the
features.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vibespace_model.config import DNA_DIMENSIONS
from vibespace_model.errors import LeakageError

LOGGER = logging.getLogger(__name__)

FEATURE_GROUPS: tuple[str, ...] = (
    "plot_embedding",
    "genres",
    "runtime",
    "year",
    "language",
    "popularity",
    "flags",
)

# Genome tags, relevance scores, raw DNA scores and "has genome" markers must never be inputs.
_LEAKAGE_PATTERNS = (
    re.compile(r"genome", re.IGNORECASE),
    re.compile(r"relevance", re.IGNORECASE),
    re.compile(r"^tag_?id$", re.IGNORECASE),
    re.compile(r"^tags?$", re.IGNORECASE),
    re.compile(r"^tag[_:=]", re.IGNORECASE),
    re.compile(rf"^({'|'.join(DNA_DIMENSIONS)})(_raw)?$", re.IGNORECASE),
    re.compile(r"mapping_version", re.IGNORECASE),
)


def assert_no_leakage(columns: Iterable[str], targets: Iterable[str] = ()) -> None:
    """Raise :class:`LeakageError` if any column looks like a Tag Genome, relevance or label column."""
    target_set = {t.lower() for t in targets}
    bad = sorted(
        str(c)
        for c in columns
        if str(c).lower() in target_set or any(p.search(str(c)) for p in _LEAKAGE_PATTERNS)
    )
    if bad:
        raise LeakageError(
            f"Tag Genome / label columns must never be model inputs, found: {bad[:10]}"
            + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        )


def _as_genre_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[|,]", value)
        return [p.strip().lower() for p in parts if p.strip() and p.strip() != "(no genres listed)"]
    if isinstance(value, float) and np.isnan(value):
        return []
    try:
        return [str(v).strip().lower() for v in value if str(v).strip()]
    except TypeError:
        return []


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()


def _bool_column(catalog: pd.DataFrame, column: str) -> np.ndarray:
    if column not in catalog.columns:
        return np.zeros(len(catalog), dtype=np.float64)
    return catalog[column].fillna(False).astype(bool).to_numpy(dtype=np.float64)


def _scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / (std if std > 0 else 1.0)


class FeatureBuilder:
    """Fit on the training rows, then turn any catalog + embedding rows into the model matrix."""

    def __init__(self, feature_cfg: dict[str, Any], embedding_dim: int = 384) -> None:
        self.cfg = dict(feature_cfg)
        self.embedding_dim = int(embedding_dim)
        self.state: dict[str, Any] = {}
        self.feature_names: list[str] = []
        self.groups: dict[str, list[int]] = {}

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` (or :meth:`load`) has run."""
        return bool(self.state)

    # ------------------------------------------------------------------ fitting (train rows only)

    def fit(self, catalog: pd.DataFrame, embeddings: np.ndarray) -> FeatureBuilder:
        """Learn vocabularies, fill values and scaling statistics from the training rows."""
        self._check_inputs(catalog, embeddings)
        cfg = self.cfg
        genres = sorted({g for v in catalog[cfg["genres_column"]] for g in _as_genre_list(v)})

        runtime = self._valid_runtime(catalog)
        valid_rt = runtime[~np.isnan(runtime)]
        rt_fill = float(np.median(valid_rt)) if valid_rt.size else 100.0
        rt_lo, rt_hi = np.quantile(valid_rt, [0.01, 0.99]) if valid_rt.size else (rt_fill, rt_fill)
        rt_train = np.clip(np.where(np.isnan(runtime), rt_fill, runtime), rt_lo, rt_hi)

        year = _numeric(catalog[cfg["year_column"]])
        year_fill = float(np.nanmedian(year)) if np.isfinite(year).any() else 2000.0
        year_train = np.where(np.isnan(year), year_fill, year)

        pop_stats = {}
        for column in [*cfg.get("popularity_columns", []), *cfg.get("extra_numeric_columns", [])]:
            raw = self._popularity_raw(catalog, column)
            fill = float(np.nanmedian(raw)) if np.isfinite(raw).any() else 0.0
            filled = np.where(np.isnan(raw), fill, raw)
            pop_stats[column] = {"fill": fill, "mean": float(filled.mean()), "std": float(filled.std())}

        languages = sorted(catalog[cfg["language_column"]].fillna("unknown").astype(str).unique())
        flags = [c for c in cfg.get("flag_columns", []) if c in catalog.columns]
        missing_flags = sorted(set(cfg.get("flag_columns", [])) - set(flags))
        if missing_flags:
            LOGGER.warning("Flag columns not in the catalog (left out): %s", missing_flags)

        self.state = {
            "genres": genres,
            "runtime": {
                "fill": rt_fill,
                "clip": [float(rt_lo), float(rt_hi)],
                "mean": float(rt_train.mean()),
                "std": float(rt_train.std()),
            },
            "year": {"fill": year_fill, "mean": float(year_train.mean()), "std": float(year_train.std())},
            "popularity": pop_stats,
            "languages": languages,
            "flags": flags,
        }
        self._build_names()
        return self

    # ------------------------------------------------------------------ transforming

    def transform(self, catalog: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """Feature matrix (float32) in the fixed column order of :attr:`feature_names`."""
        if not self.is_fitted:
            raise RuntimeError("FeatureBuilder.transform called before fit")
        self._check_inputs(catalog, embeddings)
        cfg, st = self.cfg, self.state
        n = len(catalog)
        blocks: list[np.ndarray] = [np.asarray(embeddings, dtype=np.float64)]

        genre_index = {g: i for i, g in enumerate(st["genres"])}
        multi_hot = np.zeros((n, len(genre_index)))
        for row, value in enumerate(catalog[cfg["genres_column"]]):
            for genre in _as_genre_list(value):
                if genre in genre_index:  # unseen genres are ignored
                    multi_hot[row, genre_index[genre]] = 1.0
        blocks.append(multi_hot)

        rt = st["runtime"]
        runtime = self._valid_runtime(catalog)
        runtime = np.clip(np.where(np.isnan(runtime), rt["fill"], runtime), *rt["clip"])
        blocks.append(_scale(runtime, rt["mean"], rt["std"])[:, None])

        yr = st["year"]
        year = _numeric(catalog[cfg["year_column"]])
        year_missing = np.isnan(year).astype(np.float64)
        blocks.append(
            np.column_stack(
                [_scale(np.where(year_missing > 0, yr["fill"], year), yr["mean"], yr["std"]), year_missing]
            )
        )

        lang = catalog[cfg["language_column"]].fillna("unknown").astype(str).to_numpy()
        blocks.append(
            np.column_stack([(lang == code).astype(np.float64) for code in st["languages"]])
            if st["languages"]
            else np.zeros((n, 0))
        )

        pop_cols = []
        for column, stats in st["popularity"].items():
            raw = self._popularity_raw(catalog, column)
            pop_cols.append(_scale(np.where(np.isnan(raw), stats["fill"], raw), stats["mean"], stats["std"]))
        blocks.append(np.column_stack(pop_cols) if pop_cols else np.zeros((n, 0)))

        blocks.append(
            np.column_stack([_bool_column(catalog, c) for c in st["flags"]])
            if st["flags"]
            else np.zeros((n, 0))
        )

        matrix = np.hstack(blocks).astype(np.float32)
        if matrix.shape[1] != len(self.feature_names):
            raise RuntimeError("feature matrix width does not match feature names")  # pragma: no cover
        return matrix

    def fit_transform(self, catalog: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """:meth:`fit` on these rows, then :meth:`transform` them."""
        return self.fit(catalog, embeddings).transform(catalog, embeddings)

    # ------------------------------------------------------------------ persistence

    def save(self, path: Path) -> None:
        """Write the fitted state as JSON."""
        payload = {"cfg": self.cfg, "embedding_dim": self.embedding_dim, "state": self.state}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FeatureBuilder:
        """Rebuild a fitted builder from :meth:`save` output."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        builder = cls(payload["cfg"], payload["embedding_dim"])
        builder.state = payload["state"]
        builder._build_names()
        return builder

    # ------------------------------------------------------------------ helpers

    def group_of(self, name: str) -> str:
        """Feature group of one feature name."""
        for group, members in self.groups.items():
            if name in {self.feature_names[i] for i in members}:
                return group
        raise KeyError(name)

    def _build_names(self) -> None:
        st = self.state
        named: list[tuple[str, list[str]]] = [
            ("plot_embedding", [f"emb_{i:03d}" for i in range(self.embedding_dim)]),
            ("genres", [f"genre={g}" for g in st["genres"]]),
            ("runtime", ["runtime_scaled"]),
            ("year", ["release_year_scaled", "release_year_missing"]),
            ("language", [f"language={code}" for code in st["languages"]]),
            ("popularity", [self._popularity_name(c) for c in st["popularity"]]),
            ("flags", list(st["flags"])),
        ]
        self.feature_names, self.groups, start = [], {}, 0
        for group, names in named:
            self.feature_names.extend(names)
            self.groups[group] = list(range(start, start + len(names)))
            start += len(names)
        assert_no_leakage(self.feature_names)

    def _popularity_name(self, column: str) -> str:
        return f"log1p_{column}" if column in self.cfg.get("popularity_columns", []) else f"{column}_scaled"

    def _popularity_raw(self, catalog: pd.DataFrame, column: str) -> np.ndarray:
        values = _numeric(catalog[column]) if column in catalog.columns else np.full(len(catalog), np.nan)
        if column in self.cfg.get("popularity_columns", []):
            values = np.log1p(np.clip(values, 0.0, None))
        return values

    def _valid_runtime(self, catalog: pd.DataFrame) -> np.ndarray:
        runtime = _numeric(catalog[self.cfg["runtime_column"]])
        invalid = ~np.isfinite(runtime) | (runtime <= 0) | (_bool_column(catalog, "flag_missing_runtime") > 0)
        return np.where(invalid, np.nan, runtime)

    def _check_inputs(self, catalog: pd.DataFrame, embeddings: np.ndarray) -> None:
        assert_no_leakage(catalog.columns, [])
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"embeddings must have shape (n, {self.embedding_dim}), got {embeddings.shape}")
        if len(catalog) != embeddings.shape[0]:
            raise ValueError("catalog and embeddings have a different number of rows")
