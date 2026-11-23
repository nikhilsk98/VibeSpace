"""Load and verify the data pipeline outputs.

The four processed files are checked against ``Data-Pipeline/dvc.lock`` (DVC MD5 hashes) so every run is
tied to a DVC-versioned dataset. Labels, catalog and embeddings are then joined on ``movieId``; every
dropped movie is counted with its reason. Movies without an embedding are excluded, never zero-filled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess  # noqa: S404  (git rev-parse only, fixed arguments)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vibespace_model.config import ModelConfig
from vibespace_model.errors import DataValidationError, DataVersionError
from vibespace_model.features import assert_no_leakage

LOGGER = logging.getLogger(__name__)
_CHUNK = 1 << 20


@dataclass
class Dataset:
    """Joined, validated data: one row per movie, aligned across all fields."""

    catalog: pd.DataFrame
    labels: np.ndarray
    embeddings: np.ndarray
    join_report: dict[str, Any]
    data_version: dict[str, Any]
    targets: list[str] = field(default_factory=list)

    @property
    def movie_ids(self) -> np.ndarray:
        """``movieId`` of every row."""
        return self.catalog["movieId"].to_numpy()

    def __len__(self) -> int:
        return len(self.catalog)

    def subset(self, mask_or_index: np.ndarray) -> Dataset:
        """Rows selected by a boolean mask or an integer index."""
        index = np.flatnonzero(mask_or_index) if mask_or_index.dtype == bool else mask_or_index
        return Dataset(
            catalog=self.catalog.iloc[index].reset_index(drop=True),
            labels=self.labels[index],
            embeddings=self.embeddings[index],
            join_report=self.join_report,
            data_version=self.data_version,
            targets=self.targets,
        )


# ---------------------------------------------------------------------------- hashing (DVC compatible)


def file_md5(path: Path) -> str:
    """MD5 of a file's bytes, as DVC 3 records it in ``dvc.lock``."""
    digest = hashlib.md5()  # noqa: S324  (content fingerprint, not security)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_md5(path: Path) -> str:
    """DVC hash of a directory output: MD5 of the sorted JSON file listing, with a ``.dir`` suffix."""
    entries = [
        {"md5": file_md5(item), "relpath": item.relative_to(path).as_posix()}
        for item in sorted(p for p in path.rglob("*") if p.is_file())
    ]
    entries.sort(key=lambda entry: entry["relpath"])
    payload = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.md5(payload).hexdigest() + ".dir"  # noqa: S324


def output_md5(path: Path) -> str:
    """DVC-style hash of a file or directory output."""
    return dir_md5(path) if path.is_dir() else file_md5(path)


def read_lock_hashes(lock_path: Path) -> dict[str, str]:
    """``{output path: md5}`` for every stage output listed in ``dvc.lock``."""
    if not lock_path.is_file():
        raise DataVersionError(f"dvc.lock not found: {lock_path}")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    hashes: dict[str, str] = {}
    for stage in (lock.get("stages") or {}).values():
        for out in stage.get("outs") or []:
            if "path" in out and "md5" in out:
                hashes[str(out["path"])] = str(out["md5"])
    if not hashes:
        raise DataVersionError(f"no outputs with MD5 hashes in {lock_path}")
    return hashes


def verify_hashes(cfg: ModelConfig) -> dict[str, str]:
    """Check each hash-checked file against ``dvc.lock``; return ``{lock path: md5}``."""
    lock_hashes = read_lock_hashes(cfg.paths.dvc_lock)
    try:
        prefix = cfg.paths.processed_dir.relative_to(cfg.paths.data_pipeline_dir)
    except ValueError as exc:
        raise DataVersionError(
            f"processed_dir {cfg.paths.processed_dir} is not inside {cfg.paths.data_pipeline_dir}, "
            "so it cannot be matched with dvc.lock"
        ) from exc

    verified: dict[str, str] = {}
    for name in cfg.data["hash_checked_files"]:
        lock_key = (prefix / name).as_posix()
        local = cfg.paths.processed_dir / name
        if lock_key not in lock_hashes:
            raise DataVersionError(f"{lock_key} is not an output in {cfg.paths.dvc_lock}")
        if not local.exists():
            raise DataVersionError(f"missing pipeline output {local}; run `dvc pull` in the repository")
        actual = output_md5(local)
        if actual != lock_hashes[lock_key]:
            raise DataVersionError(
                f"hash mismatch for {lock_key}: dvc.lock has {lock_hashes[lock_key]}, file has {actual}. "
                "Run `dvc pull` (or `dvc status`) so the data matches the committed version."
            )
        verified[lock_key] = actual
    LOGGER.info("Hash check passed for %d files against %s", len(verified), cfg.paths.dvc_lock)
    return verified


def _git(args: list[str], root: Path) -> str:
    """Output of a read-only git command run in ``root``."""
    command = ["git", *args]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
    return result.stdout.strip()


def git_commit(root: Path) -> dict[str, Any]:
    """Current Git commit (``GIT_COMMIT`` env var wins, e.g. inside Docker) and whether the tree is dirty."""
    env_commit = os.environ.get("GIT_COMMIT")
    if env_commit:
        return {"git_commit": env_commit, "git_dirty": None}
    try:
        commit = _git(["rev-parse", "HEAD"], root)
        dirty = _git(["status", "--porcelain"], root)
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(dirty)}


# ---------------------------------------------------------------------------- loading + validation


def _read_inputs(cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    processed = cfg.paths.processed_dir
    files = {key: processed / cfg.data[f"{key}_file"] for key in ("catalog", "labels", "embedding_index")}
    files["embeddings"] = processed / cfg.data["embeddings_file"]
    for name, path in files.items():
        if not path.is_file():
            raise DataValidationError(f"missing {name} file: {path}; run `dvc pull` in the repository")
    catalog = pd.read_parquet(files["catalog"])
    labels = pd.read_parquet(files["labels"])
    embeddings = np.load(files["embeddings"])
    index = pd.read_parquet(files["embedding_index"])
    return catalog, labels, embeddings, index


def validate_labels(labels: pd.DataFrame, targets: list[str]) -> None:
    """12 label columns present, numeric, without gaps, in [0, 1], one row per movie."""
    missing = [c for c in ["movieId", *targets] if c not in labels.columns]
    if missing:
        raise DataValidationError(f"labels file is missing columns: {missing}")
    if labels["movieId"].duplicated().any():
        dupes = labels.loc[labels["movieId"].duplicated(), "movieId"].head(5).tolist()
        raise DataValidationError(f"duplicate movieIds in labels: {dupes}")
    values = labels[targets].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise DataValidationError("labels contain missing values")
    if values.min() < 0.0 or values.max() > 1.0:
        bad = [t for t in targets if labels[t].min() < 0.0 or labels[t].max() > 1.0]
        raise DataValidationError(f"labels outside [0, 1] in: {bad}")


def validate_catalog(catalog: pd.DataFrame, required: list[str]) -> None:
    """``movieId`` unique and every column needed for features and slices present."""
    if "movieId" not in catalog.columns:
        raise DataValidationError("catalog has no movieId column")
    if catalog["movieId"].duplicated().any():
        dupes = catalog.loc[catalog["movieId"].duplicated(), "movieId"].head(5).tolist()
        raise DataValidationError(f"duplicate movieIds in catalog: {dupes}")
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise DataValidationError(
            f"catalog is missing columns the model needs: {missing}. Check the data pipeline outputs, or set "
            "features.allow_missing_columns: true to train without missing flag columns on purpose."
        )


def validate_embeddings(embeddings: np.ndarray, index: pd.DataFrame, dim: int) -> None:
    """Embedding matrix of the expected width whose rows match the index file one to one."""
    if embeddings.ndim != 2 or embeddings.shape[1] != dim:
        raise DataValidationError(f"embeddings must have shape (n, {dim}), got {embeddings.shape}")
    if not {"row", "movieId"} <= set(index.columns):
        raise DataValidationError("embedding index needs 'row' and 'movieId' columns")
    if len(index) != embeddings.shape[0]:
        raise DataValidationError(
            f"embedding index has {len(index)} rows but the matrix has {embeddings.shape[0]}"
        )
    if not np.array_equal(index["row"].to_numpy(), np.arange(len(index))):
        raise DataValidationError("embedding index 'row' column is not 0..n-1 in order")
    if index["movieId"].duplicated().any():
        raise DataValidationError("duplicate movieIds in the embedding index")
    if not np.isfinite(embeddings).all():
        raise DataValidationError("embeddings contain NaN or infinite values")


def _required_catalog_columns(cfg: ModelConfig) -> list[str]:
    feats = cfg.features
    columns = [
        feats["genres_column"],
        feats["runtime_column"],
        feats["year_column"],
        feats["language_column"],
    ]
    columns += list(feats.get("popularity_columns", [])) + list(feats.get("extra_numeric_columns", []))
    if not feats.get("allow_missing_columns", False):
        columns += list(feats.get("flag_columns", []))
    columns += list(cfg.slices)
    return list(dict.fromkeys(columns))


def join_inputs(
    catalog: pd.DataFrame,
    labels: pd.DataFrame,
    embeddings: np.ndarray,
    index: pd.DataFrame,
    targets: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """labels ∩ catalog ∩ embeddings on ``movieId``, with drop counts per step."""
    label_ids = set(labels["movieId"])
    catalog_ids = set(catalog["movieId"])
    embedding_ids = set(index["movieId"])

    step1 = labels[labels["movieId"].isin(catalog_ids)]
    step2 = step1[step1["movieId"].isin(embedding_ids)]
    if step2.empty:
        raise DataValidationError("no movie has labels, a catalog row and an embedding")

    report = {
        "labels_total": len(labels),
        "catalog_total": len(catalog),
        "embeddings_total": len(index),
        "dropped_no_catalog_row": len(labels) - len(step1),
        "after_catalog_join": len(step1),
        "dropped_no_embedding": len(step1) - len(step2),
        "after_embedding_join": len(step2),
        "catalog_rows_without_labels": len(catalog_ids - label_ids),
        "drop_reasons": {
            "dropped_no_catalog_row": "labelled movie missing from movies_clean (TMDB join or cleaning)",
            "dropped_no_embedding": "no plot embedding for the movie (excluded, not zero-filled)",
        },
    }
    ordered = step2.sort_values("movieId").reset_index(drop=True)
    merged = catalog.set_index("movieId").loc[ordered["movieId"]].reset_index()
    rows = index.set_index("movieId").loc[ordered["movieId"], "row"].to_numpy()
    if not np.array_equal(merged["movieId"].to_numpy(), ordered["movieId"].to_numpy()):
        raise DataValidationError("catalog and labels are misaligned after the join")  # pragma: no cover
    return merged, ordered[targets].to_numpy(dtype=np.float64), embeddings[rows].astype(np.float32), report


def sample_rows(n_rows: int, sample_size: int | None, seed: int) -> np.ndarray:
    """Sorted random subset of row positions for quick ``--sample-size`` runs (all rows if None)."""
    if not sample_size or sample_size >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def load_dataset(cfg: ModelConfig, skip_hash_check: bool = False) -> Dataset:
    """Verify, load, validate and join the pipeline outputs into one aligned :class:`Dataset`."""
    hashes: dict[str, str] = {}
    if skip_hash_check:
        LOGGER.warning("Hash check skipped (--skip-hash-check): only valid for fixture data")
    else:
        hashes = verify_hashes(cfg)

    catalog, labels, embeddings, index = _read_inputs(cfg)
    assert_no_leakage(catalog.columns, cfg.targets)
    validate_labels(labels, cfg.targets)
    validate_catalog(catalog, _required_catalog_columns(cfg))
    validate_embeddings(embeddings, index, int(cfg.data["embedding_dim"]))

    merged, label_matrix, emb, report = join_inputs(catalog, labels, embeddings, index, cfg.targets)
    keep = sample_rows(len(merged), cfg.sample_size, cfg.random_seed)
    report["sample_size"] = int(len(keep)) if cfg.sample_size else None
    LOGGER.info(
        "Join: %d labelled, -%d without catalog row, -%d without embedding -> %d movies%s",
        report["labels_total"],
        report["dropped_no_catalog_row"],
        report["dropped_no_embedding"],
        report["after_embedding_join"],
        f" (sampled {len(keep)})" if cfg.sample_size else "",
    )

    mapping = sorted(labels["mapping_version"].astype(str).unique()) if "mapping_version" in labels else []
    data_version = {
        "dvc_lock_hashes": hashes,
        "hash_check": "skipped" if skip_hash_check else "passed",
        "mapping_version": ",".join(mapping) or "unknown",
        "embedding_encoder": str(index["encoder"].iloc[0])
        if "encoder" in index and len(index)
        else "unknown",
        **git_commit(cfg.root),
    }
    data_version["data_version_id"] = hashlib.sha256(
        json.dumps(
            {k: data_version[k] for k in ("dvc_lock_hashes", "mapping_version")}, sort_keys=True
        ).encode()
    ).hexdigest()[:16]

    return Dataset(
        catalog=merged.iloc[keep].reset_index(drop=True),
        labels=label_matrix[keep],
        embeddings=emb[keep],
        join_report=report,
        data_version=data_version,
        targets=list(cfg.targets),
    )
