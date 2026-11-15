"""Builder for small, synthetic pipeline outputs used by the tests (fully offline).

It writes a miniature ``Data-Pipeline/`` tree (the four processed files plus a ``dvc.lock`` with their
real MD5s) and a ``Model-Pipeline/configs/`` folder with tiny model, tuning and bias settings. Labels
depend on the embedding and the genres, so models can learn something and the gates are meaningful.

Run directly to inspect a fixture tree: ``python tests/fixtures/build_fixtures.py /tmp/vibespace-fixture``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

DIMS = [
    "energy",
    "intensity",
    "pace",
    "humor",
    "darkness",
    "hopefulness",
    "emotional_intensity",
    "romance",
    "suspense",
    "complexity",
    "realism",
    "inspiration",
]
GENRES = ["action", "comedy", "drama", "horror", "romance", "thriller", "animation", "documentary"]
EMBEDDING_DIM = 16
PROCESSED = "data/processed"
FILES = (
    "movies_clean.parquet",
    "movie_dna_labels.parquet",
    "plot_embeddings.npy",
    "plot_embeddings_index.parquet",
)


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def make_tables(n_movies: int = 240, seed: int = 0) -> dict[str, Any]:
    """Catalog, labels, embeddings and index with a few deliberate join drops."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n_movies + 1) * 3
    languages = rng.choice(
        ["english", "other_international", "hindi", "indian_regional", "unknown"],
        size=n_movies,
        p=[0.66, 0.2, 0.08, 0.04, 0.02],
    )
    genres = [sorted(set(rng.choice(GENRES, size=rng.integers(1, 4), replace=False).tolist())) for _ in ids]
    years = rng.integers(1950, 2020, size=n_movies).astype("float")
    years[rng.choice(n_movies, size=4, replace=False)] = np.nan
    runtime = rng.normal(105, 20, size=n_movies).clip(60, 200)
    runtime[:2] = 0.0
    popularity = rng.lognormal(1.5, 0.8, size=n_movies)
    embeddings = rng.normal(size=(n_movies, EMBEDDING_DIM)).astype(np.float32)

    weights = rng.normal(size=(EMBEDDING_DIM, len(DIMS)))
    genre_effect = rng.normal(scale=0.8, size=(len(GENRES), len(DIMS)))
    multi_hot = np.array([[g in gs for g in GENRES] for gs in genres], dtype=float)
    logits = (
        embeddings @ weights * 0.6
        + multi_hot @ genre_effect
        + rng.normal(scale=0.3, size=(n_movies, len(DIMS)))
    )
    labels = 1 / (1 + np.exp(-logits))

    year_int = pd.array(years, dtype="Int64")
    catalog = pd.DataFrame(
        {
            "movieId": ids,
            "title": [f"Movie {i}" for i in ids],
            "year": year_int,
            "year_raw": years,
            "overview": ["A story." for _ in ids],
            "runtime": runtime,
            "language_group": languages,
            "era": np.where(years < 1980, "pre_1980", np.where(years < 2000, "1980_1999", "2000_plus")),
            "genres": [np.array(g, dtype=object) for g in genres],
            "primary_genre": [g[0] for g in genres],
            "popularity": popularity,
            "vote_count": rng.integers(0, 5000, size=n_movies).astype(float),
            "vote_average": rng.uniform(4, 9, size=n_movies),
            "popularity_bucket": pd.qcut(popularity, 3, labels=["low", "mid", "high"]).astype(str),
            "flag_short_overview": rng.random(n_movies) < 0.05,
            "flag_missing_overview": np.zeros(n_movies, dtype=bool),
            "flag_missing_runtime": runtime <= 0,
            "flag_runtime_outlier": np.zeros(n_movies, dtype=bool),
            "flag_year_mismatch": rng.random(n_movies) < 0.03,
            "flag_unknown_language": languages == "unknown",
        }
    )
    label_frame = pd.DataFrame(labels, columns=DIMS)
    label_frame.insert(0, "movieId", ids)
    for k, dim in enumerate(DIMS):
        label_frame[f"{dim}_raw"] = labels[:, k] * 0.5
    label_frame["mapping_version"] = "v1"

    # join drops: 5 labelled movies have no catalog row, 3 more have no embedding
    no_catalog, no_embedding = ids[:5], ids[5:8]
    catalog = catalog[~catalog["movieId"].isin(no_catalog)].reset_index(drop=True)
    extra = catalog.iloc[:2].copy()
    extra["movieId"] = [10_001, 10_002]  # catalog rows without labels
    catalog = pd.concat([catalog, extra], ignore_index=True)
    keep = ~np.isin(ids, np.concatenate([no_catalog, no_embedding]))
    emb_ids = np.concatenate([ids[keep], [10_001, 10_002]])
    emb = np.vstack([embeddings[keep], rng.normal(size=(2, EMBEDDING_DIM)).astype(np.float32)])
    index = pd.DataFrame(
        {"row": np.arange(len(emb_ids)), "movieId": emb_ids, "text_sha1": "x", "encoder": "fixture-encoder"}
    )
    return {"catalog": catalog, "labels": label_frame, "embeddings": emb, "index": index}


def write_processed(processed: Path, tables: dict[str, Any]) -> None:
    """Write the four pipeline output files."""
    processed.mkdir(parents=True, exist_ok=True)
    tables["catalog"].to_parquet(processed / FILES[0], index=False)
    tables["labels"].to_parquet(processed / FILES[1], index=False)
    np.save(processed / FILES[2], tables["embeddings"])
    tables["index"].to_parquet(processed / FILES[3], index=False)


def write_lock(data_pipeline: Path) -> Path:
    """``dvc.lock`` listing the processed files with their MD5s (plus an unrelated directory output)."""
    outs = [
        {
            "path": f"{PROCESSED}/{name}",
            "hash": "md5",
            "md5": _md5(data_pipeline / PROCESSED / name),
            "size": (data_pipeline / PROCESSED / name).stat().st_size,
        }
        for name in FILES
    ]
    lock = {
        "schema": "2.0",
        "stages": {
            "clean": {"cmd": "python -m scripts.run_pipeline clean", "outs": outs[:1]},
            "build_dna_labels": {"cmd": "python -m scripts.run_pipeline build_dna_labels", "outs": outs[1:2]},
            "encode_plots": {
                "cmd": "python -m scripts.run_pipeline encode_plots",
                "outs": [
                    *outs[2:],
                    {"path": "data/cache/embeddings", "hash": "md5", "md5": "0" * 32 + ".dir", "nfiles": 1},
                ],
            },
        },
    }
    path = data_pipeline / "dvc.lock"
    path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def model_config(root: Path) -> dict[str, Any]:
    """Tiny model config pointing at the fixture tree."""
    return {
        "random_seed": 42,
        "experiment_name": "vibespace-movie-dna-test",
        "registered_model_name": "vibespace-movie-dna",
        "paths": {
            "data_pipeline_dir": str(root / "Data-Pipeline"),
            "processed_dir": str(root / "Data-Pipeline" / PROCESSED),
            "reports_dir": str(root / "Model-Pipeline" / "reports"),
            "artifacts_dir": str(root / "Model-Pipeline" / "artifacts"),
            "mlflow_dir": str(root / "Model-Pipeline" / "mlflow"),
        },
        "data": {
            "catalog_file": FILES[0],
            "labels_file": FILES[1],
            "embeddings_file": FILES[2],
            "embedding_index_file": FILES[3],
            "embedding_dim": EMBEDDING_DIM,
            "hash_checked_files": list(FILES),
        },
        "targets": DIMS,
        "features": {
            "genres_column": "genres",
            "runtime_column": "runtime",
            "year_column": "year",
            "language_column": "language_group",
            "popularity_columns": ["popularity", "vote_count"],
            "extra_numeric_columns": ["vote_average"],
            "flag_columns": [
                "flag_short_overview",
                "flag_missing_overview",
                "flag_missing_runtime",
                "flag_runtime_outlier",
                "flag_year_mismatch",
                "flag_unknown_language",
                "flag_not_in_catalog",
            ],
        },
        "split": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15,
            "stratify_on": "language_group",
            "min_stratum_size": 20,
        },
        "slices": ["language_group", "era", "primary_genre", "popularity_bucket"],
        "models": {
            "baseline": {},
            "ridge": {"alpha": 1.0},
            "lightgbm": {
                "n_estimators": 40,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "min_child_samples": 5,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 10,
                "n_jobs": 2,
            },
            "mlp": {
                "hidden_sizes": [32],
                "dropout": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
                "batch_size": 32,
                "max_epochs": 25,
                "patience": 5,
                "huber_delta": 0.1,
                "num_threads": 1,
            },
        },
        "explain": {
            "sample_size": 25,
            "background_size": 20,
            "gradient_nsamples": 10,
            "permutation_repeats": 2,
        },
        "gate": {"min_mae_improvement": 0.15, "min_r2": 0.30},
        "registry": {
            "project": "fixture-project",
            "location": "us-east1",
            "repository": "fixture-models",
            "package": "movie-dna",
            "champion_package": "movie-dna-champions",
            "bucket": "fixture-registry-bucket",
            "images_repository": "fixture-images",
            "promotion_tolerance": 0.002,
            "dry_run_dir": str(root / "Model-Pipeline" / "artifacts" / "registry-dry-run"),
        },
        "ci": {
            "family": "mlp",
            "params": {
                "hidden_sizes": [32],
                "dropout": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 0.0001,
                "batch_size": 32,
                "max_epochs": 25,
                "patience": 5,
                "huber_delta": 0.1,
                "num_threads": 1,
            },
            "local_alias": "ci-candidate",
        },
    }


def tuning_config() -> dict[str, Any]:
    """Tiny Optuna budget (a few trials per model)."""
    return {
        "sampler": "tpe",
        "tpe_n_startup_trials": 2,
        "pruner": {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 2},
        "total_time_cap_minutes": 10,
        "model_time_cap_minutes": {"ridge": 2, "lightgbm": 4, "mlp": 4},
        "model_order": ["ridge", "lightgbm", "mlp"],
        "storage_file": "optuna/optuna.db",
        "models": {
            "ridge": {
                "n_trials": 3,
                "search_space": {"alpha": {"type": "float", "low": 0.01, "high": 100, "log": True}},
            },
            "lightgbm": {
                "n_trials": 3,
                "fixed": {"n_estimators": 40, "early_stopping_rounds": 10, "n_jobs": 2},
                "search_space": {
                    "num_leaves": {"type": "int", "low": 4, "high": 16},
                    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
                },
            },
            "mlp": {
                "n_trials": 4,
                "fixed": {"max_epochs": 12, "patience": 4, "num_threads": 1},
                "search_space": {
                    "hidden_sizes": {"type": "categorical", "choices": ["16", "32"]},
                    "learning_rate": {"type": "float", "low": 0.001, "high": 0.03, "log": True},
                    "dropout": {"type": "float", "low": 0.0, "high": 0.3},
                },
            },
        },
        "sensitivity": {"top_k": 2, "points": 3, "importance_evaluator": "fanova"},
    }


def bias_config() -> dict[str, Any]:
    """Bias settings scaled to the small fixture (gated slices need 10 movies)."""
    return {
        "slice_columns": [
            "language_group",
            "era",
            "primary_genre",
            "popularity_bucket",
            "flag_short_overview",
        ],
        "min_gated_slice_size": 10,
        "max_mae_ratio": 1.25,
        "bootstrap": {"n_resamples": 200, "confidence": 0.95},
        "notable_gap_ratio": 1.15,
        "mitigation": {
            "column": "auto",
            "reweighting": {"power": 1.0, "max_weight": 20.0},
            "resampling": {"min_share_of_largest": 0.1},
        },
    }


def build_fixture_tree(root: Path, n_movies: int = 240, seed: int = 0) -> dict[str, Path]:
    """Write the whole fixture tree under ``root`` and return the important paths."""
    data_pipeline = root / "Data-Pipeline"
    write_processed(data_pipeline / PROCESSED, make_tables(n_movies, seed))
    lock = write_lock(data_pipeline)
    configs = root / "Model-Pipeline" / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("model_config.yaml", model_config(root)),
        ("tuning.yaml", tuning_config()),
        ("bias.yaml", bias_config()),
    ):
        (configs / name).write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8", newline="\n")
    return {
        "root": root,
        "data_pipeline": data_pipeline,
        "processed": data_pipeline / PROCESSED,
        "lock": lock,
        "config": configs / "model_config.yaml",
        "configs": configs,
    }


if __name__ == "__main__":  # pragma: no cover
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "fixture-tree")
    print(build_fixture_tree(target))
