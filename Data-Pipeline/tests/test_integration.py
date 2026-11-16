"""End-to-end run of the whole pipeline on fixture data, outside Airflow.

The MovieLens download and TMDB API are served by fake sessions and the text
encoder is a deterministic fake, so the test is fully offline.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts import acquire_movielens, encode_plots, run_pipeline
from scripts.build_dna_labels import DNA_DIMENSIONS
from tests.helpers import FakeEncoder, FakeResponse, FakeSession, tmdb_fixture_handler
from tests.test_acquire_movielens import movielens_zip

pytestmark = pytest.mark.integration


def parse_results(stdout: str) -> dict:
    """The CLI prints its JSON summary last; skip any log lines printed before it."""
    start = 0 if stdout.startswith("{") else stdout.index("\n{\n") + 1
    return json.loads(stdout[start:])


@pytest.fixture
def offline(monkeypatch):
    """Route all network access to fixtures."""
    zip_bytes = movielens_zip()
    md5 = hashlib.md5(zip_bytes).hexdigest()  # noqa: S324

    routes = {"tmdb": tmdb_fixture_handler}

    def handler(url, **kwargs):
        if "grouplens" in url:
            if url.endswith(".md5"):
                return FakeResponse(200, text=f"{md5}  ml-25m.zip")
            return FakeResponse(200, content=zip_bytes)
        return routes["tmdb"](url, **kwargs)

    # Import GX before ``requests.Session`` is patched: with ``dvc[gs]`` installed, GX imports google-auth,
    # which subclasses ``requests.Session`` at import time.
    import great_expectations  # noqa: F401

    # acquire_movielens and acquire_tmdb share the ``requests`` module, so one patch serves both.
    monkeypatch.setattr(acquire_movielens.requests, "Session", lambda: FakeSession(handler))
    monkeypatch.setattr(encode_plots, "load_encoder", lambda name: FakeEncoder())
    monkeypatch.setenv("MIN_MATCH_RATE", "0.9")
    return routes


def test_full_pipeline_on_fixtures(offline, tmp_path, capsys):
    code = run_pipeline.main(["all", "--skip", "dvc_version", "--run-id", "it1"])
    out = capsys.readouterr()
    assert code == 0, out.err
    results = parse_results(out.out)
    data = tmp_path / "data"

    assert results["fetch_tmdb"]["selected_movies"] == 58 and results["fetch_tmdb"]["not_found"] == 1
    join = results["join_ids"]
    assert (join["no_tmdb_id"], join["duplicate_tmdb_id"], join["tmdb_missing"]) == (1, 1, 1)
    assert join["match_rate"] > 0.9 and join["year_mismatches"] == 1

    catalog = pd.read_parquet(data / "processed" / "movies_clean.parquet")
    assert 12 not in set(catalog["movieId"])  # adult title quarantined
    assert set(catalog["language_group"]) >= {"english", "hindi", "indian_regional", "other_international"}

    labels = pd.read_parquet(data / "processed" / "movie_dna_labels.parquet")
    values = labels[list(DNA_DIMENSIONS)].to_numpy()
    assert values.min() >= 0 and values.max() <= 1

    emb = np.load(data / "processed" / "plot_embeddings.npy")
    index = pd.read_parquet(data / "processed" / "plot_embeddings_index.parquet")
    assert emb.shape == (len(index), 384) and 11 not in set(index["movieId"])  # no overview -> skipped

    assert results["validate_raw"]["success"] and results["validate_processed"]["success"]
    assert results["generate_stats"]["previous_run_id"] is None
    assert "baseline" in results["bias_slicing"]["overall_mae"]
    assert (tmp_path / "reports" / "bias_report.md").is_file()
    assert (tmp_path / "reports" / "dna_sanity_report.md").is_file()
    assert (tmp_path / "logs" / "pipeline.log").stat().st_size > 0

    # A second run resumes from caches and compares stats with the first run.
    code = run_pipeline.main(["all", "--skip", "dvc_version", "--run-id", "it2"])
    second = parse_results(capsys.readouterr().out)
    assert code == 0
    assert second["download_movielens"]["skipped"] is True
    assert second["fetch_tmdb"]["attempted"] == 0
    assert second["encode_plots"]["newly_encoded"] == 0
    assert second["generate_stats"]["previous_run_id"] == "it1"


def test_pipeline_stops_on_bad_tmdb_key(offline, capsys, tmp_path):
    offline["tmdb"] = lambda url, **_: FakeResponse(401)
    code = run_pipeline.main(["download_movielens", "fetch_tmdb"])
    assert code == 1
    assert "TMDBAuthError" in capsys.readouterr().err
    assert "401" in (tmp_path / "logs" / "pipeline.log").read_text()


def _tree_hashes(root):
    """``{relative path: md5}`` of every file under ``root``."""
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()  # noqa: S324
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_sample_run_never_touches_full_outputs(offline, tmp_path, capsys, monkeypatch):
    """A sample run (including dvc_version) writes only to data/sample/ and is never versioned."""
    assert run_pipeline.main(["all", "--skip", "dvc_version", "--run-id", "full"]) == 0
    capsys.readouterr()
    data = tmp_path / "data"
    before = {d: _tree_hashes(data / d) for d in ("interim", "processed", "metrics")}
    full_reports = _tree_hashes(tmp_path / "reports")

    calls: list[list[str]] = []
    from scripts import dvc_versioning

    monkeypatch.setattr(dvc_versioning, "run_dvc", lambda args, *a, **k: calls.append(args) or "")
    code = run_pipeline.main(["all", "--sample-size", "50", "--run-id", "sample"])
    out = capsys.readouterr()
    assert code == 0, out.err
    results = parse_results(out.out)

    # full outputs and committed-style reports are byte-identical
    assert {d: _tree_hashes(data / d) for d in ("interim", "processed", "metrics")} == before
    assert {k: v for k, v in _tree_hashes(tmp_path / "reports").items() if not k.startswith("sample/")} \
        == full_reports
    # the sample has its own tree with the normal layout
    sample_catalog = pd.read_parquet(data / "sample" / "processed" / "movies_clean.parquet")
    assert 0 < len(sample_catalog) <= 50
    assert (data / "sample" / "processed" / "plot_embeddings.npy").is_file()
    assert (data / "sample" / "metrics" / "validation_processed.json").is_file()
    assert (tmp_path / "reports" / "sample" / "bias_report.md").is_file()
    # dvc_version ran but did nothing
    assert results["dvc_version"]["skipped"] is True and not results["dvc_version"]["committed"]
    assert calls == []
