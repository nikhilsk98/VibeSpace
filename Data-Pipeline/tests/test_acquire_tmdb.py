"""Tests for scripts.acquire_tmdb (fake sessions and clients only)."""

from __future__ import annotations

import json
import threading

import pandas as pd
import pytest
import requests

from scripts import acquire_tmdb as at
from scripts.exceptions import ConfigError, DataAcquisitionError, FailureRateExceededError, TMDBAuthError
from tests.helpers import MISSING_TMDB_ID, FakeResponse, FakeSession, NoSleep, tmdb_fixture_handler


def client(session, retries=3, sleep=None):
    return at.TMDBClient("k" * 32, session=session, rate_limiter=at.RateLimiter(1e6), max_retries=retries,
                         backoff_base=0.01, sleep=sleep or NoSleep())


def test_ok_response_and_v3_auth():
    session = FakeSession(tmdb_fixture_handler)
    result = client(session).get_movie(862)
    assert result.status == "ok" and result.data["id"] == 862
    call = session.calls[0]
    assert call["params"]["api_key"] == "k" * 32
    assert call["params"]["append_to_response"] == "keywords,videos"
    assert "Authorization" not in call["headers"]


def test_v4_token_uses_bearer_header():
    session = FakeSession(tmdb_fixture_handler)
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.signature"
    at.TMDBClient(token, session=session, rate_limiter=at.RateLimiter(1e6)).get_movie(862)
    call = session.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {token}"
    assert "api_key" not in call["params"]


def test_missing_key_is_config_error():
    with pytest.raises(ConfigError):
        at.TMDBClient("")


def test_404_is_not_found():
    assert client(FakeSession(responses=[FakeResponse(404)])).get_movie(1).status == "not_found"


def test_401_raises_immediately():
    session = FakeSession(responses=[FakeResponse(401)])
    with pytest.raises(TMDBAuthError):
        client(session).get_movie(1)
    assert len(session.calls) == 1


def test_429_honours_retry_after():
    sleep = NoSleep()
    session = FakeSession(responses=[FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"id": 1})])
    result = client(session, sleep=sleep).get_movie(1)
    assert result.status == "ok"
    assert 7.0 in sleep.calls


def test_429_without_header_uses_backoff():
    sleep = NoSleep()
    session = FakeSession(responses=[FakeResponse(429, headers={"Retry-After": "soon"}), FakeResponse(200, {"id": 1})])
    assert client(session, sleep=sleep).get_movie(1).status == "ok"
    assert sleep.calls == [0.01]


@pytest.mark.parametrize("first", [FakeResponse(500), FakeResponse(503), requests.Timeout("t"),
                                   requests.ConnectionError("c"), FakeResponse(200, bad_json=True)])
def test_transient_errors_are_retried(first):
    session = FakeSession(responses=[first, FakeResponse(200, {"id": 5})])
    result = client(session).get_movie(5)
    assert result.status == "ok" and result.attempts == 2


def test_retries_exhausted_is_failed():
    session = FakeSession(responses=[FakeResponse(502)] * 3)
    result = client(session, retries=3).get_movie(5)
    assert result.status == "failed" and "502" in result.error


def test_other_4xx_fails_without_retry():
    session = FakeSession(responses=[FakeResponse(400)])
    assert client(session).get_movie(5).status == "failed"
    assert len(session.calls) == 1


def test_unexpected_request_exception():
    session = FakeSession(responses=[requests.RequestException("weird")])
    assert client(session).get_movie(5).status == "failed"


def test_rate_limiter_spaces_calls():
    now = [0.0]
    sleeps = []
    limiter = at.RateLimiter(2.0, clock=lambda: now[0], sleep=sleeps.append)
    for _ in range(3):
        limiter.acquire()
    assert sleeps == [0.5, 1.0]
    with pytest.raises(ValueError):
        at.RateLimiter(0)


def test_rate_limiter_thread_safe():
    limiter = at.RateLimiter(1e9)
    threads = [threading.Thread(target=limiter.acquire) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert limiter._next_slot > 0


def test_fetch_writes_cache_and_resumes(tmp_path):
    session = FakeSession(tmdb_fixture_handler)
    ids = [862, 949, MISSING_TMDB_ID]
    summary = at.fetch_movies(client(session), ids, tmp_path, max_workers=2)
    assert (summary.fetched, summary.not_found, summary.failed) == (2, 1, 0)
    marker = json.loads((tmp_path / f"{MISSING_TMDB_ID}.json").read_text())
    assert marker[at.NOT_FOUND_MARKER] is True
    calls = len(session.calls)
    again = at.fetch_movies(client(session), ids, tmp_path)
    assert again.cached == 3 and again.cached_not_found == 1 and again.attempted == 0
    assert len(session.calls) == calls  # 404s are never requested again


def test_corrupt_cache_file_is_refetched(tmp_path):
    (tmp_path / "862.json").write_text("{broken")
    session = FakeSession(tmdb_fixture_handler)
    summary = at.fetch_movies(client(session), [862], tmp_path)
    assert summary.fetched == 1


def test_use_cache_false_refetches(tmp_path):
    session = FakeSession(tmdb_fixture_handler)
    at.fetch_movies(client(session), [862], tmp_path)
    summary = at.fetch_movies(client(session), [862], tmp_path, use_cache=False)
    assert summary.fetched == 1 and summary.cached == 0


def test_failure_rate_exceeded(tmp_path):
    session = FakeSession(lambda url, **_: FakeResponse(500))
    with pytest.raises(FailureRateExceededError):
        at.fetch_movies(client(session, retries=1), [1, 2, 3], tmp_path, max_failure_rate=0.05)
    assert not list(tmp_path.glob("*.json"))  # failures are not cached


def test_auth_error_stops_fetch(tmp_path):
    session = FakeSession(lambda url, **_: FakeResponse(401))
    with pytest.raises(TMDBAuthError):
        at.fetch_movies(client(session), list(range(1, 50)), tmp_path, max_workers=4)
    assert len(session.calls) < 49


def test_summary_dict():
    s = at.FetchSummary(stage="genome", requested=10, fetched=8, failed=2, failed_ids=[1, 2])
    d = s.to_dict()
    assert d["attempted"] == 10 and d["failure_rate"] == 0.2


def test_select_movies_stages():
    links = pd.DataFrame({"movieId": [1, 2, 3, 4], "imdbId": [1, 2, 3, 4], "tmdbId": [10, None, 30, 40]})
    genome = {1, 2}
    assert at.select_movies(links, genome, "genome")["movieId"].tolist() == [1, 2]
    assert at.select_movies(links, genome, "rest")["movieId"].tolist() == [3, 4]
    assert len(at.select_movies(links, genome, "all")) == 4
    assert pd.isna(at.select_movies(links, genome, "genome")["tmdbId"].iloc[1])
    with pytest.raises(ConfigError):
        at.select_movies(links, genome, "bogus")


def test_select_movies_sample_is_reproducible():
    links = pd.DataFrame({"movieId": range(100), "imdbId": range(100), "tmdbId": range(100)})
    a = at.select_movies(links, set(range(100)), "all", sample_size=10, seed=1)
    b = at.select_movies(links, set(range(100)), "all", sample_size=10, seed=1)
    assert a["movieId"].tolist() == b["movieId"].tolist() and len(a) == 10


def test_run_fetch_tmdb(ml_cfg):
    summary = at.run_fetch_tmdb(ml_cfg, client=client(FakeSession(tmdb_fixture_handler)))
    selection = pd.read_parquet(ml_cfg.interim_dir / at.SELECTION_FILE)
    assert summary["selected_movies"] == len(selection) == 58  # 60 fixture movies minus 2 without genome
    assert summary["movies_without_tmdb_id"] == 1
    assert summary["not_found"] == 1
    assert (ml_cfg.interim_dir / at.FETCH_SUMMARY_FILE).is_file()


def test_run_fetch_tmdb_rest_stage(ml_cfg):
    from dataclasses import replace

    rest = replace(ml_cfg, tmdb_stage="rest")
    summary = at.run_fetch_tmdb(rest, client=client(FakeSession(tmdb_fixture_handler)))
    assert summary["selected_movies"] == 2 and summary["fetched"] == 2


def test_run_fetch_tmdb_empty_selection(ml_cfg):
    from dataclasses import replace

    (ml_cfg.movielens_csv_dir / "links.csv").write_text("movieId,imdbId,tmdbId\n")
    with pytest.raises(DataAcquisitionError):
        at.run_fetch_tmdb(replace(ml_cfg, tmdb_stage="genome"), client=client(FakeSession(tmdb_fixture_handler)))
