"""Fetch TMDB movie details (with keywords and videos) into a per-movie JSON cache.

Design:
* a thread pool shares one thread-safe :class:`RateLimiter` (default 20 req/s);
* timeouts, connection errors and 5xx responses are retried with exponential
  backoff, and ``Retry-After`` is honoured on HTTP 429;
* HTTP 404 is cached as "not found" and never requested again;
* HTTP 401 stops the whole run immediately with a clear message;
* every movie gets one cached JSON file, so reruns resume where they stopped;
* the step fails when more than ``max_failure_rate`` of the requests failed.

Stages pick which movies to fetch: ``genome`` (movies with Tag Genome data),
``rest`` (all other MovieLens movies) or ``all``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.csv as pacsv
import requests

from scripts.config import TMDB_STAGES, PipelineConfig
from scripts.exceptions import ConfigError, DataAcquisitionError, FailureRateExceededError, TMDBAuthError
from scripts.io_utils import ensure_dir, read_csv_checked, write_bytes_atomic, write_json, write_parquet
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

NOT_FOUND_MARKER = "_vibespace_not_found"
APPEND_TO_RESPONSE = "keywords,videos"
SELECTION_FILE = "tmdb_selection.parquet"
FETCH_SUMMARY_FILE = "fetch_tmdb_summary.json"


class RateLimiter:
    """Thread-safe limiter that spaces calls at least ``1 / rate`` seconds apart."""

    def __init__(self, rate_per_second: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.interval = 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> float:
        """Block until the caller may send a request; return the time waited."""
        with self._lock:
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
        wait_for = slot - now
        if wait_for > 0:
            self._sleep(wait_for)
        return max(wait_for, 0.0)


@dataclass
class FetchResult:
    """Outcome of one TMDB lookup."""

    tmdb_id: int
    status: str  # "ok" | "not_found" | "failed"
    data: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0


class TMDBClient:
    """Minimal TMDB v3 client with retries, backoff and rate limiting."""

    def __init__(self, api_key: str, base_url: str = "https://api.themoviedb.org/3",
                 session: requests.Session | None = None, rate_limiter: RateLimiter | None = None,
                 timeout: float = 10.0, max_retries: int = 5, backoff_base: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise ConfigError(
                "TMDB_API_KEY is not set. Add it to Data-Pipeline/.env (see .env.example)."
            )
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.rate_limiter = rate_limiter or RateLimiter(20.0)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._params: dict[str, str] = {"append_to_response": APPEND_TO_RESPONSE}
        self._headers: dict[str, str] = {"Accept": "application/json"}
        if api_key.startswith("eyJ") and api_key.count(".") == 2:
            self._headers["Authorization"] = f"Bearer {api_key}"
        else:
            self._params["api_key"] = api_key

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff in seconds for a 0-based attempt number."""
        return min(60.0, self.backoff_base * (2**attempt))

    @staticmethod
    def _retry_after(resp: requests.Response, default: float) -> float:
        """Parse a ``Retry-After`` header in seconds, falling back to ``default``."""
        raw = resp.headers.get("Retry-After", "")
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return default

    def get_movie(self, tmdb_id: int) -> FetchResult:
        """Fetch one movie. Raises :class:`TMDBAuthError` on HTTP 401."""
        url = f"{self.base_url}/movie/{int(tmdb_id)}"
        last_error = "unknown error"
        attempts = 0
        for attempt in range(self.max_retries):
            attempts = attempt + 1
            self.rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=self._params, headers=self._headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._sleep(self._backoff(attempt))
                continue
            except requests.RequestException as exc:
                return FetchResult(tmdb_id, "failed", error=f"{type(exc).__name__}: {exc}", attempts=attempts)

            status = resp.status_code
            if status == 200:
                try:
                    return FetchResult(tmdb_id, "ok", data=resp.json(), attempts=attempts)
                except ValueError:
                    last_error = "invalid JSON in response"
                    self._sleep(self._backoff(attempt))
                    continue
            if status == 404:
                return FetchResult(tmdb_id, "not_found", attempts=attempts)
            if status == 401:
                raise TMDBAuthError(
                    "TMDB returned 401 Unauthorized. Check TMDB_API_KEY in Data-Pipeline/.env "
                    "(v3 API key or v4 read access token)."
                )
            if status == 429:
                delay = self._retry_after(resp, self._backoff(attempt))
                last_error = "HTTP 429 rate limited"
                logger.warning("TMDB rate limit hit for %s; waiting %.1fs", tmdb_id, delay)
                self._sleep(delay)
                continue
            if status >= 500:
                last_error = f"HTTP {status}"
                self._sleep(self._backoff(attempt))
                continue
            return FetchResult(tmdb_id, "failed", error=f"HTTP {status}", attempts=attempts)
        return FetchResult(tmdb_id, "failed", error=last_error, attempts=attempts)


# ----- cache ---------------------------------------------------------------
def cache_path(cache_dir: Path, tmdb_id: int) -> Path:
    """Location of the cached JSON for ``tmdb_id``."""
    return Path(cache_dir) / f"{int(tmdb_id)}.json"


def read_cache_status(path: Path) -> str | None:
    """Return ``"ok"`` / ``"not_found"`` for a valid cache file, else ``None``."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        logger.warning("Cached file %s is unreadable; it will be fetched again", path.name)
        return None
    if isinstance(payload, dict) and payload.get(NOT_FOUND_MARKER):
        return "not_found"
    return "ok" if isinstance(payload, dict) and "id" in payload else None


def write_cache(cache_dir: Path, result: FetchResult) -> None:
    """Persist a successful or not-found result (failures are not cached)."""
    if result.status == "ok":
        payload: dict[str, Any] = result.data or {}
    elif result.status == "not_found":
        payload = {NOT_FOUND_MARKER: True, "id": int(result.tmdb_id)}
    else:
        return
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    write_bytes_atomic(cache_path(cache_dir, result.tmdb_id), data)


# ----- selection -------------------------------------------------------------
def load_genome_movie_ids(genome_scores_csv: Path) -> set[int]:
    """Return the movie ids present in ``genome-scores.csv`` (reads one column)."""
    table = pacsv.read_csv(
        genome_scores_csv,
        convert_options=pacsv.ConvertOptions(include_columns=["movieId"]),
    )
    return set(pd.unique(table.column("movieId").to_numpy()).tolist())


def select_movies(links: pd.DataFrame, genome_ids: set[int], stage: str, sample_size: int | None = None,
                  seed: int = 42) -> pd.DataFrame:
    """Pick the MovieLens movies for a TMDB stage (optionally a reproducible sample).

    Returns columns ``movieId``, ``tmdbId`` (nullable Int64) and ``has_genome``.
    """
    if stage not in TMDB_STAGES:
        raise ConfigError(f"Unknown TMDB stage {stage!r}; expected one of {TMDB_STAGES}")
    df = links[["movieId", "tmdbId"]].copy()
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").astype("Int64")
    df["has_genome"] = df["movieId"].isin(genome_ids)
    if stage == "genome":
        df = df[df["has_genome"]]
    elif stage == "rest":
        df = df[~df["has_genome"]]
    df = df.sort_values("movieId").reset_index(drop=True)
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).sort_values("movieId").reset_index(drop=True)
    return df


@dataclass
class FetchSummary:
    """Counts for one fetch run."""

    stage: str
    requested: int = 0
    cached: int = 0
    fetched: int = 0
    not_found: int = 0
    cached_not_found: int = 0
    failed: int = 0
    failed_ids: list[int] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def attempted(self) -> int:
        """Requests actually sent to TMDB in this run."""
        return self.fetched + self.not_found + self.failed

    @property
    def failure_rate(self) -> float:
        """Share of attempted requests that failed."""
        return self.failed / self.attempted if self.attempted else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialisable summary (failed ids are truncated to keep it small)."""
        out = asdict(self)
        out["failed_ids"] = self.failed_ids[:200]
        out["attempted"] = self.attempted
        out["failure_rate"] = round(self.failure_rate, 6)
        return out


def fetch_movies(client: TMDBClient, tmdb_ids: Iterable[int], cache_dir: Path, stage: str = "genome",
                 max_workers: int = 8, use_cache: bool = True, max_failure_rate: float = 0.05,
                 progress_every: int = 1000) -> FetchSummary:
    """Fetch every id not yet cached, in parallel; return a :class:`FetchSummary`."""
    cache_dir = ensure_dir(cache_dir)
    ids = sorted({int(i) for i in tmdb_ids})
    summary = FetchSummary(stage=stage, requested=len(ids))
    start = time.perf_counter()

    todo: list[int] = []
    for tmdb_id in ids:
        status = read_cache_status(cache_path(cache_dir, tmdb_id)) if use_cache else None
        if status is None:
            todo.append(tmdb_id)
        else:
            summary.cached += 1
            if status == "not_found":
                summary.cached_not_found += 1
    logger.info("TMDB stage %s: %d requested, %d cached, %d to fetch (workers=%d)",
                stage, summary.requested, summary.cached, len(todo), max_workers)

    lock = threading.Lock()

    def work(tmdb_id: int) -> None:
        result = client.get_movie(tmdb_id)
        write_cache(cache_dir, result)
        with lock:
            if result.status == "ok":
                summary.fetched += 1
            elif result.status == "not_found":
                summary.not_found += 1
            else:
                summary.failed += 1
                summary.failed_ids.append(tmdb_id)
                logger.warning("TMDB id %s failed after %d attempts: %s", tmdb_id, result.attempts, result.error)
            done = summary.attempted
            if progress_every and done % progress_every == 0:
                logger.info("TMDB progress: %d/%d", done, len(todo))

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tmdb") as pool:
            futures = [pool.submit(work, tmdb_id) for tmdb_id in todo]
            done, pending = wait(futures, return_when=FIRST_EXCEPTION)
            errors = [f.exception() for f in done if f.exception() is not None]
            if errors:
                for f in pending:
                    f.cancel()
                raise errors[0]

    summary.duration_seconds = round(time.perf_counter() - start, 2)
    logger.info("TMDB stage %s finished: fetched=%d not_found=%d failed=%d cached=%d (%.1fs)",
                stage, summary.fetched, summary.not_found, summary.failed, summary.cached,
                summary.duration_seconds)
    if summary.failure_rate > max_failure_rate:
        raise FailureRateExceededError(
            f"TMDB failure rate {summary.failure_rate:.1%} exceeds {max_failure_rate:.1%} "
            f"({summary.failed} of {summary.attempted} requests failed)"
        )
    return summary


def run_fetch_tmdb(cfg: PipelineConfig, client: TMDBClient | None = None) -> dict[str, Any]:
    """Select the movies for ``cfg.tmdb_stage``, fetch them and record the selection."""
    csv_dir = cfg.movielens_csv_dir
    links = read_csv_checked(csv_dir / "links.csv", ["movieId", "imdbId", "tmdbId"])
    genome_ids = load_genome_movie_ids(csv_dir / "genome-scores.csv")
    selection = select_movies(links, genome_ids, cfg.tmdb_stage, cfg.sample_size, cfg.random_seed)
    if selection.empty:
        raise DataAcquisitionError(f"No MovieLens movies selected for TMDB stage {cfg.tmdb_stage!r}")
    selection["stage"] = cfg.tmdb_stage
    write_parquet(cfg.interim_dir / SELECTION_FILE, selection)

    if client is None:
        limiter = RateLimiter(cfg.tmdb_rate_limit)
        client = TMDBClient(cfg.tmdb_api_key, cfg.tmdb_base_url, rate_limiter=limiter, timeout=cfg.tmdb_timeout,
                            max_retries=cfg.tmdb_max_retries, backoff_base=cfg.tmdb_backoff_base)
    tmdb_ids = selection["tmdbId"].dropna().astype(int).tolist()
    summary = fetch_movies(client, tmdb_ids, cfg.tmdb_cache_dir, stage=cfg.tmdb_stage,
                           max_workers=cfg.tmdb_max_workers, use_cache=cfg.use_cache,
                           max_failure_rate=cfg.tmdb_max_failure_rate)
    result = summary.to_dict()
    result.update({"selected_movies": int(len(selection)),
                   "movies_without_tmdb_id": int(selection["tmdbId"].isna().sum()),
                   "sample_size": cfg.sample_size})
    write_json(cfg.interim_dir / FETCH_SUMMARY_FILE, result)
    return result
