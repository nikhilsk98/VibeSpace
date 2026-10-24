# Pipeline flow optimisation (Airflow Gantt)

## Method

Both configurations ran in the same Docker stack (Airflow 2.10.5, LocalExecutor, Postgres, Apple M4,
Docker limited to 10 CPUs / 7.75 GB) on the **same 1,500-movie sample** of TMDB stage `genome`, with
**caches disabled** (`use_cache=false`) so every TMDB request and every embedding was really done.

| | Baseline (`vibespace_data_pipeline_baseline`, run `gantt_before_1500`) | Optimised (`vibespace_data_pipeline`, run `gantt_after_1500`) |
|---|---|---|
| Task layout | strict chain of 13 tasks | parallel branches (see below) |
| TMDB workers / rate limit | 1 / 20 req/s | 8 / 40 req/s |
| Encoder batch size | 8 | 128 |
| Caches | off | off for this comparison (on by default) |

Durations were read from the Airflow REST API (`scripts/gantt.py`), which also drew the charts and took
the Airflow UI screenshots with Playwright.

* Airflow UI Gantt view: [`gantt_before.png`](gantt_before.png), [`gantt_after.png`](gantt_after.png)
* Same runs drawn from task start/end times: [`gantt_before_chart.png`](gantt_before_chart.png),
  [`gantt_after_chart.png`](gantt_after_chart.png)
* Raw numbers: [`gantt_before.json`](gantt_before.json), [`gantt_after.json`](gantt_after.json)

## Bottleneck

The baseline Gantt chart shows a single dominant bar: **`fetch_tmdb` took 78.0 s of the 117.8 s wall time
(66%)**. It was limited by the rate limiter (20 req/s × 1,500 ids ≈ 75 s), not by network latency: one
worker already reached ~19 req/s because TMDB answers in ~50 ms. The second-largest bar was
`encode_plots` (14.6 s) with tiny batches of 8 texts. Everything else, including the independent
`build_dna_labels` step, waited in line behind these two.

## Fixes

1. **TMDB concurrency + higher rate limit:** 8 worker threads sharing one thread-safe rate limiter set to
   40 req/s, below TMDB's ~50 req/s per-IP limit. Retries, `Retry-After` handling and the 5% failure
   gate are unchanged.
2. **Parallel branches:** `build_dna_labels` needs only MovieLens, so it now runs next to `fetch_tmdb`
   instead of after `clean`. `bias_slicing` runs next to `generate_stats -> detect_anomalies`, and
   `dvc_version` waits for both.
3. **Larger encoder batches** (128 instead of 8), so the CPU does fewer, fuller matrix multiplications.
4. **Caching (default on):** TMDB responses are cached per movie and embeddings per (movieId, overview
   hash, encoder), so later runs only fetch or encode what is new.

## Results (cold cache, 1,500 movies)

| Task | Before (s) | After (s) | Change |
|---|---|---|---|
| fetch_tmdb | 78.0 | 38.2 | −51% |
| encode_plots | 14.6 | 10.8 | −26% |
| dvc_version | 7.5 | 5.5 | −26% |
| bias_slicing | 2.6 | 2.6 | parallel with stats + anomalies |
| build_dna_labels | 1.6 | 1.7 | now overlaps fetch_tmdb |
| all other tasks | 4.2 | 4.3 | — |
| **Sum of task durations** | **108.5** | **63.1** | −42% |
| **Wall time (first start → last end)** | **117.8** | **67.7** | **−43% (1.74× faster)** |

## Caching on the full genome stage

Real triggered runs of the optimised DAG on all 13,816 Tag Genome movies (13,666 after the join):

| Run | fetch_tmdb | encode_plots | dvc_version | Wall time |
|---|---|---|---|---|
| `genome_full_1`: first run, empty caches (rate limit 20 req/s at that time) | 681.4 s | 72.9 s | 43.4 s | **813.7 s** |
| `genome_full_3_warm`: rerun, warm caches | 2.5 s | 0.2 s | 5.6 s | **23.6 s** (34× faster) |

The warm run still re-reads every cached file, re-validates, re-computes statistics and re-checks the
bias tables, so a scheduled rerun costs about 24 s unless TMDB data really changed. At the new 40 req/s
limit, a cold genome fetch would take about 13,600 / 40 ≈ 340 s instead of 681 s. Chart:
[`gantt_full_warm_chart.png`](gantt_full_warm_chart.png); numbers: [`gantt_full_cold.json`](gantt_full_cold.json),
[`gantt_full_warm.json`](gantt_full_warm.json).

While measuring this we found and fixed a cache bug: a run with `use_cache=false` used to overwrite the
embedding cache with only that run's movies. It now re-encodes but merges into the stored cache
(test: `test_no_cache_run_does_not_shrink_cache`).

## What did not help and remaining limits

* More than 8 workers would not help: at 40 req/s the limiter, not the thread count, sets the pace.
  Raising it further would risk HTTP 429s from TMDB.
* Running `build_dna_labels` in parallel saves only ~1.7 s here because Tag Genome loading is already fast
  (streamed pyarrow reads of only the 78 needed tags). It matters more when the genome file is not in the OS
  page cache.
* On this data size the fixed per-task overhead of Airflow (~1 s for scheduling plus interpreter start per
  task) is visible for the small tasks. Merging them would hide failure points, so we kept one task per step.
