# VibeSpace data pipeline

VibeSpace is a mood-aware movie recommender. This folder holds its **data pipeline**: it downloads
MovieLens 25M and TMDB metadata and turns them into the inputs of the Movie DNA model:

| Output | File (under `data/processed/`) |
|---|---|
| Clean movie catalog (MovieLens + TMDB) with slice columns and data-quality flags | `movies_clean.parquet` |
| 12 **Movie DNA** labels per movie, built from the MovieLens Tag Genome | `movie_dna_labels.parquet` |
| 384-dim plot embeddings (`all-MiniLM-L6-v2`) + row -> movieId index | `plot_embeddings.npy`, `plot_embeddings_index.parquet` |

The 12 DNA dimensions are `energy, intensity, pace, humor, darkness, hopefulness, emotional_intensity,
romance, suspense, complexity, realism, inspiration`. **The Tag Genome is used only to build labels and is
never a model input.** No LLMs are used anywhere; the only pretrained model is the small local text encoder.

Everything is orchestrated by **Airflow** (LocalExecutor + Postgres in Docker), validated with
**Great Expectations**, monitored with statistics, drift and anomaly alerts, checked for bias with
**Fairlearn**, and versioned with **DVC**.

---

## 1. Folder structure

```
Data-Pipeline/
├── dags/data_pipeline_dag.py      # Airflow DAGs (optimised + sequential baseline)
├── scripts/                       # one module per step (pure functions) + task layer
│   ├── config.py                  # settings from env/.env (secrets never logged)
│   ├── acquire_movielens.py       # download + MD5 + safe unzip + header checks
│   ├── acquire_tmdb.py            # TMDB client: thread pool, rate limiter, retries, cache
│   ├── build_tmdb_table.py        # flatten cached JSON, quarantine bad files
│   ├── join_ids.py                # MovieLens <-> TMDB join + quarantine reasons
│   ├── clean.py                   # genres, languages, slices, flags, quarantine
│   ├── build_dna_labels.py        # Tag Genome -> 12 DNA percentiles
│   ├── dna_sanity_report.py       # top/bottom 10 per dimension + automatic flags
│   ├── encode_plots.py            # MiniLM plot embeddings, cached
│   ├── gx_suites.py, validate.py  # Great Expectations suites + runner
│   ├── generate_stats.py          # schema/statistics per run + drift (PSI)
│   ├── anomalies.py, alerts.py    # anomaly detection + Slack/email/log alerts
│   ├── bias_slicing.py            # Fairlearn slicing, mitigation, bias report
│   ├── dvc_versioning.py          # dvc commit / push / status
│   ├── tasks.py                   # thin Airflow task layer
│   ├── run_pipeline.py            # run any step without Airflow
│   ├── gantt.py                   # Gantt charts from the Airflow REST API
│   ├── io_utils.py, logging_utils.py, exceptions.py
├── configs/movie_dna_mapping_v1.yaml   # tag -> DNA mapping (v1, pending team review)
├── gx/expectations/*.json         # exported Great Expectations suites
├── tests/                         # pytest unit, integration and DAG tests + fixtures
├── data/                          # all data (DVC-tracked, git-ignored)
├── logs/                          # pipeline.log, alerts.log, Airflow task logs (git-ignored)
├── reports/                       # dna_sanity_report.md, bias_report.md
├── docs/                          # optimization.md, Gantt charts, requirements_checklist.md
├── dvc.yaml / dvc.lock            # DVC pipeline stages and output hashes
├── Dockerfile, docker-compose.yaml
├── requirements.txt, requirements-dev.txt, pytest.ini, ruff.toml, .env.example
```

## 2. Pipeline

```
download_movielens ─┬─> fetch_tmdb -> build_tmdb_table -> validate_raw -> join_ids -> clean -> encode_plots ─┐
                    └─> build_dna_labels ───────────────────────────────────────────────────────────────────┴─> validate_processed
validate_processed ─┬─> generate_stats -> detect_anomalies ─┐
                    └─> bias_slicing ────────────────────────┴─> dvc_version
```

| Step | What it does | Main outputs |
|---|---|---|
| `download_movielens` | Streams `ml-25m.zip` with retries, verifies the published MD5, rejects corrupt zips and path-traversal / symlink entries, checks every CSV header, writes `manifest.json`. Skips if already present and verified. | `data/raw/movielens/` |
| `fetch_tmdb` | Picks movies for `TMDB_STAGE` (`genome` ≈13.8k with Tag Genome, `rest` ≈48k others, `all`), fetches `/movie/{id}?append_to_response=keywords,videos` with a thread pool and a shared 20 req/s limiter. Retries timeouts / 5xx with exponential backoff, honours `Retry-After` on 429, caches 404s (never re-requested), stops at once on 401, fails above 5% failures. One JSON per movie, so reruns resume. | `data/raw/tmdb/movies/*.json`, `data/interim/tmdb_selection.parquet` |
| `build_tmdb_table` | Flattens title, overview, runtime, language, genres, keywords, countries, release date, popularity, votes, poster, adult flag and the best YouTube trailer. Unreadable files are moved to quarantine. | `data/interim/tmdb_movies.parquet` |
| `validate_raw` | GX suites on MovieLens movies/links/genome tags and the TMDB table. | `data/metrics/validation_raw.json` |
| `join_ids` | Joins via `links.csv`; quarantines `no_tmdb_id`, `duplicate_tmdb_id` (lowest movieId kept), `tmdb_missing`; flags year mismatches > 2 years; fails below a 90% match rate. | `data/interim/joined.parquet`, `join_quarantine.parquet`, `join_stats.json` |
| `clean` | Unifies MovieLens/TMDB genres, normalises language codes, adds `language_group`, `era`, `popularity_bucket`, `primary_genre`; flags (never silently fixes) missing/short overviews, missing/outlier runtimes, invalid years, unknown languages; quarantines duplicates, missing titles and adult titles. | `data/processed/movies_clean.parquet` |
| `build_dna_labels` | Validates every mapping tag against `genome-tags.csv` (unknown tag = hard failure), streams only the needed rows of the 15M-row `genome-scores.csv`, computes `mean(positive + (1 - inverse))` per dimension, converts to percentiles, records `mapping_version`, writes the sanity report. | `data/processed/movie_dna_labels.parquet`, `reports/dna_sanity_report.md` |
| `encode_plots` | Encodes overviews with `all-MiniLM-L6-v2` on CPU, batched; cache keyed by movieId + overview hash + encoder version; missing overviews are skipped and logged. | `data/processed/plot_embeddings.npy` + index |
| `validate_processed` | GX suites: unique non-null ids, runtime / year ranges, language format, slice values, DNA scores in [0,1], embedding dim = 384, finite unit-norm vectors, every embedding linked to the catalog, match rate ≥ threshold. Failures stop downstream tasks. | `data/metrics/validation_processed.json` |
| `generate_stats` | Schema + per-column statistics + slice counts per run; drift versus the previous run of the same profile (stage + sample size): schema changes, null-rate jumps, row-count change, PSI of slice distributions. | `data/metrics/stats/<profile>/stats_<run>.json` |
| `detect_anomalies` | Missing values, outliers, invalid formats, schema violations, failed GX suites, drift and join quality, each with a severity. Warnings and above trigger an alert; blocking critical anomalies fail the task. | `data/metrics/anomalies_current.json` |
| `bias_slicing` | Fairlearn slice analysis and mitigation (section 8). | `reports/bias_report.md`, `data/metrics/bias/` |
| `dvc_version` | `dvc commit --force`, `dvc push`, `dvc status`. | `dvc.lock`, `../dvc-storage/` |

## 3. Setup

Requirements: macOS/Linux, Python 3.11, Docker Desktop (≥ 6 GB memory for Docker), about 6 GB of free disk
for data + DVC cache + remote, and a free TMDB API key.

**Get a TMDB key:** create a free account at <https://www.themoviedb.org>, then go to
*Settings → API → Request an API key* (developer). Either the **v3 API key** or the **v4 read access token**
works.

```bash
cd VibeSpace/Data-Pipeline
cp .env.example .env            # then edit .env and paste your TMDB_API_KEY
echo "AIRFLOW_UID=$(id -u)" >> .env
```

### 3a. Local environment (tests and running without Airflow)

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt
.venv/bin/python -m pytest              # all tests, fully offline
```

The Airflow constraints file must be used so local versions match the Airflow image (see section 12).

### 3b. Airflow in Docker

```bash
docker compose build                    # custom image: Airflow 2.10.5 + requirements.txt, CPU-only torch
docker compose up airflow-init          # migrates the DB and creates the web user
docker compose up -d                    # postgres, scheduler, webserver
open http://localhost:8080              # login airflow / airflow (change in .env)
```

The whole repository is mounted at `/opt/vibespace`, `PYTHONPATH` points to `Data-Pipeline/`, so the DAG
imports `scripts` directly and DVC sees the repository root. Images are multi-arch (works on Apple Silicon).

## 4. Running the pipeline

**In Airflow (recommended):** open the `vibespace_data_pipeline` DAG → *Trigger DAG w/ config*. Params:

| Param | Default | Meaning |
|---|---|---|
| `tmdb_stage` | `genome` | `genome` (≈13.8k movies with Tag Genome; needed for labels), `rest` (≈48k others, catalog only, slow), `all` |
| `sample_size` | empty | run on N movies for a fast end-to-end check |
| `use_cache` | `true` | reuse cached TMDB responses and embeddings |
| `tmdb_max_workers`, `tmdb_rate_limit`, `encoder_batch_size` | 8, 20, 128 | performance knobs |

From the command line:

```bash
# quick end-to-end check on 200 movies (pause the DAG first so the scheduler does not pick up tasks)
docker compose exec airflow-scheduler airflow dags pause vibespace_data_pipeline
docker compose exec airflow-scheduler airflow dags test vibespace_data_pipeline -c '{"sample_size": 200}'
docker compose exec airflow-scheduler airflow dags unpause vibespace_data_pipeline

# full genome stage
docker compose exec airflow-scheduler airflow dags trigger vibespace_data_pipeline -c '{"tmdb_stage": "genome"}'
```

`vibespace_data_pipeline_baseline` is the same pipeline as a strict chain with one TMDB worker, small
batches and no cache. It exists only for the Gantt before/after comparison. Set `VIBESPACE_SCHEDULE`
(cron) in `.env` to schedule the optimised DAG; by default it is manual.

**Without Airflow** (same code, local `.venv`):

```bash
.venv/bin/python -m scripts.run_pipeline all --sample-size 200 --skip dvc_version
.venv/bin/python -m scripts.run_pipeline clean            # one step
.venv/bin/python -m scripts.run_pipeline all --stage rest  # TMDB stage "rest" (≈48k requests, ~45 min)
```

## 5. Data versioning with DVC

DVC is initialised at the repository root with a local remote `VibeSpace/dvc-storage/` (git-ignored).
`dvc.yaml` declares one stage per pipeline step with its dependencies and outputs; caches and history
folders are `persist: true`. Airflow's last task runs `dvc commit --force` (records hashes in `dvc.lock`),
`dvc push` and `dvc status`.

```bash
dvc status                 # outputs match dvc.lock?
dvc push / dvc pull        # sync data with dvc-storage/
dvc repro                  # reproduce all stages without Airflow
dvc dag                    # show the stage graph
git add dvc.yaml dvc.lock  # Git tracks code + pointers, DVC tracks data
```

To restore the data for a given commit: `git checkout <commit> && dvc pull`.

## 6. Validation, statistics and anomaly alerts

* **Great Expectations (GX Core 1.x)**: suites are declared in `scripts/gx_suites.py`, exported to
  `gx/expectations/*.json` and run on in-memory DataFrames. Results land in `data/metrics/validation_*.json`.
  A failed suite raises `ValidationFailedError`, which fails the task without retries and stops downstream.
* **Statistics**: every run writes schema + statistics; `latest.json` per profile is compared with the next
  run to record drift (PSI ≥ 0.10 warning, ≥ 0.25 critical; null-rate jump > 5 points; row drop > 20%).
* **Alerts**: every task has `on_failure_callback` (task, run, try, error, log URL) and `detect_anomalies`
  alerts on warnings. Channels: Slack (`SLACK_WEBHOOK_URL`) and/or email (`SMTP_HOST`, `ALERT_EMAIL`,
  `SMTP_*`). Every alert is also appended to `logs/alerts.log`, which is the only channel when nothing is
  configured. Alerting never crashes the pipeline.
* **Logging**: `logs/pipeline.log` (rotating) plus Airflow task logs in `logs/airflow/`; each step logs
  START / DONE with duration and full tracebacks on failure.

## 7. Error handling

| Failure | Behaviour |
|---|---|
| MovieLens unavailable / truncated download | retries with backoff, `.part` file, clear `DataAcquisitionError` |
| Checksum mismatch, corrupt or unsafe zip | `ChecksumMismatchError` / `CorruptArchiveError`, bad zip deleted |
| TMDB 401 | `TMDBAuthError`, run stops immediately, no Airflow retries |
| TMDB 429 / 5xx / timeouts | backoff and `Retry-After`; step fails only above 5% failures |
| Corrupt cached JSON | moved to `data/interim/tmdb_quarantine_files/`, re-fetched next run |
| Unknown tag in DNA mapping | `MappingError` listing the tags, no retries |
| GX suite failure / blocking anomaly | task fails, downstream tasks do not run, alert sent |
| Missing upstream file | `FileNotFoundError` naming the upstream step |

## 8. Bias detection and mitigation (summary)

Full report: [`reports/bias_report.md`](reports/bias_report.md).

* **Why movie slices:** MovieLens 25M and the app hold no user demographics, so bias is measured on content
  slices that decide which films get recommended: `language_group`, `era`, `primary_genre`,
  `popularity_bucket`.
* **Found:** the main bias is **representation**. English films are 82.9% of the 13,666-film genome catalog,
  Hindi 0.56% (76 films), Indian regional languages 0.07% (9 films), because the Tag Genome covers mostly
  well-known, often-rated films.
* **Model behaviour:** a diagnostic Ridge model (plot embedding + metadata → 12 DNA scores, 70/15/15 split)
  evaluated with Fairlearn `MetricFrame` has an overall test MAE of 0.196. No slice with ≥ 15 test movies is
  more than 10% worse than overall. The largest gap is between genres (family 0.179 vs western 0.209).
  Hindi / Indian-regional slices are too small to judge.
* **Mitigation:** inverse-frequency re-weighting and slice re-sampling were applied and re-measured.
  Re-weighting narrowed the language gap (0.0084 → 0.0069) and the era gap but widened the genre gap, at
  +0.0008 overall MAE. Re-sampling of small genres narrowed every worst-case gap slightly at no accuracy cost,
  so the report's decision rule adopts it. Both effects are small; trade-offs are in the report. The real fix is more data
  for small slices (TMDB stage `rest`), with the slice tables re-checked on every run.

## 9. Pipeline optimisation (summary)

Full write-up and charts: [`docs/optimization.md`](docs/optimization.md).

| Run (1,500 movies, cold cache) | Wall time | fetch_tmdb | encode_plots |
|---|---|---|---|
| Baseline: sequential chain, 1 worker @ 20 req/s, batch 8 | 117.8 s | 78.0 s | 14.6 s |
| Optimised: parallel branches, 8 workers @ 40 req/s, batch 128 | 67.7 s (−43%) | 38.2 s | 10.8 s |

The Gantt chart showed `fetch_tmdb` as the bottleneck (66% of the run), capped by the rate limiter.
With caching, a full genome-stage rerun takes 23.6 s instead of 813.7 s.

## 10. Tests

```bash
.venv/bin/python -m pytest                               # unit + integration + DAG tests (offline)
.venv/bin/python -m pytest --cov=scripts --cov=dags       # with coverage
.venv/bin/python -m pytest -m integration                # end-to-end run on fixture data
.venv/bin/ruff check .                                    # lint
```

Tests never touch the network: fake HTTP sessions, fake TMDB responses (`tests/fixtures/tmdb/`), a tiny
MovieLens layout (`tests/fixtures/movielens/`, regenerate with `python -m tests.fixtures.build_fixtures`)
and a deterministic fake text encoder.

Current results: **223 tests passed** (214 unit, 2 integration, 7 DAG), **97% line coverage** on `scripts/` and
`dags/`, `ruff check` clean.

## 11. Reproducibility

Clone → `cp .env.example .env` (add the TMDB key) → section 3a → `pytest` → section 3b → trigger the DAG.
All randomness uses `RANDOM_SEED=42`; library versions are fixed by `requirements.txt` plus the Airflow
constraints file; data versions are fixed by `dvc.lock`.

## 12. Dependency notes

* Airflow 2.10.5 pins many shared libraries (pandas 2.1.4, numpy 1.26.4, pyarrow 16.1, SQLAlchemy 1.4,
  sympy 1.13.3, ...). All packages are installed **with Airflow's constraints file**, so the image and the
  local venv resolve to the same versions.
* `sympy==1.13.3` (pinned by Airflow) is incompatible with torch ≤ 2.6 (which require `sympy==1.13.1`), so
  torch is pinned to **2.7.1**; the image installs the CPU-only build from the PyTorch CPU index.
* `great_expectations==1.3.14` works with pandas 2.1 and SQLAlchemy 1.4; `fairlearn==0.12.0` with
  scikit-learn 1.5.2; `sentence-transformers==3.4.1` with `transformers==4.48.3`.
* The encoder weights (~90 MB) download on first use into `.cache/huggingface/` (git-ignored, shared by
  the venv and the containers).

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `TMDB_API_KEY is not set` / 401 | add the key to `Data-Pipeline/.env`, restart containers (`docker compose up -d`) |
| `dags test` shows "unrunnable tasks" | the running scheduler picked them up; pause the DAG during `dags test` |
| Permission errors on `data/` or `logs/` | set `AIRFLOW_UID=$(id -u)` in `.env` and recreate containers |
| Webserver not reachable | `docker compose ps`; wait for `healthy`; check `docker compose logs airflow-webserver` |
| Containers killed while encoding | give Docker Desktop ≥ 6 GB memory |
| `dvc push` fails | make sure `../dvc-storage` is writable; `dvc remote list` should show `localstore` |
| Match rate below 90% | check `data/interim/join_quarantine.parquet` reasons and `fetch_tmdb_summary.json` failures |

## 14. Data attribution

* **MovieLens 25M** — F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and
  Context.* ACM TiiS 5(4). Provided by GroupLens Research, University of Minnesota, under their usage
  license (non-commercial research and education; do not redistribute).
* **TMDB** — This product uses the TMDB API but is not endorsed or certified by TMDB. Metadata, posters
  and trailer references come from <https://www.themoviedb.org>.
* **all-MiniLM-L6-v2** — sentence-transformers model, Apache 2.0.

Owner: VibeSpace team.
