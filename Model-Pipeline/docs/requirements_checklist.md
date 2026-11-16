# Model Development Guidelines: requirements checklist

Every item of the course's Model Development Guidelines mapped to the code, tests and evidence in this
folder. Paths are relative to `Model-Pipeline/` unless they start with `.github/`. All numbers come from the
full run on 13,666 movies (split hash `30100836287cbcfc…`). Test counts: 195 tests, coverage 98% of
`src/vibespace_model`. CI evidence: `model-train` on `main`, [run 43800488718](https://github.com/nikhilsk98/VibeSpace/actions/runs/43800488718); release tag `model-pipeline-v1.0`.

## 1. Model development and ML code

| # | Requirement | Implementation (files) | Tests | Evidence |
|---|---|---|---|---|
| 1.1 | Load data from the data pipeline (versioned) | `src/vibespace_model/data_loader.py` (reads `Data-Pipeline/data/processed/`, MD5 check against `Data-Pipeline/dvc.lock`, join, validation, data version), `configs/model_config.yaml` (`paths`, `data`) | `tests/test_data_loader.py` (happy path, hash mismatch, missing file, lock problems, duplicates, label range, embedding shape / order, leakage guard, join counts) | README §8; `reports/part1/part1_summary.json` (`join_report`, `data_version`: 13,816 → −150 → 13,666 movies, hash check passed) |
| 1.2 | Train models and select the best one | `train.py` (4 candidates: mean baseline, Ridge, LightGBM, MLP), `models/`, `select.py` (`select_best`: lowest validation MAE, Spearman tie-break), `workflows.py` (`run_tune`) | `tests/test_models.py`, `tests/test_select.py`, `tests/test_part1_integration.py`, `tests/test_part2_integration.py` | `reports/part1/model_comparison.csv` (LightGBM 0.1643 selected in Part 1), `reports/part2/default_vs_tuned.csv` (tuned MLP 0.1541 selected) |
| 1.3 | Model validation on a hold-out set | `splits.py` (70/15/15 by movie, stratified, saved assignment + SHA-256), `evaluate.py` (MAE, RMSE, R², Spearman), `select.py` (`validation_gate`: ≥ 15% better than baseline, R² ≥ 0.30); test split used only after selection | `tests/test_splits.py` (deterministic, disjoint, ratios, test guard), `tests/test_evaluate.py`, `tests/test_select.py` | README §10, §12, §20; `reports/part2/final_candidate_val_vs_test.csv` (val 0.1541 / test 0.1563) |
| 1.4 | Model bias detection (slicing) | `bias.py` (`evaluate_slices` with Fairlearn `MetricFrame` per slice of language group, era, primary genre, popularity bucket, short overview; bootstrap CIs for small slices), `configs/bias.yaml` | `tests/test_bias.py` (slice metrics with a built-in gap, small-slice exemption, bootstrap CI) | `reports/part3/slices_val_final.csv`, `slices_test_final.csv`, `slice_mae_val_final.png`, `bias_report.md`; `docs/bias.md` |
| 1.5 | Code to check bias, plus mitigation | `bias.py` (`bias_gate`: gated slices ≥ 30 movies must have MAE ≤ 1.25 × overall; `reweighting_weights`, `resample_indices` on the training split only), `workflows.py` (`run_bias`: both mitigations on the final family) | `tests/test_bias.py` (gate pass / fail, re-weighting, re-sampling touches training rows only), `tests/test_part3_integration.py` | `reports/part3/mitigation_comparison.csv` (re-weighting: worst ratio 1.104 → 1.084, MAE +4.2%; re-sampling +0.6%), `docs/bias.md` (trade-off) |
| 1.6 | Push the model to an artifact / model registry | Local: `tracking.py` (MLflow registry, aliases `candidate`, `champion-candidate`, `champion`). GCP: `registry.py` (versioned bundles + SHA-256 in Artifact Registry `vibespace-models`, metadata in `gs://vibespace-model-registry-nsk`, protected champion package `movie-dna-champions`), `promote.py`, `registry_cli.py`, `infra/gcp_setup.sh` | `tests/test_tracking.py`, `tests/test_registry.py`, `tests/test_promote.py`, `tests/test_champion_store.py` | README §18; `infra/README.md`; laptop champions `v20261115-005730-e63b305` (identical predictions) and `v20261115-011119-e63b305` pulled back from GCP with valid checksums; current champion `v20261116-035311-46abd39`, promoted by `model-train` on `main` |

## 2. Hyperparameter tuning

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 2.1 | Tuning with a documented search space and process | `tuning.py` (Optuna TPE, seed 42, MedianPruner, SQLite studies that resume, objective = mean validation MAE, test movies rejected by `TuningData`), `configs/tuning.yaml` (search spaces, budgets 25 / 50 / 40 trials, 90 min cap) | `tests/test_tuning.py` (search-space validation, finite objective, determinism, pruning, resume, caps, test-split guard), `tests/test_part2_integration.py` | `docs/tuning.md` (search spaces, results, selection reasoning); `reports/part2/best_params.json`; tuning took 57.9 min |

## 3. Experiment tracking and results

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 3.1 | Experiment tracking (MLflow) | `tracking.py` (SQLite store `mlflow/mlflow.db`, experiment `vibespace-movie-dna`, parent run per command, nested runs per candidate / Optuna trial / bias experiment, params, per-dimension metrics, data version + split hash + Git commit tags, artifacts) | `tests/test_tracking.py`, `tests/test_tuning.py` (trial runs nested under the parent) | README §7; MLflow UI (`mlflow ui --port 5001`); Docker compose `mlflow` service verified to show the runs |
| 3.2 | Results visualisations and model comparison | `evaluate.py` (comparison bar chart, per-dimension chart, predicted vs true), `workflows.py` (Optuna history, parameter importance, parallel coordinates, default vs tuned) | `tests/test_evaluate.py`, `tests/test_part2_integration.py` (plots exist) | `reports/part1/val_mae_by_model.png`, `val_mae_per_dimension.png`, `pred_vs_true_val.png`; `reports/part2/default_vs_tuned.png`, `optuna_*`, `variants_val_mae.png` |

## 4. Model sensitivity analysis

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 4.1 | Feature importance (SHAP) | `explain.py` (SHAP: TreeExplainer for LightGBM, GradientExplainer for the MLP, exact linear SHAP for Ridge; grouped by feature type; permutation importance by group on validation), `workflows.py` (`run_explain`) | `tests/test_explain.py` (group sums, additivity, permutation importance finds the known feature) | `docs/explainability.md` (permutation: genres +0.0765, popularity +0.0417, plot embedding +0.0193; SHAP shares and why group sums favour the 384-dim embedding); `reports/part3/grouped_importance.png`, `shap_heatmap_mlp.png`, `shap_metadata_mlp.png` |
| 4.2 | Hyperparameter sensitivity | `sensitivity.py` (fANOVA importance per study, one-at-a-time sweeps of the top-3 parameters) | `tests/test_sensitivity.py` | `docs/sensitivity.md` (MLP `weight_decay` importance 0.38, collapse to the baseline at ≥ 1.5e-3); `reports/part2/sensitivity/sweeps_mlp.png`, `param_importances.png` |

## 5. Model bias detection (slicing)

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 5.1 | Slice the data into subgroups | `configs/bias.yaml` (`slice_columns`), `bias.py` (`slice_values`, `evaluate_slices`); movie attributes only (no user demographics exist) | `tests/test_bias.py` | `docs/bias.md` ("Why the slices are movie attributes") |
| 5.2 | Track metrics across slices | Fairlearn `MetricFrame` (MAE, RMSE, count) per slice on validation (decisions) and test (report); bootstrap 95% CIs for slices < 30 movies | `tests/test_bias.py` | `reports/part3/slices_val_*.csv`, `slices_test_final.csv`, `slice_mae_*_final.png` |
| 5.3 | Mitigate bias | re-weighting and re-sampling (`bias.py`), retraining with the tuned parameters (`workflows.run_bias`) | `tests/test_bias.py`, `tests/test_part3_integration.py` | `reports/part3/mitigation_comparison.csv` |
| 5.4 | Document bias and trade-offs | `docs/bias.md`, `reports/part3/bias_report.md` (written by the `bias` command) | `tests/test_part3_integration.py` (report written) | final selection: unmitigated tuned MLP (passes the gate; mitigations cost accuracy) |
| 5.5 | Select the model after bias checks | `select.py` (`final_selection`: best validation MAE among models passing both gates; smallest gap + flag if none pass) | `tests/test_select.py` (all branches), `tests/test_part3_integration.py` ("none pass" branch → exit code 2) | `reports/part3/part3_summary.json` (`decision`: tuned MLP, not flagged) |

## 6. CI/CD pipeline automation

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 6.1 | CI/CD for model training (trigger on code / data changes) | `.github/workflows/model-train.yml` (push to `main`, `model-*`, `data-pipeline` touching `Model-Pipeline/**`, `Data-Pipeline/dvc.lock`, `Data-Pipeline/configs/**`; manual run with `dry_run`), `.github/workflows/model-ci.yml` (lint + tests on PRs), `src/vibespace_model/ci_run.py` (`ci-run`: train with tuned params, no Optuna) | `tests/test_ci_run.py` | actionlint 0 findings; local `act` runs of both workflows passed; on GitHub `model-train` ran green on `main` after the merge ([run 43800488718](https://github.com/nikhilsk98/VibeSpace/actions/runs/43800488718)) and `model-ci` runs on every PR; `docs/ci_setup_for_later.md` |
| 6.2 | Automated model validation with a threshold | `ci-run --ci-step validate` (`select.validation_gate`); failure → exit 1, the job stops | `tests/test_ci_run.py` (validation failure blocks the registry) | `act` run with a forced impossible R² threshold failed at "Validation gate", promote skipped; GitHub [run 43800488718](https://github.com/nikhilsk98/VibeSpace/actions/runs/43800488718): passed (MAE 38.5% lower than the baseline, R² 0.513) |
| 6.3 | Automated bias detection that blocks deployment | `ci-run --ci-step bias` (`bias.bias_gate`); failure → exit 2, registry push blocked | `tests/test_ci_run.py` (bias failure blocks the registry; promote guard) | `act` run with a forced bias failure failed at "Bias gate", promote skipped; GitHub [run 43800488718](https://github.com/nikhilsk98/VibeSpace/actions/runs/43800488718): passed (worst gated slice 1.102 × overall) |
| 6.4 | Push to the model registry | `ci-run --ci-step promote` → `promote.promote` (push + champion rule) to Artifact Registry / bucket; keyless Workload Identity Federation (`infra/gcp_setup.sh`) | `tests/test_promote.py`, `tests/test_ci_run.py` (dry run never calls gcloud; GCP mode builds gcloud commands) | real `ci-run` against GCP: promoted within tolerance, champion `v20261115-011119-e63b305`; on GitHub: `v20261116-034810-ec4bd52` (`ml-integration`) and `v20261116-035311-46abd39` (`main`, current champion) pushed and promoted within tolerance |
| 6.5 | Notifications and alerts | job summary (`ci_run.job_summary`), "Notification message" step and optional Slack (`ci/notify.py`, only when `SLACK_WEBHOOK_URL` exists), `::warning::` annotation for "not promoted", GitHub failure emails | `tests/test_notify.py` (payloads per event, annotation, no-op without webhook, post to a local server), `tests/test_ci_run.py` | `act` logs show the summary, the notification text and the NOT PROMOTED warning; GitHub [run 43800488718](https://github.com/nikhilsk98/VibeSpace/actions/runs/43800488718) job summary starts with `PROMOTED: v20261116-035311-46abd39 is the new champion` |
| 6.6 | Rollback mechanism | champion / challenger rule (`promote.decide`: "not promoted (rollback)" keeps the champion), `promote.rollback` (to previous or a given version, checksum verified), protected champion package `movie-dna-champions` (Keep cleanup rule) so past champions always exist | `tests/test_promote.py` (rollback to previous / specific / missing / corrupted), `tests/test_champion_store.py` (rollback after cleanup, champions never deleted) | real GCP rollback test (TEST candidate not promoted, temporary switch rolled back); applied cleanup policy in `infra/README.md` |

## 7. Code implementation and containerisation

| # | Requirement | Implementation | Tests | Evidence |
|---|---|---|---|---|
| 7.1 | Docker / containerised training | `Dockerfile` (python:3.11-slim, CPU PyTorch, pinned requirements, non-root), `docker-compose.yaml` (`trainer` + `mlflow` UI), `.dockerignore` | `model-ci.yml` under `act` (Linux container) | container `train` and `run-all` matched the host (same selection, MLP within 3.6e-5 MAE); image pushed to `vibespace-images/trainer` (by the `trainer image` job on `main`: `trainer:46abd39` and `:latest`) |
| 7.2 | Code for loading data | `data_loader.py`, `features.py` | `tests/test_data_loader.py`, `tests/test_features.py` | README §8–9 |
| 7.3 | Code for training and selection | `train.py`, `models/`, `tuning.py`, `select.py` | `tests/test_models.py`, `tests/test_tuning.py`, `tests/test_select.py` | README §11–13, §17 |
| 7.4 | Code for validation | `evaluate.py`, `select.validation_gate`, `splits.py` | `tests/test_evaluate.py`, `tests/test_select.py`, `tests/test_splits.py` | README §12 |
| 7.5 | Code for bias checking | `bias.py`, `workflows.run_bias` | `tests/test_bias.py`, `tests/test_part3_integration.py` | README §16 |
| 7.6 | Code for model selection after bias checking | `select.final_selection`, `workflows.run_bias` | `tests/test_select.py`, `tests/test_part3_integration.py` | README §17 |
| 7.7 | Code for pushing the model to the registry | `registry.py`, `promote.py`, `registry_cli.py`, `ci_run.step_promote` | `tests/test_registry.py`, `tests/test_promote.py`, `tests/test_champion_store.py`, `tests/test_ci_run.py` | README §18 |

## 8. Quality

| Item | Evidence |
|---|---|
| Tests | 195 offline tests (`pytest`), coverage 98% (`model-ci.yml` fails under 85%) |
| Style | `ruff check` and `ruff format --check` clean (`ruff.toml`) |
| Workflows | `actionlint` (with shellcheck) 0 findings on `.github/workflows/` |
| Reproducibility | fixed seeds; two Part 1 runs bit-identical; Docker matches within tolerance |
| No secrets | keyless Workload Identity Federation; no service-account keys; `.env`, data, models, MLflow stores and Optuna studies are git-ignored |
