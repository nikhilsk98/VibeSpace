# Hyperparameter tuning (Part 2)

Command: `python -m vibespace_model.cli tune --config configs/model_config.yaml` (settings in
`configs/tuning.yaml`). All numbers below are from the full run on 13,666 movies (split hash
`30100836287cbcfc…`).

## Setup

| Item | Value |
|---|---|
| Library | Optuna 5.0 |
| Sampler | TPE (Bayesian), `seed=42`, 10 random start-up trials |
| Pruner | MedianPruner (`n_startup_trials=5`, `n_warmup_steps=10`) on the MLP's per-epoch validation MAE |
| Objective | mean validation MAE over the 12 DNA dimensions, on the saved Part 1 split |
| Data used | train (9,565) and validation (2,050) only; `TuningData` asserts that no test movie is present |
| Budgets | Ridge 25 trials, LightGBM 50, MLP 40; total cap 90 min, per-model caps Ridge 5 / LightGBM 55 / MLP 30 min |
| Storage | `artifacts/part2/optuna/optuna.db` (SQLite, git-ignored); a rerun resumes and only runs missing trials |
| Tracking | every trial is a nested MLflow run (params, `val_mae`, duration, pruned flag) under `part2-tune` |

### Search spaces

| Model | Parameter | Range |
|---|---|---|
| Ridge | `alpha` | 0.01 – 1000, log |
| LightGBM | `num_leaves` | 8 – 128, log (int) |
| | `learning_rate` | 0.01 – 0.2, log |
| | `max_depth` | {-1, 4, 6, 8, 12} |
| | `min_child_samples` | 5 – 100, log (int) |
| | `subsample` | 0.5 – 1.0 |
| | `colsample_bytree` | 0.1 – 1.0 |
| | `reg_alpha`, `reg_lambda` | 1e-8 – 10, log |
| | `n_estimators` | fixed upper bound 2000; early stopping (50 rounds, validation L1) picks the number of trees |
| MLP | `hidden_sizes` | {128, 256, 256-128, 512-256, 512-256-128} |
| | `dropout` | 0.0 – 0.5 |
| | `learning_rate` | 1e-4 – 1e-2, log |
| | `weight_decay` | 1e-7 – 1e-2, log |
| | `batch_size` | {64, 128, 256} |
| | `huber_delta` | {0.05, 0.1, 0.5} |
| | `max_epochs` / `patience` | fixed 200 / 15 (early stopping on validation loss) |

## Results

Total tuning time 57.9 min (within the 90 min cap).

| Model | Trials | Complete / pruned | Best trial | Best validation MAE | Part 1 default | Change |
|---|---|---|---|---|---|---|
| Ridge | 25 / 25 | 25 / 0 | #19 | 0.17195 | 0.17206 | −0.1% |
| LightGBM | 24 / 50 (stopped at its 55 min cap) | 24 / 0 | #23 | 0.16266 | 0.16427 | −1.0% |
| MLP | 40 / 40 | 21 / 19 | #32 | **0.15414** | 0.16925 | **−8.9%** |

LightGBM trials took 12–315 s (median 112 s), because slow learning rates grow up to 2,000 trees for each of
the 12 dimensions; the per-model cap stopped it after 24 of the 50 planned trials, and the best so far was
kept (the prompt's "stop early at the cap" rule). MLP trials took about 1.6 s on average thanks to pruning.

Best parameters (`reports/part2/best_params.json`):

- **Ridge:** `alpha = 4.29`.
- **LightGBM:** `learning_rate = 0.0100`, `num_leaves = 102`, `max_depth = -1`, `min_child_samples = 71`,
  `subsample = 0.771`, `colsample_bytree = 0.860`, `reg_alpha = 0.074`, `reg_lambda = 0.00044`.
- **MLP:** `hidden_sizes = [512, 256]`, `dropout = 0.199`, `learning_rate = 0.00062`,
  `weight_decay = 2.9e-6`, `batch_size = 128`, `huber_delta = 0.05`.

Default vs tuned (validation, `reports/part2/default_vs_tuned.csv`):

| Variant | MAE | RMSE | R² | Spearman | vs baseline |
|---|---|---|---|---|---|
| **mlp_tuned** | **0.1541** | 0.2008 | 0.513 | 0.717 | +38.5% |
| lightgbm_tuned | 0.1627 | 0.2012 | 0.512 | 0.710 | +35.1% |
| lightgbm_default | 0.1643 | 0.2030 | 0.503 | 0.703 | +34.5% |
| mlp_default | 0.1692 | 0.2112 | 0.462 | 0.671 | +32.5% |
| ridge_tuned | 0.1719 | 0.2109 | 0.464 | 0.680 | +31.4% |
| ridge_default | 0.1721 | 0.2107 | 0.465 | 0.681 | +31.4% |
| baseline | 0.2508 | 0.2891 | −0.001 | 0.000 | – |

## How the model was selected

1. Part 1 trained the four candidates with default parameters; LightGBM was best (validation MAE 0.1643).
2. Part 2 tuned Ridge, LightGBM and the MLP on the same split, retrained each best configuration on the
   training rows (validation only for early stopping) and compared all seven variants (4 defaults + 3 tuned)
   by **mean validation MAE** (tie-break: mean Spearman). The test split played no role in any of this.
3. The tuned MLP won by a clear margin: 0.1541 vs 0.1627 for the tuned LightGBM (−5.3%), and it is also best
   on RMSE, R² and Spearman. It is better than the tuned LightGBM on all 12 dimensions.
4. Validation gate: MAE improvement over the mean baseline 38.5% (≥ 15%) and mean R² 0.513 (≥ 0.30) → **passed**.
5. Only after the decision was the test split used, once, as a sanity check:

| Split | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|
| validation | 0.1541 | 0.2008 | 0.513 | 0.717 |
| test | 0.1563 | 0.2029 | 0.500 | 0.708 |

The small validation → test gap (+0.002 MAE) shows the choice did not overfit the validation split. The
tuned MLP was registered as version 3 of `vibespace-movie-dna` with alias `candidate` (the Part 1 versions are
kept).

Plots: `reports/part2/optuna_<model>_{optimization_history,param_importances,parallel_coordinate}.png`,
`default_vs_tuned.png`, `variants_val_mae.png`, `final_candidate_per_dimension_val_mae.png`.

## Notes

- LightGBM's best learning rate sits at the lower edge of its range (0.0100), so a wider range and a higher
  tree cap might help LightGBM a little, at a large time cost; with the MLP ahead by 5%, this was not pursued.
- Early stopping and the objective both use the validation split. This is standard practice, but it makes
  validation scores slightly optimistic; the test score above is the unbiased estimate.
- In Docker, `--tuned-params reports/part2/best_params.json` reuses these parameters instead of repeating
  the search.
