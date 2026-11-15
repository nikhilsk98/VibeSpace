# Hyperparameter sensitivity (Part 2)

Command: `python -m vibespace_model.cli sensitivity --config configs/model_config.yaml`. Outputs in
`reports/part2/sensitivity/`.

## Method

1. **Parameter importance** for every tuned model: Optuna's fANOVA evaluator (seed 42) on the completed
   trials of each study, normalised to sum to 1.
2. **One-at-a-time sweeps** for the final candidate (the tuned MLP): its 3 most important hyperparameters are
   varied one at a time over a grid (7 log-spaced values for numeric parameters plus the best value; every
   choice for categorical ones) while all other parameters stay at their best values. Each point is a full
   training run on the train split, scored on validation MAE.

## Parameter importance (fANOVA)

| MLP | importance | LightGBM | importance |
|---|---|---|---|
| `weight_decay` | 0.384 | `learning_rate` | 0.266 |
| `hidden_sizes` | 0.230 | `colsample_bytree` | 0.193 |
| `huber_delta` | 0.158 | `max_depth` | 0.163 |
| `dropout` | 0.112 | `subsample` | 0.134 |
| `learning_rate` | 0.085 | `min_child_samples` | 0.121 |
| `batch_size` | 0.031 | `num_leaves` | 0.082 |
| | | `reg_alpha` | 0.037 |
| | | `reg_lambda` | 0.004 |

Ridge has a single parameter (`alpha`, importance 1.0); its MAE barely moved across 25 trials
(best 0.17195 vs default 0.17206).

## Sweeps of the final candidate (tuned MLP)

| Parameter | Values tried → validation MAE | Spread |
|---|---|---|
| `weight_decay` | 1e-7: 0.1554 · 6.8e-7: 0.1550 · **2.9e-6: 0.1541** · 4.6e-6: 0.1543 · 3.2e-5: 0.1636 · 2.2e-4: 0.2088 · 1.5e-3: 0.2508 · 1e-2: 0.2508 | 0.0966 (63%) |
| `hidden_sizes` | 128: 0.1558 · 256: 0.1545 · 256-128: 0.1549 · **512-256: 0.1541** · 512-256-128: 0.1558 | 0.0017 (1.1%) |
| `huber_delta` | **0.05: 0.1541** · 0.1: 0.1547 · 0.5: 0.1579 | 0.0038 (2.4%) |

Plot: `reports/part2/sensitivity/sweeps_mlp.png`; raw values: `sweeps_mlp.csv`, `sweep_summary_mlp.csv`.

## Findings

- **Weight decay is the parameter that matters.** Below about 5e-6 the model is stable (0.154–0.155). From
  about 3e-5 the error rises quickly, and at 1.5e-3 or more the network collapses to the mean prediction:
  0.2508 is exactly the mean baseline's MAE. With Adam, L2 decay this strong shrinks the 421 input weights to
  nothing. Keep it at or below 1e-5.
- **Architecture is not critical.** Every width from 128 to 512-256-128 lands within 0.002 MAE; the
  2-layer 512-256 network is best, and a third layer slightly hurts.
- **Huber delta:** a small delta (0.05, close to L1 / MAE) is best, which fits an MAE objective; 0.5 (close
  to MSE) costs about 2.4%.
- **LightGBM** is most sensitive to the learning rate, and the best value is at the lower bound of the range
  (0.01), so a slower rate with more trees could still help it.

## Practical takeaways

- When retraining (e.g. in CI, Part 5), the tuned MLP is robust to width and batch size, so they can stay
  fixed; `weight_decay` should be guarded (≤ 1e-5) because large values silently produce a mean predictor.
  The validation gate catches that case (R² would fall to about 0).
- The MLP trains in about 3 s on CPU, so a small re-tune is cheap; LightGBM tuning is the expensive part
  (median 112 s per trial).
