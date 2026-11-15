# Bias detection, mitigation and final selection (Part 3)

Command: `python -m vibespace_model.cli bias --config configs/model_config.yaml` (settings in
`configs/bias.yaml`). Full tables: `reports/part3/bias_report.md`, `slices_*.csv`, `mitigation_comparison.csv`.

## Why the slices are movie attributes

MovieLens (and the VibeSpace app) holds **no user demographics**: no age, gender or location. The Movie DNA
predictor scores movies, so the fairness question is whether it is markedly worse for some *kinds of movies*
(e.g. non-English films, old films, niche genres), which would make the app recommend those films less well.
Slices:

| Slice column | Values |
|---|---|
| `language_group` | english, other_international, hindi, indian_regional, unknown |
| `era` | pre_1980, 1980_1999, 2000_plus |
| `primary_genre` | 19 genres |
| `popularity_bucket` | low, mid, high |
| `flag_short_overview` | True / False (no movie in the catalog has a short overview, so only False occurs) |

## Method

- **Metrics:** Fairlearn `MetricFrame` with MAE and RMSE over all 12 dimensions, per slice, on the
  **validation** split (decisions) and the **test** split (final report).
- **Bias gate:** every slice with **≥ 30 movies** must have MAE ≤ **1.25 ×** the overall MAE.
- **Small slices** (< 30 movies) are reported with a bootstrap 95% confidence interval (1,000 resamples of
  per-movie errors) and marked "insufficient data"; they do not gate. Hindi films have 11 movies in validation
  and 12 in test; Indian regional films have 1 in each.
- **Mitigation**, run once as an experiment even though nothing failed, on the final candidate's family
  (tuned MLP, same parameters), applied to the slice column with the largest gap (`primary_genre`):
  - **re-weighting:** inverse-frequency sample weights `1 / share`, capped at 20, normalised to mean 1;
  - **re-sampling:** slices smaller than 10% of the largest slice are oversampled with replacement, in the
    **training split only**.

## Results (validation)

All candidates pass both gates:

| Model | Val MAE | Validation gate | Bias gate | Worst gated slice | Ratio to overall |
|---|---|---|---|---|---|
| **mlp_tuned** | **0.1541** | pass | pass | primary_genre=romance | 1.104 |
| mlp_resampled | 0.1551 | pass | pass | primary_genre=romance | 1.100 |
| mlp_reweighted | 0.1605 | pass | pass | primary_genre=drama | 1.084 |
| lightgbm_tuned | 0.1627 | pass | pass | primary_genre=romance | 1.097 |
| ridge_tuned | 0.1719 | pass | pass | primary_genre=romance | 1.092 |

Final model (tuned MLP), gated slices, validation (overall MAE 0.1541):

| Slice | n | MAE | Ratio | | Slice | n | MAE | Ratio |
|---|---|---|---|---|---|---|---|---|
| english | 1,699 | 0.152 | 0.98 | | drama | 540 | 0.166 | 1.08 |
| other_international | 338 | 0.166 | 1.08 | | comedy | 504 | 0.151 | 0.98 |
| 1980_1999 | 640 | 0.144 | 0.94 | | action | 185 | 0.146 | 0.95 |
| 2000_plus | 1,050 | 0.159 | 1.03 | | horror | 131 | 0.136 | 0.88 |
| pre_1980 | 360 | 0.159 | 1.03 | | romance | 48 | 0.170 | **1.10** |
| popularity high | 390 | 0.138 | 0.89 | | family | 41 | 0.122 | 0.79 |
| popularity mid | 872 | 0.150 | 0.98 | | fantasy | 45 | 0.122 | 0.79 |
| popularity low | 788 | 0.166 | 1.08 | | documentary | 64 | 0.159 | 1.03 |

Small slices (validation, not gated, bootstrap 95% CI): hindi n=11, MAE 0.170 [0.139, 0.201];
indian_regional n=1, 0.234; history n=13, 0.166 [0.139, 0.195]; music n=16, 0.178 [0.154, 0.202];
mystery n=29, 0.142 [0.125, 0.161]; war n=19, 0.146 [0.124, 0.173]; western n=20, 0.163 [0.142, 0.185];
tv_movie n=2, 0.166.

## Mitigation comparison (tuned MLP, slice column `primary_genre`)

| Variant | Val MAE | Val R² | Worst slice ratio | Worst / best gated slice | Gates |
|---|---|---|---|---|---|
| unmitigated | **0.1541** | **0.513** | 1.104 (romance) | 1.400 | pass / pass |
| re-weighted | 0.1605 | 0.478 | **1.084** (drama) | **1.396** | pass / pass |
| re-sampled | 0.1551 | 0.508 | 1.100 (romance) | 1.430 | pass / pass |

**Trade-off:** re-weighting narrows the worst slice gap slightly (1.104 → 1.084) but costs 4.2% overall MAE
(0.1541 → 0.1605) and 0.035 R², i.e. it makes the model worse for most movies to make it a little less
uneven. Re-sampling costs 0.6% MAE and does not reduce the gap (worst/best spread even grows to 1.43). Since
the unmitigated model already passes the bias gate with margin (worst 1.10 vs limit 1.25), neither
mitigation is worth its cost.

## Final selection

Rule: best validation MAE among models that pass **both** the validation gate and the bias gate
(unmitigated or mitigated). → **mlp_tuned** (not flagged).

Test split (reported once, after the decision): MAE **0.1563**, RMSE 0.2029, R² 0.500, Spearman 0.708.
Per dimension (test MAE): energy 0.136, intensity 0.166, pace 0.149, humor 0.141, darkness 0.151,
hopefulness 0.167, emotional_intensity 0.157, romance 0.176, suspense 0.137, complexity 0.171, realism 0.144,
inspiration 0.180.

Bias gate on test: passes; the worst gated slice is `primary_genre=documentary` (n=75, MAE 0.179,
ratio 1.148). Other notable test slices: science_fiction 1.11, drama 1.06, other_international 1.05,
low popularity 1.07.

The final model, its preprocessing and `model_card.json` (both gate results, slice summary, SHAP top groups,
data version, split hash, parameters, metrics) are in `artifacts/part3/final_model/`, and it is registered in
the local MLflow registry as version 4 of `vibespace-movie-dna` with alias **`champion-candidate`**.

## Limits of this analysis

- Hindi (11–12 films per split) and Indian regional films (1 per split) are far too few to gate or to judge;
  their confidence intervals are wide. This mirrors the data pipeline's finding that the Tag Genome covers
  mostly English and older or popular films. More data for these slices (e.g. the TMDB `rest` stage, which
  needs its own labels) is the real fix, not re-weighting.
- Popularity: less popular films have 8% higher error than average and 20% higher than popular films
  (validation 0.166 vs 0.138). This is within the gate, but worth monitoring, because popular films also
  have richer genome labels.
