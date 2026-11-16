# Feature importance and explainability (Part 3)

Command: `python -m vibespace_model.cli explain --config configs/model_config.yaml`. Outputs in
`reports/part3/` (`explain_summary.json`, `shap_*`, `permutation_importance_*`, `grouped_importance.png`).

## Method

- **Models explained:** the final candidate (tuned MLP) and the next-best other family (tuned LightGBM).
- **SHAP:** MLP with `shap.GradientExplainer` (expected gradients, 100 samples per movie); LightGBM with
  `shap.TreeExplainer` on each of the 12 boosters. 1,000 randomly chosen **test** movies are explained; the
  background is 200 training movies (seed 42).
- **Grouping:** |SHAP| is summed over the members of each feature group (the 384 embedding dimensions become
  one `plot_embedding` value), then averaged over movies, per DNA dimension and overall. Shares are
  normalised to sum to 1.
- **Cross-check:** permutation importance by group on the **validation** set: all columns of a group are
  shuffled together (5 repeats) and the increase in validation MAE is recorded. This is model-agnostic.

Note that summing |SHAP| over many features favours large groups (384 embedding dimensions vs one runtime
column), so the permutation test is the better guide to how much the model *relies* on a group.

## Results: importance by feature group

| Group | MLP SHAP share | MLP permutation (MAE increase) | LightGBM SHAP share | LightGBM permutation |
|---|---|---|---|---|
| plot_embedding | **0.603** | 0.0193 | **0.567** | 0.0192 |
| genres | 0.226 | **0.0765** | 0.253 | **0.0619** |
| popularity | 0.109 | 0.0417 | 0.116 | 0.0304 |
| year | 0.028 | 0.0084 | 0.031 | 0.0065 |
| language | 0.022 | 0.0046 | 0.021 | 0.0036 |
| runtime | 0.012 | 0.0022 | 0.012 | 0.0013 |
| flags | 0.000 | 0.0000 | 0.000 | 0.0000 |

(Validation MAE of the MLP is 0.1541, so shuffling the genres raises it by about 50%.)

Per DNA dimension (MLP, share of |SHAP| within each dimension; `shap_heatmap_mlp.png`):

| Dimension | embedding | genres | popularity | year | language | runtime |
|---|---|---|---|---|---|---|
| energy | 0.62 | 0.21 | 0.12 | 0.03 | 0.02 | 0.01 |
| intensity | 0.61 | 0.28 | 0.08 | 0.01 | 0.01 | 0.01 |
| pace | 0.61 | 0.21 | 0.12 | 0.02 | 0.02 | 0.01 |
| humor | 0.50 | **0.33** | 0.12 | 0.02 | 0.02 | 0.01 |
| darkness | 0.63 | 0.23 | 0.10 | 0.02 | 0.02 | 0.01 |
| hopefulness | **0.67** | 0.19 | 0.06 | 0.04 | 0.03 | 0.01 |
| emotional_intensity | 0.54 | 0.23 | **0.17** | 0.04 | 0.01 | 0.02 |
| romance | 0.63 | 0.23 | 0.07 | 0.03 | 0.03 | 0.02 |
| suspense | 0.54 | 0.24 | **0.17** | 0.03 | 0.02 | 0.01 |
| complexity | 0.60 | 0.21 | 0.14 | 0.02 | 0.01 | 0.01 |
| realism | 0.54 | 0.25 | 0.12 | 0.04 | 0.03 | 0.01 |
| inspiration | **0.68** | 0.14 | 0.08 | 0.04 | 0.04 | 0.02 |

Top metadata features (mean |SHAP| over movies and dimensions, MLP): `vote_average_scaled` 0.075,
`genre=comedy` 0.041, `genre=drama` 0.033, `release_year_scaled` 0.032, `log1p_vote_count` 0.029,
`genre=romance` 0.028, `genre=action` 0.025, `genre=thriller` 0.024. LightGBM has the same top features in
almost the same order.

## Plain-language findings

- **The plot summary carries most of the signal.** About 60% of the attribution goes to the plot embedding,
  for both model families. Its permutation effect is smaller than its SHAP share because 384 correlated
  dimensions spread the information (shuffling them all together still costs 0.019 MAE).
- **Genres are the single most relied-on metadata.** Shuffling genres hurts more than shuffling the plot
  embedding. They matter most for **humor** (33% of its attribution: `genre=comedy` is a strong hint) and
  **intensity**, and least for **inspiration**.
- **Darkness, hopefulness, romance and inspiration are read mostly from the plot text** (63–68% embedding):
  the tone is in the story, not in the genre label.
- **Popularity and rating** (`vote_average`, vote count) help most for **suspense** and **emotional
  intensity** (17% each): highly rated, much-voted films tend to be the tense, emotionally heavy ones in the
  genome labels.
- **Year, language and runtime** add a little (1–4% each); the quality **flags** contribute nothing (they
  are almost always false in this catalog: e.g. no short overviews at all).
- Both families agree on the ranking of groups, so the conclusions are not an artefact of one explainer.

## Plots

`grouped_importance.png` (SHAP share and permutation importance per group, both models),
`shap_heatmap_mlp.png` / `shap_heatmap_lightgbm.png` (groups × dimensions),
`shap_metadata_mlp.png` / `shap_metadata_lightgbm.png` (SHAP beeswarm of the metadata features for
`humor` and `darkness`).
