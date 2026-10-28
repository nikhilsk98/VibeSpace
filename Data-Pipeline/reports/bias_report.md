# Bias report: data slicing, Fairlearn analysis and mitigation

Generated 2026-10-28 00:49 UTC from 13,666 movies that have a clean catalog entry, Movie DNA labels and a plot embedding.

## Why slices are movie attributes

MovieLens 25M ratings are anonymous and the VibeSpace app does not collect user demographics, so there are no user groups to compare. The realistic risk for a recommender is unequal treatment of *content*: if the Movie DNA model is less accurate for non-English, older, niche-genre or less-popular films, those films get recommended for the wrong moods and lose exposure. We therefore slice by `language_group`, `era`, `primary_genre` and `popularity_bucket`.

## 1. Representation

Movie counts and Tag Genome label coverage per slice (whole clean catalog).

### language_group

| language_group | movies | with_labels | share | label_coverage |
|---|---|---|---|---|
| english | 11325 | 11325 | 0.829 | 1.000 |
| other_international | 2252 | 2252 | 0.165 | 1.000 |
| hindi | 76 | 76 | 0.006 | 1.000 |
| indian_regional | 9 | 9 | 0.001 | 1.000 |
| unknown | 4 | 4 | 0.000 | 1.000 |

### era

| era | movies | with_labels | share | label_coverage |
|---|---|---|---|---|
| 2000_plus | 6854 | 6854 | 0.502 | 1.000 |
| 1980_1999 | 4307 | 4307 | 0.315 | 1.000 |
| pre_1980 | 2505 | 2505 | 0.183 | 1.000 |

### primary_genre

| primary_genre | movies | with_labels | share | label_coverage |
|---|---|---|---|---|
| drama | 3608 | 3608 | 0.264 | 1.000 |
| comedy | 3196 | 3196 | 0.234 | 1.000 |
| action | 1350 | 1350 | 0.099 | 1.000 |
| horror | 936 | 936 | 0.068 | 1.000 |
| adventure | 657 | 657 | 0.048 | 1.000 |
| crime | 635 | 635 | 0.046 | 1.000 |
| documentary | 578 | 578 | 0.042 | 1.000 |
| thriller | 478 | 478 | 0.035 | 1.000 |
| animation | 421 | 421 | 0.031 | 1.000 |
| romance | 346 | 346 | 0.025 | 1.000 |
| science_fiction | 318 | 318 | 0.023 | 1.000 |
| fantasy | 261 | 261 | 0.019 | 1.000 |
| family | 226 | 226 | 0.017 | 1.000 |
| mystery | 176 | 176 | 0.013 | 1.000 |
| western | 141 | 141 | 0.010 | 1.000 |
| music | 122 | 122 | 0.009 | 1.000 |
| war | 119 | 119 | 0.009 | 1.000 |
| history | 72 | 72 | 0.005 | 1.000 |
| tv_movie | 26 | 26 | 0.002 | 1.000 |

### popularity_bucket

| popularity_bucket | movies | with_labels | share | label_coverage |
|---|---|---|---|---|
| mid | 5471 | 5471 | 0.400 | 1.000 |
| low | 5462 | 5462 | 0.400 | 1.000 |
| high | 2733 | 2733 | 0.200 | 1.000 |

## 2. Label distribution per slice

Mean Movie DNA percentile per slice (labelled movies).

### language_group

| language_group | energy | intensity | pace | humor | darkness | hopefulness | emotional_intensity | romance | suspense | complexity | realism | inspiration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| english | 0.539 | 0.486 | 0.540 | 0.512 | 0.467 | 0.532 | 0.477 | 0.508 | 0.485 | 0.472 | 0.497 | 0.515 |
| hindi | 0.403 | 0.438 | 0.402 | 0.483 | 0.436 | 0.682 | 0.747 | 0.738 | 0.481 | 0.509 | 0.708 | 0.668 |
| indian_regional | 0.227 | 0.248 | 0.203 | 0.209 | 0.658 | 0.156 | 0.712 | 0.427 | 0.438 | 0.464 | 0.486 | 0.318 |
| other_international | 0.313 | 0.567 | 0.308 | 0.440 | 0.659 | 0.337 | 0.597 | 0.450 | 0.568 | 0.629 | 0.500 | 0.412 |
| unknown | 0.245 | 0.506 | 0.209 | 0.610 | 0.710 | 0.171 | 0.516 | 0.238 | 0.623 | 0.635 | 0.439 | 0.304 |

### era

| era | energy | intensity | pace | humor | darkness | hopefulness | emotional_intensity | romance | suspense | complexity | realism | inspiration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1980_1999 | 0.540 | 0.496 | 0.541 | 0.512 | 0.463 | 0.527 | 0.476 | 0.497 | 0.454 | 0.462 | 0.469 | 0.502 |
| 2000_plus | 0.522 | 0.504 | 0.509 | 0.493 | 0.482 | 0.547 | 0.511 | 0.514 | 0.477 | 0.511 | 0.512 | 0.531 |
| pre_1980 | 0.374 | 0.490 | 0.408 | 0.495 | 0.607 | 0.327 | 0.501 | 0.463 | 0.633 | 0.526 | 0.514 | 0.406 |

### popularity_bucket

| popularity_bucket | energy | intensity | pace | humor | darkness | hopefulness | emotional_intensity | romance | suspense | complexity | realism | inspiration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 0.633 | 0.537 | 0.637 | 0.578 | 0.462 | 0.600 | 0.491 | 0.561 | 0.531 | 0.477 | 0.455 | 0.598 |
| low | 0.436 | 0.471 | 0.433 | 0.472 | 0.504 | 0.458 | 0.509 | 0.478 | 0.462 | 0.502 | 0.531 | 0.459 |
| mid | 0.500 | 0.508 | 0.500 | 0.488 | 0.512 | 0.493 | 0.491 | 0.490 | 0.518 | 0.505 | 0.488 | 0.489 |

## 3. Model behaviour per slice (diagnostic model)

A Ridge regression from plot embedding + metadata (runtime, year, log vote count, rating, language group) to the 12 DNA scores, trained on a 70/15/15 split by movie (9,566/2,050/2,050). It exists only to measure per-slice error with Fairlearn `MetricFrame`; it is not the Movie DNA model. MAE is averaged over the 12 dimensions on the held-out test movies.

Overall test MAE (baseline): **0.1963** (alpha=1.0).

A slice is a *significant* gap when it has at least 15 test movies and its MAE is more than 10% above the overall MAE.

### language_group

| language_group | test_movies | mae_baseline | mae_reweighted | mae_resampled |
|---|---|---|---|---|
| english | 1688 | 0.1948 | 0.1958 | 0.1949 |
| other_international | 347 | 0.2033 | 0.2027 | 0.2030 |
| hindi | 11 | 0.2035 | 0.2051 | 0.2041 |
| indian_regional | 2 | 0.1563 | 0.1498 | 0.1559 |
| unknown | 2 | 0.2008 | 0.1932 | 0.2067 |

### era

| era | test_movies | mae_baseline | mae_reweighted | mae_resampled |
|---|---|---|---|---|
| 2000_plus | 1008 | 0.1987 | 0.2001 | 0.1986 |
| 1980_1999 | 662 | 0.1869 | 0.1875 | 0.1871 |
| pre_1980 | 380 | 0.2059 | 0.2055 | 0.2060 |

### primary_genre

| primary_genre | test_movies | mae_baseline | mae_reweighted | mae_resampled |
|---|---|---|---|---|
| drama | 545 | 0.2080 | 0.2086 | 0.2080 |
| comedy | 473 | 0.1925 | 0.1955 | 0.1928 |
| action | 209 | 0.1815 | 0.1819 | 0.1810 |
| horror | 136 | 0.1878 | 0.1892 | 0.1883 |
| crime | 99 | 0.2085 | 0.2092 | 0.2080 |
| adventure | 86 | 0.1969 | 0.1979 | 0.1980 |
| documentary | 84 | 0.1925 | 0.1906 | 0.1924 |
| thriller | 76 | 0.1917 | 0.1901 | 0.1917 |
| animation | 69 | 0.1962 | 0.1931 | 0.1954 |
| romance | 45 | 0.1875 | 0.1910 | 0.1878 |
| fantasy | 43 | 0.1919 | 0.1897 | 0.1907 |
| science_fiction | 43 | 0.1940 | 0.1949 | 0.1938 |
| mystery | 35 | 0.1997 | 0.2011 | 0.2011 |
| family | 33 | 0.1793 | 0.1778 | 0.1798 |
| war | 22 | 0.1846 | 0.1795 | 0.1838 |
| western | 21 | 0.2094 | 0.2015 | 0.2091 |
| music | 18 | 0.2011 | 0.2042 | 0.2017 |
| history | 10 | 0.1941 | 0.1953 | 0.1958 |
| tv_movie | 3 | 0.2040 | 0.2069 | 0.2036 |

### popularity_bucket

| popularity_bucket | test_movies | mae_baseline | mae_reweighted | mae_resampled |
|---|---|---|---|---|
| mid | 835 | 0.1961 | 0.1966 | 0.1960 |
| low | 804 | 0.1979 | 0.1990 | 0.1980 |
| high | 411 | 0.1934 | 0.1939 | 0.1935 |

## 4. Mitigation

* **Re-weighting:** each training movie gets an inverse-frequency weight averaged over the four slice columns (clipped to (0.2, 10.0), mean 1; observed range 0.53-11.02).
* **Re-sampling:** small `primary_genre` slices (the column with the largest baseline gap) are oversampled to at least half the median slice size.

| Run | Overall MAE | Worst gap language | Worst gap era | Worst gap genre | Worst gap popularity |
|---|---|---|---|---|---|
| baseline | 0.1963 | 0.0084 | 0.0190 | 0.0301 | 0.0045 |
| reweighted | 0.1970 | 0.0069 | 0.0180 | 0.0314 | 0.0051 |
| resampled | 0.1963 | 0.0082 | 0.0190 | 0.0293 | 0.0045 |

## 5. Findings in plain language

- **Representation is the main bias.** English-language films are 82.9% of the catalog, other international films 16.5%, Hindi 0.56% (76 films) and Indian regional languages 0.07% (9 films). Tag Genome labels exist mostly for well-known, frequently rated films, so this skew is inherited by the Movie DNA labels.
- **Model behaviour is fairly even.** No slice with at least 15 test movies has an MAE more than 10% above the overall MAE (0.1963).
- The largest baseline gap is in `primary_genre`: 0.0301 MAE between `family` (0.1793) and `western` (0.2094).
- **Too small to judge** (fewer than 15 test movies): language_group=hindi (11), language_group=unknown (2), language_group=indian_regional (2), primary_genre=tv_movie (3), primary_genre=history (10).
- **Re-weighting** changes overall MAE by +0.0008 and worst gaps by: language_group 0.0084->0.0069, era 0.0190->0.0180, primary_genre 0.0301->0.0314, popularity_bucket 0.0045->0.0051.
- **Re-sampling** changes overall MAE by +0.0000 and worst gaps by: language_group 0.0084->0.0082, era 0.0190->0.0190, primary_genre 0.0301->0.0293, popularity_bucket 0.0045->0.0045.
- **Trade-off:** mitigation shifts model capacity towards small slices, which usually costs a little overall accuracy. Rule used: adopt a mitigation only if it narrows most worst-case gaps and raises overall MAE by at most 2%.
- **Decision:** adopt re-sampling for the Movie DNA model training. Because error gaps are small but representation is very uneven, the priority is more data for under-represented slices: run TMDB stage `rest` for catalog coverage, keep inverse-frequency weights available in training, and re-check these tables on every pipeline run.
