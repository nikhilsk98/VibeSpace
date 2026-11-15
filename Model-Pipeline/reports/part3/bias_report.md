# Bias report: Movie DNA predictor (Part 3)

Written by `python -m vibespace_model.cli bias`. Slices are movie attributes; MovieLens and the app hold no user demographics. Gate: slices with at least 30 movies must have MAE <= 1.25 x overall MAE (validation split decides; test split is reported). Smaller slices are reported with a bootstrap 95% CI and marked 'insufficient data'.

## Candidates and gates (validation)

| model | val_mae | validation_gate | bias_gate | worst_slice | worst_ratio |
|---|---|---|---|---|---|
| mlp_tuned | 0.1541 | pass | pass | primary_genre=romance | 1.1037 |
| lightgbm_tuned | 0.1627 | pass | pass | primary_genre=romance | 1.0975 |
| ridge_tuned | 0.1719 | pass | pass | primary_genre=romance | 1.0921 |
| mlp_reweighted | 0.1605 | pass | pass | primary_genre=drama | 1.0841 |
| mlp_resampled | 0.1551 | pass | pass | primary_genre=romance | 1.1002 |

## Mitigation on `mlp` (slice column `primary_genre`)

| model | kind | val_mae | val_r2 | worst_slice_ratio | worst_slice | max_worst_over_best | validation_gate | bias_gate |
|---|---|---|---|---|---|---|---|---|
| mlp_tuned | tuned | 0.1541 | 0.5134 | 1.1037 | primary_genre=romance | 1.4002 | pass | pass |
| mlp_reweighted | reweighted | 0.1605 | 0.4781 | 1.0841 | primary_genre=drama | 1.3963 | pass | pass |
| mlp_resampled | resampled | 0.1551 | 0.5084 | 1.1002 | primary_genre=romance | 1.4295 | pass | pass |

Validation MAE per `primary_genre` slice and variant:

| slice | n | tuned | reweighted | resampled |
|---|---|---|---|---|
| action | 185 | 0.1458 | 0.1524 | 0.1463 |
| adventure | 78 | 0.1451 | 0.1464 | 0.1440 |
| animation | 72 | 0.1545 | 0.1582 | 0.1562 |
| comedy | 504 | 0.1513 | 0.1585 | 0.1506 |
| crime | 113 | 0.1609 | 0.1704 | 0.1620 |
| documentary | 64 | 0.1589 | 0.1619 | 0.1592 |
| drama | 540 | 0.1662 | 0.1740 | 0.1680 |
| family | 41 | 0.1220 | 0.1246 | 0.1256 |
| fantasy | 45 | 0.1215 | 0.1274 | 0.1193 |
| history | 13 | 0.1655 | 0.1680 | 0.1579 |
| horror | 131 | 0.1363 | 0.1409 | 0.1376 |
| music | 16 | 0.1776 | 0.1820 | 0.1862 |
| mystery | 29 | 0.1424 | 0.1538 | 0.1427 |
| romance | 48 | 0.1701 | 0.1739 | 0.1706 |
| science_fiction | 49 | 0.1506 | 0.1536 | 0.1521 |
| thriller | 81 | 0.1589 | 0.1612 | 0.1629 |
| tv_movie | 2 | 0.1660 | 0.1566 | 0.1622 |
| war | 19 | 0.1462 | 0.1574 | 0.1570 |
| western | 20 | 0.1632 | 0.1761 | 0.1674 |

## Final decision

- Final model: **mlp_tuned** (best validation MAE among models passing the validation and bias gates).
- Flagged for review: no.
- Validation MAE 0.1541; test MAE 0.1563, test R² 0.500, test Spearman 0.708.

## Final model: gated slices (validation)

| slice_column | slice | n | mae | mae_ratio | status |
|---|---|---|---|---|---|
| language_group | english | 1699 | 0.1516 | 0.9834 | pass |
| language_group | other_international | 338 | 0.1662 | 1.0783 | pass |
| era | 1980_1999 | 640 | 0.1441 | 0.9350 | pass |
| era | 2000_plus | 1050 | 0.1586 | 1.0289 | pass |
| era | pre_1980 | 360 | 0.1590 | 1.0314 | pass |
| primary_genre | action | 185 | 0.1458 | 0.9458 | pass |
| primary_genre | adventure | 78 | 0.1451 | 0.9412 | pass |
| primary_genre | animation | 72 | 0.1545 | 1.0024 | pass |
| primary_genre | comedy | 504 | 0.1513 | 0.9815 | pass |
| primary_genre | crime | 113 | 0.1609 | 1.0441 | pass |
| primary_genre | documentary | 64 | 0.1589 | 1.0311 | pass |
| primary_genre | drama | 540 | 0.1662 | 1.0780 | pass |
| primary_genre | family | 41 | 0.1220 | 0.7915 | pass |
| primary_genre | fantasy | 45 | 0.1215 | 0.7883 | pass |
| primary_genre | horror | 131 | 0.1363 | 0.8844 | pass |
| primary_genre | romance | 48 | 0.1701 | 1.1037 | pass |
| primary_genre | science_fiction | 49 | 0.1506 | 0.9767 | pass |
| primary_genre | thriller | 81 | 0.1589 | 1.0306 | pass |
| popularity_bucket | high | 390 | 0.1379 | 0.8944 | pass |
| popularity_bucket | low | 788 | 0.1663 | 1.0788 | pass |
| popularity_bucket | mid | 872 | 0.1504 | 0.9760 | pass |
| flag_short_overview | False | 2050 | 0.1541 | 1.0000 | pass |

## Final model: small slices (validation, not gated)

| slice_column | slice | n | mae | ci_low | ci_high |
|---|---|---|---|---|---|
| language_group | hindi | 11 | 0.1701 | 0.1392 | 0.2012 |
| language_group | indian_regional | 1 | 0.2342 | 0.2342 | 0.2342 |
| language_group | unknown | 1 | 0.1646 | 0.1646 | 0.1646 |
| primary_genre | history | 13 | 0.1655 | 0.1393 | 0.1952 |
| primary_genre | music | 16 | 0.1776 | 0.1539 | 0.2021 |
| primary_genre | mystery | 29 | 0.1424 | 0.1247 | 0.1608 |
| primary_genre | tv_movie | 2 | 0.1660 | 0.1186 | 0.2133 |
| primary_genre | war | 19 | 0.1462 | 0.1236 | 0.1725 |
| primary_genre | western | 20 | 0.1632 | 0.1415 | 0.1854 |

## Final model: all slices (test, reporting only)

| slice_column | slice | n | mae | mae_ratio | status |
|---|---|---|---|---|---|
| language_group | english | 1699 | 0.1549 | 0.9904 | pass |
| language_group | hindi | 12 | 0.1708 | 1.0922 | insufficient data |
| language_group | indian_regional | 1 | 0.0810 | 0.5181 | insufficient data |
| language_group | other_international | 338 | 0.1637 | 1.0470 | pass |
| language_group | unknown | 1 | 0.1107 | 0.7080 | insufficient data |
| era | 1980_1999 | 633 | 0.1475 | 0.9432 | pass |
| era | 2000_plus | 1048 | 0.1618 | 1.0349 | pass |
| era | pre_1980 | 370 | 0.1561 | 0.9983 | pass |
| primary_genre | action | 222 | 0.1398 | 0.8942 | pass |
| primary_genre | adventure | 90 | 0.1454 | 0.9302 | pass |
| primary_genre | animation | 65 | 0.1480 | 0.9467 | pass |
| primary_genre | comedy | 487 | 0.1559 | 0.9970 | pass |
| primary_genre | crime | 88 | 0.1588 | 1.0160 | pass |
| primary_genre | documentary | 75 | 0.1795 | 1.1478 | pass |
| primary_genre | drama | 551 | 0.1659 | 1.0613 | pass |
| primary_genre | family | 37 | 0.1338 | 0.8559 | pass |
| primary_genre | fantasy | 42 | 0.1582 | 1.0119 | pass |
| primary_genre | history | 12 | 0.1771 | 1.1329 | insufficient data |
| primary_genre | horror | 150 | 0.1380 | 0.8830 | pass |
| primary_genre | music | 22 | 0.1541 | 0.9858 | insufficient data |
| primary_genre | mystery | 16 | 0.1564 | 1.0005 | insufficient data |
| primary_genre | romance | 48 | 0.1522 | 0.9736 | pass |
| primary_genre | science_fiction | 45 | 0.1735 | 1.1099 | pass |
| primary_genre | thriller | 61 | 0.1595 | 1.0202 | pass |
| primary_genre | tv_movie | 5 | 0.1721 | 1.1005 | insufficient data |
| primary_genre | war | 17 | 0.1832 | 1.1719 | insufficient data |
| primary_genre | western | 18 | 0.1668 | 1.0666 | insufficient data |
| popularity_bucket | high | 421 | 0.1405 | 0.8989 | pass |
| popularity_bucket | low | 816 | 0.1665 | 1.0647 | pass |
| popularity_bucket | mid | 814 | 0.1544 | 0.9874 | pass |
| flag_short_overview | False | 2051 | 0.1563 | 1.0000 | pass |
