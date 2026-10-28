"""Tests for scripts.anomalies."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import anomalies as an
from scripts.clean import CLEAN_FILE
from scripts.exceptions import AnomalyError
from scripts.io_utils import write_json, write_parquet
from scripts.join_ids import JOIN_STATS_FILE
from scripts.validate import VALIDATION_PROCESSED_FILE


def catalog(n: int = 100, **overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "movieId": range(n), "tmdbId": range(1000, 1000 + n), "title": [f"T{i}" for i in range(n)],
        "overview": ["A plot."] * n, "runtime": [100.0] * n, "language": ["en"] * n,
        "language_group": ["english"] * n, "era": ["2000_plus"] * n, "primary_genre": ["drama"] * n,
        "popularity_bucket": ["mid"] * n, "vote_count": [10.0] * n,
        "year": pd.array([2000] * n, dtype="Int64"), "release_date": ["2000-01-01"] * n,
    })
    for col, values in overrides.items():
        df[col] = values
    return df


def checks(anoms):
    return {(a.check, a.severity, a.column) for a in anoms}


def test_clean_catalog_has_no_anomalies(cfg):
    assert an.detect_anomalies(catalog(), cfg) == []


def test_missing_values():
    df = catalog()
    df.loc[:9, "overview"] = None          # 10% -> warning
    df.loc[:30, "runtime"] = None          # 31% -> critical
    df.loc[0, "title"] = " "               # any missing title -> critical + blocking
    found = an.check_missing_values(df)
    assert ("missing_values", "warning", "overview") in checks(found)
    assert ("missing_values", "critical", "runtime") in checks(found)
    assert any(a.column == "title" and a.blocking for a in found)
    assert an.check_missing_values(df.iloc[0:0])[0].blocking


def test_outliers():
    df = catalog()
    df.loc[:4, "runtime"] = 0.0
    df.loc[5, "runtime"] = 900.0
    found = an.check_outliers(df, 400)
    assert ("outliers", "warning", "runtime") in checks(found)
    assert any("above 400" in a.message for a in found)
    assert an.check_outliers(catalog().iloc[0:0]) == []


def test_invalid_formats():
    df = catalog()
    df.loc[0, "language"] = "ENGLISH"
    df.loc[1:5, "language"] = "unknown"
    df.loc[6, "release_date"] = "01/02/2003"
    found = an.check_formats(df)
    kinds = {(a.severity, a.column) for a in found}
    assert ("critical", "language") in kinds and ("warning", "language") in kinds
    assert any(a.column == "release_date" for a in found)


def test_schema_violations():
    df = catalog().drop(columns=["era"]).assign(movieId=lambda d: d["movieId"].astype(str))
    found = an.check_schema(df)
    assert {a.column for a in found} == {"era", "movieId"} and all(a.blocking for a in found)


def test_validation_reports():
    report = {"suites": {"processed_catalog": {"success": False, "failed": 2}, "x": {"success": True}}}
    found = an.check_validation({"processed": report, "raw": None})
    assert len(found) == 1 and found[0].severity == "critical"


def test_drift(cfg):
    drift = {"previous_run_id": "r1", "psi": {"language_group": 0.3, "era": 0.12, "primary_genre": 0.01},
             "datasets": {"catalog": {"null_rate_changes": {"overview": 0.2}, "row_change_pct": -0.5,
                                      "removed_columns": ["runtime"]}}}
    found = an.check_drift(drift, cfg)
    assert ("drift", "critical", "language_group") in checks(found)
    assert ("drift", "warning", "era") in checks(found)
    assert not any(a.column == "primary_genre" for a in found)
    assert any("row count fell" in a.message for a in found)
    assert any(a.check == "schema" and a.column == "runtime" for a in found)
    assert an.check_drift({"previous_run_id": None}, cfg) == []


def test_join_quality():
    assert an.check_join({"match_rate": 0.5, "matched": 5, "year_mismatches": 0}, 0.9)[0].blocking
    near = an.check_join({"match_rate": 0.91, "matched": 100, "year_mismatches": 5}, 0.9)
    assert {a.severity for a in near} == {"warning"}
    assert an.check_join(None, 0.9) == []


def test_run_detect_anomalies_alerts_and_saves(cfg):
    df = catalog()
    df.loc[:30, "overview"] = None
    write_parquet(cfg.processed_dir / CLEAN_FILE, df)
    write_json(cfg.interim_dir / JOIN_STATS_FILE, {"match_rate": 0.99, "matched": 100, "year_mismatches": 0})
    out = an.run_detect_anomalies(cfg, run_id="run/1")
    assert out["alert_sent"] and out["counts"]["critical"] == 1
    saved = json.loads((cfg.metrics_dir / an.ANOMALIES_FILE).read_text())
    assert saved["counts"]["critical"] == 1
    assert (cfg.metrics_dir / "anomalies" / "anomalies_run_1.json").is_file()
    alerts = (cfg.logs_dir / "alerts.log").read_text()
    assert "VibeSpace data anomalies" in alerts


def test_blocking_anomaly_fails_task(cfg):
    write_parquet(cfg.processed_dir / CLEAN_FILE, catalog())
    write_json(cfg.metrics_dir / VALIDATION_PROCESSED_FILE,
               {"suites": {"processed_catalog": {"success": False, "failed": 1}}})
    with pytest.raises(AnomalyError, match="processed_catalog"):
        an.run_detect_anomalies(cfg)


def test_no_alert_when_clean(cfg):
    write_parquet(cfg.processed_dir / CLEAN_FILE, catalog())
    assert an.run_detect_anomalies(cfg) == {"counts": {"info": 0, "warning": 0, "critical": 0}, "alert_sent": False}
    assert not (cfg.logs_dir / "alerts.log").exists()
