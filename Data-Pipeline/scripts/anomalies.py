"""Detect data anomalies and raise alerts.

Checks (each produces :class:`Anomaly` records with a severity):

* missing values  - null / empty rates above thresholds for key columns
* outliers        - runtimes of 0 or above the outlier limit, IQR outliers
* invalid formats - bad language codes, unparseable release dates
* schema          - missing expected columns or changed dtypes
* validation      - failed Great Expectations suites
* drift           - PSI shift of slice columns, null-rate jumps, row drops
* join quality    - match rate and quarantine volume

Anomalies of severity ``warning`` or higher trigger an alert. Critical
anomalies marked ``blocking`` fail the task (configurable).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from scripts.alerts import send_alert
from scripts.clean import CLEAN_FILE
from scripts.config import PipelineConfig
from scripts.exceptions import AnomalyError
from scripts.generate_stats import LATEST_FILE, run_profile, stats_dir
from scripts.io_utils import read_json, read_parquet, write_json
from scripts.join_ids import JOIN_STATS_FILE
from scripts.logging_utils import get_logger
from scripts.validate import VALIDATION_PROCESSED_FILE, VALIDATION_RAW_FILE

logger = get_logger(__name__)

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
ANOMALIES_FILE = "anomalies_current.json"

EXPECTED_SCHEMA = {
    "movieId": "int", "tmdbId": "int", "title": "object", "overview": "object", "runtime": "float",
    "language": "object", "language_group": "object", "era": "object", "primary_genre": "object",
    "popularity_bucket": "object", "vote_count": "float", "year": "Int",
}
# column -> (warning null rate, critical null rate)
NULL_THRESHOLDS = {"title": (0.0, 0.0), "overview": (0.05, 0.20), "runtime": (0.05, 0.20),
                   "release_date": (0.02, 0.10), "vote_count": (0.02, 0.10)}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LANG_RE = re.compile(r"^([a-z]{2}|unknown)$")


@dataclass
class Anomaly:
    """One detected anomaly."""

    check: str
    severity: str
    dataset: str
    message: str
    column: str | None = None
    value: Any = None
    threshold: Any = None
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for JSON output."""
        return asdict(self)


def _rate_severity(rate: float, warn: float, crit: float) -> str | None:
    """Severity of a rate against warning / critical thresholds."""
    if rate > crit or (crit == 0 and rate > 0):
        return "critical"
    if rate > warn:
        return "warning"
    return None


def check_missing_values(df: pd.DataFrame, thresholds: dict[str, tuple[float, float]] = NULL_THRESHOLDS
                         ) -> list[Anomaly]:
    """Null or blank rates above thresholds."""
    out: list[Anomaly] = []
    if df.empty:
        return [Anomaly("missing_values", "critical", "catalog", "Catalog is empty", blocking=True)]
    for col, (warn, crit) in thresholds.items():
        if col not in df:
            continue
        blank = df[col].isna()
        if df[col].dtype == object:
            blank |= df[col].astype(str).str.strip().eq("")
        rate = float(blank.mean())
        sev = _rate_severity(rate, warn, crit)
        if sev:
            out.append(Anomaly("missing_values", sev, "catalog", f"{col} missing in {rate:.1%} of rows",
                               col, round(rate, 6), warn, blocking=(col == "title")))
    return out


def check_outliers(df: pd.DataFrame, runtime_limit: int = 400) -> list[Anomaly]:
    """Runtime outliers (0, above the limit, or far outside the IQR fence)."""
    out: list[Anomaly] = []
    if "runtime" not in df or df.empty:
        return out
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    zero = int(runtime.eq(0).sum())
    over = int(runtime.gt(runtime_limit).sum())
    q1, q3 = runtime.quantile(0.25), runtime.quantile(0.75)
    fence_hi = q3 + 3 * (q3 - q1)
    iqr_out = int(runtime.gt(fence_hi).sum())
    n = len(df)
    for count, what, thr in ((zero, "runtime of 0 minutes", 0), (over, f"runtime above {runtime_limit} min",
                                                                  runtime_limit)):
        if count:
            sev = "warning" if count / n > 0.01 else "info"
            out.append(Anomaly("outliers", sev, "catalog", f"{count} movies with {what}", "runtime", count, thr))
    if iqr_out:
        sev = "warning" if iqr_out / n > 0.02 else "info"
        out.append(Anomaly("outliers", sev, "catalog",
                           f"{iqr_out} runtimes above the 3xIQR fence ({fence_hi:.0f} min)", "runtime",
                           iqr_out, round(float(fence_hi), 1)))
    return out


def check_formats(df: pd.DataFrame) -> list[Anomaly]:
    """Invalid language codes and release dates."""
    out: list[Anomaly] = []
    if df.empty:
        return out
    if "language" in df:
        bad = ~df["language"].fillna("").astype(str).str.match(_LANG_RE)
        if bad.any():
            out.append(Anomaly("invalid_format", "critical", "catalog",
                               f"{int(bad.sum())} invalid language codes", "language", int(bad.sum()),
                               blocking=True))
        unknown = float(df["language"].eq("unknown").mean())
        if unknown > 0.02:
            out.append(Anomaly("invalid_format", "warning", "catalog",
                               f"{unknown:.1%} of movies have an unknown language", "language",
                               round(unknown, 6), 0.02))
    if "release_date" in df:
        dates = df["release_date"].dropna().astype(str)
        bad_dates = int((~dates.str.match(_DATE_RE)).sum())
        if bad_dates:
            sev = "warning" if bad_dates / len(df) > 0.01 else "info"
            out.append(Anomaly("invalid_format", sev, "catalog", f"{bad_dates} malformed release dates",
                               "release_date", bad_dates))
    return out


def check_schema(df: pd.DataFrame, expected: dict[str, str] = EXPECTED_SCHEMA) -> list[Anomaly]:
    """Missing columns or dtype families that do not match the expected schema."""
    out: list[Anomaly] = []
    for col, family in expected.items():
        if col not in df:
            out.append(Anomaly("schema", "critical", "catalog", f"Missing column {col}", col, blocking=True))
            continue
        dtype = str(df[col].dtype)
        ok = (family == "object" and dtype in {"object", "string"}) or dtype.startswith(family) \
            or (family == "float" and dtype.startswith("Float"))
        if not ok:
            out.append(Anomaly("schema", "critical", "catalog", f"{col} has dtype {dtype}, expected {family}*",
                               col, dtype, family, blocking=True))
    return out


def check_validation(reports: dict[str, dict[str, Any] | None]) -> list[Anomaly]:
    """Failed GX suites from the saved validation reports."""
    out: list[Anomaly] = []
    for _stage, report in reports.items():
        if not report:
            continue
        for name, suite in report.get("suites", {}).items():
            if not suite.get("success", True):
                out.append(Anomaly("validation", "critical", name,
                                   f"GX suite {name} failed {suite.get('failed')} expectation(s)",
                                   value=suite.get("failed"), blocking=True))
    return out


def check_drift(drift: dict[str, Any], cfg: PipelineConfig) -> list[Anomaly]:
    """PSI shifts, null-rate jumps and row drops versus the previous run."""
    out: list[Anomaly] = []
    if not drift or drift.get("previous_run_id") is None:
        return out
    for col, value in drift.get("psi", {}).items():
        sev = "critical" if value >= cfg.psi_critical else "warning" if value >= cfg.psi_warning else None
        if sev:
            out.append(Anomaly("drift", sev, "catalog", f"{col} distribution shifted (PSI {value:.3f})",
                               col, value, cfg.psi_warning))
    for name, info in drift.get("datasets", {}).items():
        for col, delta in info.get("null_rate_changes", {}).items():
            if delta > cfg.null_rate_jump:
                out.append(Anomaly("drift", "warning", name, f"{col} null rate rose by {delta:.1%}", col,
                                   delta, cfg.null_rate_jump))
        change = info.get("row_change_pct")
        if change is not None and change < -cfg.row_count_drop:
            out.append(Anomaly("drift", "warning", name, f"row count fell by {abs(change):.1%}", value=change,
                               threshold=-cfg.row_count_drop))
        for col in info.get("removed_columns", []):
            out.append(Anomaly("schema", "critical", name, f"column {col} disappeared since last run", col,
                               blocking=True))
    return out


def check_join(stats: dict[str, Any] | None, min_rate: float) -> list[Anomaly]:
    """Match rate close to (or below) the threshold and quarantine volume."""
    if not stats:
        return []
    out: list[Anomaly] = []
    rate = stats["match_rate"]
    if rate < min_rate:
        out.append(Anomaly("join_quality", "critical", "join", f"match rate {rate:.1%} below {min_rate:.0%}",
                           value=rate, threshold=min_rate, blocking=True))
    elif rate < min_rate + 0.03:
        out.append(Anomaly("join_quality", "warning", "join", f"match rate {rate:.1%} is close to the minimum",
                           value=rate, threshold=min_rate))
    if stats.get("year_mismatches"):
        share = stats["year_mismatches"] / max(stats["matched"], 1)
        sev = "warning" if share > 0.02 else "info"
        out.append(Anomaly("join_quality", sev, "join", f"{stats['year_mismatches']} release-year mismatches "
                           f"({share:.1%})", value=stats["year_mismatches"]))
    return out


def detect_anomalies(catalog: pd.DataFrame, cfg: PipelineConfig, drift: dict[str, Any] | None = None,
                     validation_reports: dict[str, Any] | None = None,
                     join_stats: dict[str, Any] | None = None) -> list[Anomaly]:
    """Run every check and return anomalies sorted by severity (highest first)."""
    found = (check_schema(catalog) + check_missing_values(catalog)
             + check_outliers(catalog, cfg.runtime_outlier_minutes) + check_formats(catalog)
             + check_validation(validation_reports or {}) + check_drift(drift or {}, cfg)
             + check_join(join_stats, cfg.min_match_rate))
    return sorted(found, key=lambda a: -SEVERITY_ORDER[a.severity])


def summarise(anomalies: list[Anomaly]) -> dict[str, int]:
    """Count anomalies per severity."""
    counts = {s: 0 for s in SEVERITY_ORDER}
    for a in anomalies:
        counts[a.severity] += 1
    return counts


def _maybe_read(path: Any) -> Any:
    """Read JSON if the file exists, else ``None``."""
    return read_json(path) if path.is_file() else None


def run_detect_anomalies(cfg: PipelineConfig, run_id: str | None = None) -> dict[str, Any]:
    """Detect anomalies on the current outputs, save them and alert when needed."""
    catalog = read_parquet(cfg.processed_dir / CLEAN_FILE)
    latest = _maybe_read(stats_dir(cfg, run_profile(cfg.tmdb_stage, cfg.sample_size)) / LATEST_FILE)
    reports = {"raw": _maybe_read(cfg.metrics_dir / VALIDATION_RAW_FILE),
               "processed": _maybe_read(cfg.metrics_dir / VALIDATION_PROCESSED_FILE)}
    join_stats = _maybe_read(cfg.interim_dir / JOIN_STATS_FILE)
    anomalies = detect_anomalies(catalog, cfg, (latest or {}).get("drift"), reports, join_stats)
    counts = summarise(anomalies)
    payload = {"run_id": run_id, "counts": counts, "anomalies": [a.to_dict() for a in anomalies]}
    write_json(cfg.metrics_dir / ANOMALIES_FILE, payload)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (run_id or "manual"))
    write_json(cfg.metrics_dir / "anomalies" / f"anomalies_{safe}.json", payload)
    for a in anomalies:
        log = logger.warning if SEVERITY_ORDER[a.severity] >= 1 else logger.info
        log("Anomaly [%s] %s/%s: %s", a.severity, a.check, a.dataset, a.message)

    alertable = [a for a in anomalies if SEVERITY_ORDER[a.severity] >= SEVERITY_ORDER["warning"]]
    if alertable:
        lines = [f"- [{a.severity}] {a.check}: {a.message}" for a in alertable[:20]]
        top = "critical" if counts["critical"] else "warning"
        send_alert(f"VibeSpace data anomalies ({counts['critical']} critical, {counts['warning']} warning)",
                   "\n".join(lines), severity=top, cfg=cfg, details={"run_id": run_id, "counts": counts})
    blocking = [a for a in anomalies if a.severity == "critical" and a.blocking]
    if blocking and cfg.fail_on_critical_anomaly:
        raise AnomalyError("Blocking anomalies found: " + "; ".join(a.message for a in blocking))
    return {"counts": counts, "alert_sent": bool(alertable)}
