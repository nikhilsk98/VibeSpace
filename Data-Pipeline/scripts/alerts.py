"""Alerting: Slack webhook and/or SMTP email, with a log-file fallback.

Every alert is appended to ``logs/alerts.log`` (one JSON object per line) so
there is always an audit trail. If ``SLACK_WEBHOOK_URL`` is set the alert is
posted to Slack; if ``SMTP_HOST`` and ``ALERT_EMAIL`` are set it is emailed.
Alerting never raises: a broken channel is logged and the pipeline continues.
"""

from __future__ import annotations

import json
import smtplib
from collections.abc import Callable
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

import requests

from scripts.config import PipelineConfig, load_config
from scripts.io_utils import ensure_dir
from scripts.logging_utils import get_logger

logger = get_logger(__name__)

ALERTS_LOG = "alerts.log"
SEVERITY_EMOJI = {
    "info": ":information_source:", "warning": ":warning:", "error": ":x:", "critical": ":rotating_light:",
}


def write_alert_log(cfg: PipelineConfig, record: dict[str, Any]) -> None:
    """Append one alert record to ``logs/alerts.log``."""
    path = ensure_dir(cfg.logs_dir) / ALERTS_LOG
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def send_slack(webhook_url: str, title: str, message: str, severity: str,
               session: requests.Session | None = None, timeout: float = 10.0) -> None:
    """Post an alert to a Slack incoming webhook (raises on HTTP errors)."""
    emoji = SEVERITY_EMOJI.get(severity, "")
    payload = {"text": f"{emoji} *{title}*\n{message}"}
    resp = (session or requests).post(webhook_url, json=payload, timeout=timeout)
    if resp.status_code >= 300:
        raise RuntimeError(f"Slack webhook returned HTTP {resp.status_code}")


def send_email(cfg: PipelineConfig, title: str, message: str,
               smtp_factory: Callable[..., Any] = smtplib.SMTP) -> None:
    """Send an alert email through the configured SMTP server."""
    msg = EmailMessage()
    msg["Subject"] = title
    msg["From"] = cfg.smtp_sender or cfg.smtp_user or cfg.alert_email
    msg["To"] = cfg.alert_email
    msg.set_content(message)
    with smtp_factory(cfg.smtp_host, cfg.smtp_port, timeout=15) as smtp:
        if cfg.smtp_use_tls:
            smtp.starttls()
        if cfg.smtp_user:
            smtp.login(cfg.smtp_user, cfg.smtp_password)
        smtp.send_message(msg)


def send_alert(title: str, message: str, severity: str = "error", cfg: PipelineConfig | None = None,
               details: dict[str, Any] | None = None, session: requests.Session | None = None,
               smtp_factory: Callable[..., Any] = smtplib.SMTP) -> list[str]:
    """Deliver an alert on every configured channel; return the channels that worked.

    ``"log"`` is always included because the alert is always written to the
    alerts log file.
    """
    try:
        cfg = cfg or load_config()
    except Exception as exc:  # noqa: BLE001 - alerting must never fail the caller
        logger.error("Alert config could not be loaded (%s); logging only", exc)
        cfg = PipelineConfig()
    record = {"time": datetime.now(UTC).isoformat(), "severity": severity, "title": title,
              "message": message, "details": details or {}}
    delivered: list[str] = []
    try:
        write_alert_log(cfg, record)
        delivered.append("log")
    except OSError as exc:
        logger.error("Could not write alerts log: %s", exc)

    if cfg.slack_webhook_url:
        try:
            send_slack(cfg.slack_webhook_url, title, message, severity, session=session)
            delivered.append("slack")
        except Exception as exc:  # noqa: BLE001
            logger.error("Slack alert failed: %s", type(exc).__name__)
    if cfg.smtp_host and cfg.alert_email:
        try:
            send_email(cfg, title, message, smtp_factory=smtp_factory)
            delivered.append("email")
        except Exception as exc:  # noqa: BLE001
            logger.error("Email alert failed: %s", type(exc).__name__)
    if delivered == ["log"]:
        logger.warning("ALERT (%s) %s - no Slack/email configured, written to %s", severity, title, ALERTS_LOG)
    else:
        logger.warning("ALERT (%s) %s - delivered via %s", severity, title, ", ".join(delivered))
    return delivered


def airflow_failure_callback(context: dict[str, Any]) -> None:
    """Airflow ``on_failure_callback``: alert with task, run, error and log link."""
    try:
        ti = context.get("task_instance") or context.get("ti")
        dag_id = getattr(ti, "dag_id", None) or getattr(context.get("dag"), "dag_id", "unknown")
        task_id = getattr(ti, "task_id", "unknown")
        exc = context.get("exception")
        title = f"VibeSpace pipeline task failed: {dag_id}.{task_id}"
        message = "\n".join([
            f"Run: {context.get('run_id')}",
            f"Try: {getattr(ti, 'try_number', '?')}",
            f"Error: {type(exc).__name__ if exc else 'unknown'}: {exc}",
            f"Log: {getattr(ti, 'log_url', 'n/a')}",
        ])
        send_alert(title, message, severity="error",
                   details={"dag_id": dag_id, "task_id": task_id, "run_id": context.get("run_id")})
    except Exception as cb_exc:  # noqa: BLE001 - a callback must never raise
        logger.error("Failure callback itself failed: %s", cb_exc)
