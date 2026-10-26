"""Tests for scripts.alerts (fake Slack / SMTP, no network)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from scripts import alerts as al
from tests.helpers import FakeResponse, FakeSession


class FakeSMTP:
    """Records what an SMTP client was asked to do."""

    sent: list = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.actions = host, port, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def starttls(self):
        self.actions.append("tls")

    def login(self, user, password):
        self.actions.append(("login", user))

    def send_message(self, msg):
        FakeSMTP.sent.append((self, msg))


class BrokenSMTP(FakeSMTP):
    def send_message(self, msg):
        raise OSError("smtp down")


def read_log(cfg):
    return [json.loads(line) for line in (cfg.logs_dir / "alerts.log").read_text().splitlines()]


def test_fallback_to_log_file(cfg):
    channels = al.send_alert("Title", "Body", severity="warning", cfg=cfg, details={"k": 1})
    assert channels == ["log"]
    record = read_log(cfg)[0]
    assert record["title"] == "Title" and record["severity"] == "warning" and record["details"] == {"k": 1}


def test_slack_delivery(cfg):
    session = FakeSession(responses=[FakeResponse(200)])
    channels = al.send_alert("T", "B", cfg=replace(cfg, slack_webhook_url="https://hooks.example/abc"),
                             session=session)
    assert channels == ["log", "slack"]
    assert "*T*" in session.calls[0]["json"]["text"]


def test_slack_failure_does_not_raise(cfg):
    session = FakeSession(responses=[FakeResponse(500)])
    channels = al.send_alert("T", "B", cfg=replace(cfg, slack_webhook_url="https://hooks.example/abc"),
                             session=session)
    assert channels == ["log"]


def test_email_delivery(cfg):
    FakeSMTP.sent = []
    mail_cfg = replace(cfg, smtp_host="smtp.example", alert_email="team@example.com", smtp_user="u",
                       smtp_password="p")
    channels = al.send_alert("Subject", "Body", cfg=mail_cfg, smtp_factory=FakeSMTP)
    assert channels == ["log", "email"]
    client, msg = FakeSMTP.sent[0]
    assert msg["To"] == "team@example.com" and msg["Subject"] == "Subject"
    assert client.actions == ["tls", ("login", "u")]


def test_email_failure_does_not_raise(cfg):
    mail_cfg = replace(cfg, smtp_host="smtp.example", alert_email="team@example.com")
    assert al.send_alert("S", "B", cfg=mail_cfg, smtp_factory=BrokenSMTP) == ["log"]


def test_unwritable_log_still_returns(cfg, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("file, not a folder")
    assert al.send_alert("S", "B", cfg=replace(cfg, logs_dir=blocker / "logs")) == []


def test_send_alert_without_cfg_uses_environment(tmp_path):
    assert al.send_alert("S", "B") == ["log"]
    assert (tmp_path / "logs" / "alerts.log").is_file()


def test_failure_callback_writes_alert(tmp_path):
    ti = SimpleNamespace(dag_id="vibespace_data_pipeline", task_id="clean", try_number=2, log_url="http://x/log")
    al.airflow_failure_callback({"task_instance": ti, "run_id": "manual__1", "exception": ValueError("boom")})
    record = [json.loads(x) for x in (tmp_path / "logs" / "alerts.log").read_text().splitlines()][0]
    assert record["title"].endswith("vibespace_data_pipeline.clean")
    assert "ValueError: boom" in record["message"] and record["details"]["run_id"] == "manual__1"


def test_failure_callback_never_raises(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("alerting broken")

    monkeypatch.setattr(al, "send_alert", explode)
    al.airflow_failure_callback({})  # must not raise
