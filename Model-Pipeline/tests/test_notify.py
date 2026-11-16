"""Part 5 Slack notification helper: payloads, no-op without a webhook, real HTTP post to a local server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ci import notify

STATE = {
    "event": "not_promoted",
    "val": {"mae": 0.1541, "r2": 0.513, "rmse": 0.2, "spearman": 0.71},
    "test": {"mae": 0.1563, "r2": 0.5, "rmse": 0.2, "spearman": 0.70},
    "validation_gate": {"passed": True, "reasons": ["ok"]},
    "bias_gate": {"passed": False, "reasons": ["slice drama exceeds 1.25 x overall MAE"]},
    "decision": {"label": "not promoted (rollback)", "reasons": ["worse than champion by 0.004"]},
    "version": "v20261115-010000-abc1234",
    "champion_after": {"version": "v20261115-005730-e63b305"},
    "dry_run": True,
}


@pytest.mark.parametrize("event", sorted(notify.EVENTS))
def test_every_event_has_a_message(event) -> None:
    payload = notify.build_message(event, STATE, run_url="https://example.test/run/1")
    assert notify.EVENTS[event][1] in payload["text"]
    assert payload["blocks"][0]["text"]["text"] == payload["text"]
    assert "Open the workflow run" in payload["text"]


def test_message_content() -> None:
    text = notify.build_message("not_promoted", STATE)["text"]
    assert "Validation MAE 0.1541" in text and "test MAE 0.1563" in text
    assert "bias gate: slice drama" in text
    assert "not promoted (rollback)" in text and "champion v20261115-005730-e63b305" in text
    assert "dry run" in text
    error = notify.build_message("pipeline_error", {"error": "SplitError: hash differs"})["text"]
    assert "SplitError" in error
    with pytest.raises(ValueError, match="unknown event"):
        notify.build_message("deployed", {})


def test_no_webhook_is_a_no_op() -> None:
    def explode(*args, **kwargs):
        raise AssertionError("must not open a connection")

    assert notify.send(None, {"text": "x"}, opener=explode) is False
    assert notify.send("", {"text": "x"}, opener=explode) is False


class _Recorder(BaseHTTPRequestHandler):
    received: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802  (http.server API)
        length = int(self.headers["Content-Length"])
        _Recorder.received.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args) -> None:
        pass


def test_post_to_a_fake_slack_server() -> None:
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/hook"
        payload = notify.build_message("promoted", STATE)
        assert notify.send(url, payload) is True
        assert _Recorder.received[-1] == payload
    finally:
        server.shutdown()


def test_main_reads_state_and_never_fails(tmp_path, monkeypatch, capsys) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(STATE))
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    assert notify.main(["--state", str(state_file)]) == 0
    out = capsys.readouterr().out
    assert "NOT PROMOTED: champion unchanged (rollback)" in out and "no webhook configured, skipped" in out
    assert notify.main(["--state", str(tmp_path / "missing.json")]) == 0
    assert "Pipeline error" in capsys.readouterr().out
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "http://127.0.0.1:9/unreachable")
    assert notify.main(["--state", str(state_file), "--event", "promoted"]) == 0
    assert "Slack notification failed" in capsys.readouterr().err


def test_not_promoted_annotation() -> None:
    line = notify.annotation(STATE)
    assert line.startswith("::warning title=NOT PROMOTED: champion unchanged (rollback)::")
    assert "champion v20261115-005730-e63b305 is unchanged" in line
    assert "worse than champion by 0.004" in line
    escaped = notify.annotation({**STATE, "decision": {"reasons": ["50% worse\nsecond line"]}})
    assert "50%25 worse%0Asecond line" in escaped and "\n" not in escaped
    assert notify.annotation({**STATE, "event": "promoted"}) is None
    assert "no reason recorded" in notify.annotation({"event": "not_promoted"})


def test_main_annotate(tmp_path, capsys) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(STATE))
    assert notify.main(["--state", str(state_file), "--annotate"]) == 0
    assert capsys.readouterr().out.startswith("::warning title=NOT PROMOTED")
    state_file.write_text(json.dumps({**STATE, "event": "promoted"}))
    assert notify.main(["--state", str(state_file), "--annotate"]) == 0
    assert "No annotation" in capsys.readouterr().out
