"""Slack notification for the training workflow (standard library only).

Reads the ``state.json`` written by ``ci-run`` and posts one message for the run's event:
``training_completed``, ``validation_failed``, ``bias_failed``, ``promoted``, ``not_promoted`` or
``pipeline_error``. Without a webhook (``SLACK_WEBHOOK_URL`` unset or empty) it prints the message and does
nothing else, so a missing secret never fails a run.

    python ci/notify.py --state ci-out/state.json [--event pipeline_error] [--run-url URL]
    python ci/notify.py --state ci-out/state.json --annotate     # GitHub ::warning:: for "not promoted"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

EVENTS = {
    "training_completed": ("white_check_mark", "Training completed"),
    "validation_failed": ("x", "Validation gate failed: model not registered"),
    "bias_failed": ("warning", "Bias gate failed: registry push blocked"),
    "promoted": ("trophy", "New champion promoted"),
    "not_promoted": ("leftwards_arrow_with_hook", "NOT PROMOTED: champion unchanged (rollback)"),
    "pipeline_error": ("rotating_light", "Pipeline error"),
}


def load_state(path: Path) -> dict[str, Any]:
    """``state.json`` of the run (empty when the run failed before writing it)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def build_message(event: str, state: dict[str, Any], run_url: str | None = None) -> dict[str, Any]:
    """Slack payload (``text`` + ``blocks``) for one event."""
    if event not in EVENTS:
        raise ValueError(f"unknown event '{event}'; choose from {sorted(EVENTS)}")
    emoji, title = EVENTS[event]
    lines = [f":{emoji}: *VibeSpace Movie DNA: {title}*"]
    if state.get("val"):
        val, test = state["val"], state.get("test", {})
        test_mae = test.get("mae", float("nan"))
        lines.append(f"Validation MAE {val['mae']:.4f} (R² {val['r2']:.3f}) · test MAE {test_mae:.4f}")
    for gate in ("validation_gate", "bias_gate"):
        if state.get(gate) and not state[gate].get("passed"):
            lines.append(f"{gate.replace('_', ' ')}: " + "; ".join(state[gate].get("reasons", [])))
    if state.get("decision"):
        champion = (state.get("champion_after") or {}).get("version")
        lines.append(
            f"Decision: {state['decision']['label']} · candidate {state.get('version')} · champion {champion}"
        )
        lines += [f"• {reason}" for reason in state["decision"].get("reasons", [])]
    if state.get("error"):
        lines.append(f"Error: {state['error']}")
    if state.get("dry_run"):
        lines.append("(dry run: local registry folder, nothing pushed to GCP)")
    if run_url:
        lines.append(f"<{run_url}|Open the workflow run>")
    text = "\n".join(lines)
    return {"text": text, "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]}


def _escape_annotation(text: str) -> str:
    """GitHub workflow-command escaping (``%``, newlines) for annotation messages."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotation(state: dict[str, Any]) -> str | None:
    """``::warning::`` line for a not-promoted run (None for every other outcome)."""
    if state.get("event") != "not_promoted":
        return None
    decision = state.get("decision") or {}
    champion = (state.get("champion_after") or {}).get("version")
    reasons = "; ".join(decision.get("reasons", [])) or "no reason recorded"
    message = (
        f"Candidate {state.get('version')} was not promoted; champion {champion} is unchanged. "
        f"Reasons: {reasons}"
    )
    return f"::warning title={EVENTS['not_promoted'][1]}::{_escape_annotation(message)}"


def send(
    webhook: str | None, payload: dict[str, Any], opener: Callable[..., Any] = urllib.request.urlopen
) -> bool:
    """POST the payload to the webhook; returns False (no-op) without a webhook."""
    if not webhook:
        return False
    request = urllib.request.Request(  # noqa: S310  (https webhook URL from a repository secret)
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener(request, timeout=10) as response:
        return 200 <= response.status < 300


def main(argv: list[str] | None = None) -> int:
    """Build the message for the run, print it, and post it when a webhook is configured."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", default="ci-out/state.json")
    parser.add_argument("--event", default=None, help="override the event recorded in state.json")
    parser.add_argument("--run-url", default=None)
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="print a GitHub ::warning:: annotation for a not-promoted run, then exit",
    )
    args = parser.parse_args(argv)
    state = load_state(Path(args.state))
    if args.annotate:
        line = annotation(state)
        print(line if line else f"No annotation: event is {state.get('event')!r}")
        return 0
    event = args.event or state.get("event") or "pipeline_error"
    payload = build_message(event, state, args.run_url)
    print(payload["text"])
    try:
        sent = send(os.environ.get("SLACK_WEBHOOK_URL", "").strip() or None, payload)
    except OSError as exc:  # never fail the workflow because Slack is unreachable
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        return 0
    print("Slack: message sent" if sent else "Slack: no webhook configured, skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
