"""Gantt charts and per-task durations for Airflow DAG runs.

Task-instance start/end times are read from the Airflow REST API and plotted
with matplotlib (one bar per task, relative to the start of the run). A
screenshot of the Airflow UI Gantt view can also be taken with Playwright when
it is installed. Used to document the before/after pipeline optimisation.

Example::

    python -m scripts.gantt --dag-id vibespace_data_pipeline --run-id <run_id> \\
        --out docs/gantt_after.png --json docs/gantt_after.json
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from scripts.io_utils import write_json

BAR_COLOR = "#2a78d6"
FAILED_COLOR = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _parse_time(value: str | None) -> datetime | None:
    """Parse an Airflow ISO timestamp."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def fetch_task_instances(base_url: str, dag_id: str, run_id: str, auth: tuple[str, str],
                         session: Any = requests) -> list[dict[str, Any]]:
    """Task instances (id, state, start, end, duration) of one DAG run."""
    url = f"{base_url.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances"
    resp = session.get(url, auth=auth, params={"limit": 100}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Airflow API returned HTTP {resp.status_code} for {url}")
    rows = []
    for ti in resp.json()["task_instances"]:
        rows.append({"task_id": ti["task_id"], "state": ti["state"], "start_date": ti["start_date"],
                     "end_date": ti["end_date"], "duration": ti.get("duration"), "try_number": ti.get("try_number")})
    return rows


def summarise(task_instances: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall-clock time of the run and duration per task (seconds)."""
    timed = [t for t in task_instances if t["start_date"] and t["end_date"]]
    if not timed:
        return {"wall_seconds": 0.0, "sum_task_seconds": 0.0, "tasks": {}}
    start = min(_parse_time(t["start_date"]) for t in timed)
    end = max(_parse_time(t["end_date"]) for t in timed)
    durations = {t["task_id"]: round((_parse_time(t["end_date"]) - _parse_time(t["start_date"])).total_seconds(), 2)
                 for t in timed}
    return {"wall_seconds": round((end - start).total_seconds(), 2),
            "sum_task_seconds": round(sum(durations.values()), 2),
            "tasks": dict(sorted(durations.items(), key=lambda kv: -kv[1])),
            "states": {t["task_id"]: t["state"] for t in task_instances}}


def plot_gantt(task_instances: list[dict[str, Any]], title: str, out: Path) -> Path:
    """Draw a Gantt chart (bars relative to the run start) and save it as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timed = sorted([t for t in task_instances if t["start_date"] and t["end_date"]],
                   key=lambda t: _parse_time(t["start_date"]))
    if not timed:
        raise ValueError("No finished task instances to plot")
    t0 = _parse_time(timed[0]["start_date"])
    wall = summarise(timed)["wall_seconds"]

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(timed) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, t in enumerate(timed):
        start = (_parse_time(t["start_date"]) - t0).total_seconds()
        dur = (_parse_time(t["end_date"]) - _parse_time(t["start_date"])).total_seconds()
        color = BAR_COLOR if t["state"] == "success" else FAILED_COLOR
        ax.barh(i, max(dur, wall * 0.003), left=start, height=0.55, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(start + dur + wall * 0.01, i, f"{dur:.1f}s", va="center", ha="left", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(range(len(timed)), [t["task_id"] for t in timed], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("seconds since run start", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, wall * 1.12 if wall else 1)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.set_title(f"{title}  (wall time {wall:.0f}s)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def screenshot_ui_gantt(base_url: str, dag_id: str, run_id: str, auth: tuple[str, str], out: Path) -> bool:
    """Screenshot the Airflow UI Gantt view with Playwright; False when unavailable."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(f"{base_url}/login/")
            page.fill("input#username", auth[0])
            page.fill("input#password", auth[1])
            page.click("input[type=submit]")
            page.goto(f"{base_url}/dags/{dag_id}/grid?dag_run_id={requests.utils.quote(run_id)}&tab=gantt")
            page.wait_for_timeout(5000)
            page.screenshot(path=str(out), full_page=True)
            browser.close()
        return True
    except Exception as exc:  # noqa: BLE001 - optional evidence, matplotlib chart is the fallback
        print(f"Playwright screenshot failed: {exc}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot an Airflow DAG run as a Gantt chart.")
    parser.add_argument("--dag-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True, help="PNG path for the matplotlib chart")
    parser.add_argument("--json", help="Where to save task durations")
    parser.add_argument("--title", default=None)
    parser.add_argument("--ui-screenshot", help="Optional PNG path for a Playwright UI screenshot")
    parser.add_argument("--base-url", default=os.environ.get("AIRFLOW_URL", "http://localhost:8080"))
    args = parser.parse_args(argv)
    auth = (os.environ.get("_AIRFLOW_WWW_USER_USERNAME", "airflow"),
            os.environ.get("_AIRFLOW_WWW_USER_PASSWORD", "airflow"))
    tis = fetch_task_instances(args.base_url, args.dag_id, args.run_id, auth)
    plot_gantt(tis, args.title or args.dag_id, Path(args.out))
    summary = summarise(tis) | {"dag_id": args.dag_id, "run_id": args.run_id}
    if args.json:
        write_json(Path(args.json), summary)
    if args.ui_screenshot:
        ok = screenshot_ui_gantt(args.base_url, args.dag_id, args.run_id, auth, Path(args.ui_screenshot))
        print(f"UI screenshot: {'saved' if ok else 'not available'}")
    print(f"wall={summary['wall_seconds']}s sum_tasks={summary['sum_task_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
