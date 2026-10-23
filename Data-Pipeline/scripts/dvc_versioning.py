"""Version pipeline outputs with DVC.

Airflow produces the outputs declared in ``dvc.yaml``. This step records their
hashes in ``dvc.lock`` (``dvc commit``), uploads them to the DVC remote
(``dvc push``) and reports ``dvc status``. Running ``dvc repro`` outside
Airflow reproduces the same stages from scratch.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from scripts.config import PipelineConfig
from scripts.exceptions import DVCError
from scripts.logging_utils import get_logger

logger = get_logger(__name__)


def dvc_executable() -> str:
    """Path of the ``dvc`` command, or raise when DVC is not installed."""
    exe = shutil.which("dvc")
    if not exe:
        raise DVCError("The 'dvc' command is not installed in this environment")
    return exe


def run_dvc(args: list[str], cwd: Path, timeout: int = 3600, runner: Any = subprocess.run) -> str:
    """Run one DVC command in ``cwd`` and return its output; raise :class:`DVCError` on failure."""
    cmd = [dvc_executable(), *args]
    logger.info("Running: %s", " ".join(["dvc", *args]))
    proc = runner(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise DVCError(f"'dvc {' '.join(args)}' failed (exit {proc.returncode}): {output.strip()[-2000:]}")
    if output.strip():
        logger.info(output.strip()[-2000:])
    return output


def version_outputs(cfg: PipelineConfig, push: bool | None = None, runner: Any = subprocess.run) -> dict[str, Any]:
    """Commit current outputs to the DVC cache, push them and return a summary."""
    cwd = cfg.root
    if not (cwd / "dvc.yaml").is_file():
        raise DVCError(f"No dvc.yaml found in {cwd}")
    run_dvc(["commit", "--force"], cwd, runner=runner)
    pushed = False
    if cfg.dvc_push if push is None else push:
        run_dvc(["push"], cwd, runner=runner)
        pushed = True
    status = run_dvc(["status"], cwd, runner=runner)
    up_to_date = "up to date" in status.lower() or not status.strip()
    return {"committed": True, "pushed": pushed, "status_clean": up_to_date, "status": status.strip()[-500:]}
