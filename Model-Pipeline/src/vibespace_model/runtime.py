"""Process-level setup that must happen before LightGBM or PyTorch is imported.

On macOS the pip wheels of PyTorch, scikit-learn and LightGBM each ship or link their own copy of the
OpenMP runtime (``libomp.dylib``). Two copies in one process deadlock or crash as soon as both start
threads. :func:`ensure_single_openmp` re-executes the current command once with ``DYLD_LIBRARY_PATH``
pointing at PyTorch's ``lib`` folder, so every library loads that one ``libomp``. On Linux (and in the
Docker image) all three use the same ``libgomp`` and nothing happens.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_MARKER = "VIBESPACE_SINGLE_OPENMP"


def torch_lib_dir() -> Path | None:
    """PyTorch's bundled library folder, found without importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    lib = Path(spec.origin).parent / "lib"
    return lib if (lib / "libomp.dylib").is_file() else None


def ensure_single_openmp() -> None:
    """Re-exec once on macOS so all libraries share PyTorch's ``libomp`` (no-op elsewhere)."""
    if sys.platform != "darwin" or os.environ.get(_MARKER) == "1":
        return
    lib = torch_lib_dir()
    if lib is None:  # pragma: no cover - torch always ships libomp on macOS
        return
    if "lightgbm" in sys.modules or "torch" in sys.modules:  # pragma: no cover - too late to fix in-process
        return
    env = dict(os.environ)
    env[_MARKER] = "1"
    current = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{lib}{os.pathsep}{current}" if current else str(lib)
    argv = list(getattr(sys, "orig_argv", [sys.executable, *sys.argv]))
    os.execve(sys.executable, [sys.executable, *argv[1:]], env)  # noqa: S606  # pragma: no cover
