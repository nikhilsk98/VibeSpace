"""Logging setup shared by every pipeline module.

Messages go to ``logs/pipeline.log`` (rotating) and, when nothing else is
configured (plain CLI runs), to the console. Inside Airflow the records also
propagate to Airflow's task log handler, so they appear in the task logs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "vibespace"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_HANDLER_ATTR = "_vibespace_file_handler"


def setup_logging(logs_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure the ``vibespace`` logger once and return it.

    Args:
        logs_dir: Folder for ``pipeline.log``. ``None`` disables file logging.
        level: Minimum level for the pipeline logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    if logs_dir is not None:
        log_path = (Path(logs_dir) / "pipeline.log").resolve()
        current = getattr(logger, _FILE_HANDLER_ATTR, None)
        if current is None or Path(current.baseFilename) != log_path:
            if current is not None:
                logger.removeHandler(current)
                current.close()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            setattr(logger, _FILE_HANDLER_ATTR, handler)

    root_has_handlers = bool(logging.getLogger().handlers)
    has_console = any(getattr(h, "_vibespace_console", False) for h in logger.handlers)
    if not root_has_handlers and not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._vibespace_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)
        logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger such as ``vibespace.clean``."""
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{LOGGER_NAME}.{short}")


@contextmanager
def log_step(logger: logging.Logger, step: str) -> Iterator[None]:
    """Log the start, end and duration of a step, and log failures loudly."""
    start = time.perf_counter()
    logger.info("START %s", step)
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.1fs", step, time.perf_counter() - start)
        raise
    logger.info("DONE %s in %.1fs", step, time.perf_counter() - start)
