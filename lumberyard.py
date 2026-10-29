"""
Logging setup utils for the ringforge.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_DIR: Path = Path("logs")
DEFAULT_CONSOLE_LEVEL: int = logging.INFO
DEFAULT_FILE_LEVEL: int = logging.DEBUG


class _ConsoleFormatter(logging.Formatter):
    """Message-only console output; prefix WARNING and above"""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def _unique_log_path(log_dir: Path, name: str) -> Path:
    """Build a log path unique to this process and microsecond"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = name.replace(".", "_")
    stem = f"{safe_name}_{stamp}_{os.getpid()}"
    candidate = log_dir / f"{stem}.log"
    counter = 1
    while candidate.exists():
        candidate = log_dir / f"{stem}_{counter}.log"
        counter += 1
    return candidate


def setup_logging(
    name: str = "ringforge",
    *,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_file: str | Path | None = None,
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    file_level: int = DEFAULT_FILE_LEVEL,
) -> tuple[logging.Logger, Path]:
    """
    Setup and return `(logger, log_path)`

    Parameters
        name
            Logger name
        log_dir
            Log dir
        log_file
            Explicit log path
            Auto builds from timestamp at default
        console_level
            Minimum level printed to the terminal
        file_level
            Minimum level written to the log file

    Existing handlers on `name` are closed and replaced
    """
    log_dir = Path(log_dir)
    path = Path(log_file) if log_file is not None else _unique_log_path(log_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(_ConsoleFormatter("%(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info("log file: %s", path)
    return logger, path


def configure_worker_logging(
    log_file: str | Path,
    *,
    name: str = "ringforge.test",
) -> None:
    """
    Attach a worker process to the same log file
    """
    setup_logging(
        name,
        log_file=log_file,
        console_level=logging.WARNING,
        file_level=logging.DEBUG,
    )


if __name__ == "__main__":
    LOGGER, LOG_PATH = setup_logging("ringforge.lumberyard")
    LOGGER.debug("debug message")
    LOGGER.info("info message")
    LOGGER.warning("warning message")
    print(f"wrote {LOG_PATH}")
