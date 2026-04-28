"""
Simple centralized logging for Mesofield with ANSI colors and aligned columns.

Just import and use:
    from mesofield.logging_config import get_logger
    logger = get_logger("MyClass")
    logger.info("Hello world")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from logging.handlers import TimedRotatingFileHandler

# Simple global configuration
_configured = False
_log_dir = None
# make everything from these modules WARNING+ only
for lib in ("matplotlib", "asyncio", "traitlets"):
    logging.getLogger(lib).setLevel(logging.WARNING)
    
# ANSI colorful codes bc its pretty
_LEVEL_COLORS = {
    'DEBUG':    '\033[94m',  # bright blue
    'INFO':     '\033[92m',  # bright green
    'WARNING':  '\033[93m',  # bright yellow
    'ERROR':    '\033[91m',  # bright red
    'CRITICAL': '\033[95m',  # bright magenta
}
_RESET = '\033[0m'

# fixed-width columns: name=15 chars (truncated if longer), level=8 chars
_LOG_FMT = "%(asctime)s | %(levelname)-8s | [%(name)s] --> %(message)s"
_DATE_FMT = "%H:%M:%S"


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # inject color into levelname without affecting other handlers
        lvl = record.levelname
        color = _LEVEL_COLORS.get(lvl, _RESET)
        record.levelname = f"{color}{lvl}{_RESET}"
        formatted = super().format(record)
        record.levelname = lvl
        return formatted

_log_dir: Optional[Path] = None


def install_excepthook() -> None:
    """Log uncaught exceptions to the mesofield log file."""

    def _handle(exc_type, exc_value, exc_traceback):
        logger = logging.getLogger("mesofield")
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        if _log_dir is not None:
            log_file = _log_dir / "mesofield.log"
            print(f"Uncaught exception. See {log_file} for details.")

    sys.excepthook = _handle

def setup_logging(log_dir: Optional[str] = None, level: str = "INFO") -> None:
    global _configured, _log_dir
    if _configured:
        return

    if log_dir:
        _log_dir = Path(log_dir)
    else:
        # assume this file lives in <project>/mesofield/_logger.py
        project_root = Path(__file__).resolve().parent.parent
        _log_dir = project_root / "logs"

    _log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, level.upper(), logging.INFO)

    # console handler (colored)
    formatter = ColoredFormatter(fmt=_LOG_FMT, datefmt=_DATE_FMT)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(console)

    # daily rotating file handler
    log_file = _log_dir / "mesofield.log"
    # rotate at midnight, keep 7 days (nothing good happens after midnight)
    fh = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        utc=False
    )
    fh.suffix = "%Y%m%d"
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(fmt=_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(file_formatter)

    root.addHandler(fh)

    # Suppress noisy third-party loggers
    logging.getLogger("pymmcore-plus").setLevel(logging.WARNING)

    install_excepthook()

    _configured = True


import functools

def log_this_fr(func):
    """
    Decorator that logs entry, exit (and exceptions) of the function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Entering {func.__qualname__} args={args!r}, kwargs={kwargs!r}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__qualname__} returned {result!r}")
            return result
        except Exception:
            logger.exception(f"Exception in {func.__qualname__}")
            raise
    return wrapper


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup_logging()
    return logging.getLogger(name)