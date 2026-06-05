"""
Centralized logging for Mesofield using loguru.

Usage:
    from mesofield.utils._logger import get_logger
    logger = get_logger("MyClass")
    logger.info("Hello world")
"""

import functools
import logging
import sys
from os import PathLike
from pathlib import Path
from typing import Optional

from loguru import logger

_configured = False
_log_dir: Optional[Path] = None

# Default extra so the format string always resolves {extra[logger_name]}
logger.configure(extra={"logger_name": "mesofield"})

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "[{extra[logger_name]}] <cyan>{file.name}:{line}</cyan> --> {message}"
)
_FILE_FORMAT = "{time:HH:mm:ss} | {level: <8} | [{extra[logger_name]}] {file.name}:{line} --> {message}"


class InterceptHandler(logging.Handler):
    """Route stdlib logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 0
        while frame.f_code.co_filename in (logging.__file__, __file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def install_excepthook() -> None:
    """Log uncaught exceptions through loguru with full traceback diagnostics."""

    def _handle(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception"
        )
        if _log_dir is not None:
            print(f"Uncaught exception logged to {_log_dir / 'mesofield.log'}")

    sys.excepthook = _handle


def setup_logging(log_dir: Optional[str] = None, level: str = "INFO") -> None:
    global _configured, _log_dir
    if _configured:
        return

    if log_dir:
        _log_dir = Path(log_dir)
    else:
        project_root = Path(__file__).resolve().parent.parent
        _log_dir = project_root / "logs"

    _log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level.upper(),
        colorize=True,
        diagnose=True,
    )

    logger.add(
        _log_dir / "mesofield.log",
        format=_FILE_FORMAT,
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        diagnose=True,
        backtrace=True,
    )

    # Route all stdlib logging through loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Suppress noisy third-party stdlib loggers before they reach loguru
    for lib in ("matplotlib", "asyncio", "traitlets", "pymmcore_plus", "pymmcore-plus",
                "ipykernel"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    install_excepthook()
    _configured = True


def log_this_fr(func):
    """Decorator that logs entry, exit, and exceptions of the function."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _log = get_logger(func.__module__)
        _log.debug(f"Entering {func.__qualname__} args={args!r}, kwargs={kwargs!r}")
        try:
            result = func(*args, **kwargs)
            _log.debug(f"Exiting {func.__qualname__} returned {result!r}")
            return result
        except Exception:
            _log.exception(f"Exception in {func.__qualname__}")
            raise

    return wrapper


def get_logger(name: str):
    if not _configured:
        setup_logging()
    return logger.bind(logger_name=name)


def hyperlink(path: str | PathLike[str], text: str) -> str:
    """Return an OSC-8 terminal hyperlink with custom display text.

    Args:
        path: Local filesystem path or URI target.
        text: Link label shown in the log message.
    """
    target = str(path)
    if not target:
        return text

    try:
        if target.startswith(("file://", "http://", "https://")):
            uri = target
        else:
            uri = Path(target).expanduser().resolve(strict=False).as_uri()
        return f"\033]8;;{uri}\033\\{text}\033]8;;\033\\"
    except Exception:
        return text
