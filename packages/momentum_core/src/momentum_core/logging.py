"""Repository-wide logging configuration for momentum trader."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

LOG_DIR_ENV_VAR = "MOMENTUM_LOG_DIR"
GLOBAL_LEVEL_ENV_VAR = "MOMENTUM_LOG_LEVEL"
PACKAGE_LEVEL_ENV_PREFIX = "MOMENTUM_LOG_LEVEL_"

# Define default logs directory relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOGS_DIR = _PROJECT_ROOT / "logs"


def _coerce_level(level: Optional[Union[int, str]]) -> Optional[int]:
    """Convert a string/int log level to the logging module constant."""

    if level is None:
        return None

    if isinstance(level, int):
        return level

    if isinstance(level, str):
        candidate = level.strip().upper()
        if candidate in logging._nameToLevel:  # type: ignore[attr-defined]
            return logging._nameToLevel[candidate]  # type: ignore[index]

        try:
            return int(candidate)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown log level: {level}") from exc

    raise TypeError(f"Unsupported log level type: {type(level)!r}")


def _resolve_logs_dir(logs_dir: Optional[Union[str, Path]]) -> Path:
    """Determine the logs directory, respecting override and environment variables."""

    if logs_dir is not None:
        resolved = Path(logs_dir)
    else:
        env_dir = os.getenv(LOG_DIR_ENV_VAR)
        resolved = Path(env_dir) if env_dir else DEFAULT_LOGS_DIR

    resolved = resolved.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_log_level(package_name: Optional[str], requested_level: Optional[Union[int, str]]) -> int:
    """Resolve the log level using explicit request, package override, and global fallback."""

    requested = _coerce_level(requested_level)

    candidates: list[Optional[str]] = []
    if package_name:
        env_name = PACKAGE_LEVEL_ENV_PREFIX + package_name.upper().replace(".", "_")
        candidates.append(os.getenv(env_name))

    candidates.append(os.getenv(GLOBAL_LEVEL_ENV_VAR))

    for candidate in candidates:
        level = _coerce_level(candidate)
        if level is not None:
            return level

    return requested if requested is not None else logging.INFO


def setup_logging(
    log_file_path: Optional[Union[str, Path]] = None,
    root_level: Union[int, str, None] = logging.INFO,
    level_overrides: Optional[Dict[str, Union[int, str]]] = None,
    max_bytes: int = 1 * 1024 * 1024,  # 1MB
    backup_count: int = 10,
    console_level: Union[int, str, None] = logging.INFO,
    console_stream: Optional[object] = None,
    logs_dir: Optional[Union[str, Path]] = None,
    file_level: Union[int, str, None] = None,
    propagate: bool = False,
) -> None:
    """
    Setup global logging configuration that can be used across all Python files in the momentum trader repo.

    This provides both console and file logging with rotation, supporting multiple log levels.

    Args:
        log_file_path: Path to the log file. If None, only console logging is used.
        root_level: Root logger level (default: INFO)
        level_overrides: Dictionary of logger names and their specific levels
        max_bytes: Maximum size of log file before rotation (default: 1MB)
        backup_count: Number of backup files to keep (default: 10)
        console_level: Console handler level (default: INFO)
        console_stream: Stream for console output (default: sys.stdout)
        logs_dir: Directory for log files (default: repo level 'logs' or env override)
        file_level: Optional level for the file handler (defaults to root level)
        propagate: Whether the root logger should propagate to ancestor loggers
    """

    resolved_root_level = _resolve_log_level(None, root_level)
    resolved_console_level = _coerce_level(console_level) or resolved_root_level
    resolved_file_level = _coerce_level(file_level) or resolved_root_level

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_root_level)
    root_logger.propagate = propagate

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Define formatter with consistent format across the repo
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create Console Handler
    stream = console_stream if console_stream is not None else sys.stdout
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(resolved_console_level)
    root_logger.addHandler(console_handler)

    # Create File Handler if path is provided
    if log_file_path:
        log_dir = _resolve_logs_dir(logs_dir)
        log_path = Path(log_file_path)
        if not log_path.is_absolute():
            log_path = log_dir / log_path

        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(resolved_file_level)
        root_logger.addHandler(file_handler)

        logging.getLogger(__name__).info(
            "Logging to file: %s (max: %sMB per file, %s backup files)",
            log_path,
            max_bytes // (1024 * 1024),
            backup_count,
        )

    # Apply specific level overrides
    if level_overrides:
        for name, level in level_overrides.items():
            specific_logger = logging.getLogger(name)
            specific_logger.setLevel(_coerce_level(level) or resolved_root_level)
            logging.getLogger(__name__).info(
                "Setting logger '%s' level to %s",
                name,
                logging.getLevelName(specific_logger.level),
            )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    This should be used in all Python files across the momentum trader repo to get their logger.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def setup_package_logging(
    package_name: str,
    log_filename: Optional[str] = None,
    root_level: Union[int, str, None] = logging.INFO,
    level_overrides: Optional[Dict[str, Union[int, str]]] = None,
    logs_dir: Optional[Union[str, Path]] = None,
    console_level: Union[int, str, None] = None,
    file_level: Union[int, str, None] = None,
) -> None:
    """
    Convenience function to setup logging for a specific package.

    The effective log level can be controlled via environment variables:

    * ``MOMENTUM_LOG_LEVEL`` – global fallback level
    * ``MOMENTUM_LOG_LEVEL_<PACKAGE>`` – package-specific override (``.`` replaced with ``_``)
    * ``MOMENTUM_LOG_DIR`` – root directory for all log files

    Args:
        package_name: Name of the package (used in log filename if not specified)
        log_filename: Specific log filename. If None, uses {package_name}.log
        root_level: Requested root logger level (overridden by env vars if set)
        level_overrides: Logger-specific level overrides
        logs_dir: Logs directory (default: repo `logs/` or env override)
        console_level: Optional console handler level override
        file_level: Optional file handler level override
    """

    resolved_level = _resolve_log_level(package_name, root_level)
    log_dir = _resolve_logs_dir(logs_dir)

    if log_filename is None:
        log_filename = f"{package_name}.log"

    log_file_path = log_dir / log_filename

    setup_logging(
        log_file_path=log_file_path,
        root_level=resolved_level,
        level_overrides=level_overrides,
        console_level=console_level,
        file_level=file_level,
        logs_dir=log_dir,
    )
