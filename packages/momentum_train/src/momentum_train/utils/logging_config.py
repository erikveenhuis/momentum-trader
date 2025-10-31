import logging
import logging.handlers
import sys
from pathlib import Path

# Define logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

def setup_logging(
    log_file_path: Path | None = None,
    root_level: int = logging.INFO,
    level_overrides: dict[str, int] | None = None,
    max_bytes: int = 1 * 1024 * 1024,  # 1MB
    backup_count: int = 10,
    console_level: int = logging.INFO
) -> None:
    """
    Setup global logging configuration that can be used across all Python files.
    
    Args:
        log_file_path: Path to the log file. If None, only console logging is used.
        root_level: Root logger level (default: INFO)
        level_overrides: Dictionary of logger names and their specific levels
        max_bytes: Maximum size of log file before rotation (default: 1MB)
        backup_count: Number of backup files to keep (default: 10)
        console_level: Console handler level (default: INFO)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Define formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(console_level)
    root_logger.addHandler(console_handler)

    # Create File Handler if path is provided
    if log_file_path:
        # Ensure log directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create Rotating File Handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(root_level)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file_path} (max: {max_bytes//(1024*1024)}MB, backups: {backup_count})")

    # Apply specific level overrides
    if level_overrides:
        for name, level in level_overrides.items():
            specific_logger = logging.getLogger(name)
            specific_logger.setLevel(level)
            logging.info(f"Setting logger '{name}' level to {logging.getLevelName(level)}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    This should be used in all Python files to get their logger.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name) 