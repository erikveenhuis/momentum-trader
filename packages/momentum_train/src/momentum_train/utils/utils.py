import logging
from pathlib import Path
import random
import numpy as np
import torch
import logging.handlers  # Import handlers
import sys  # Import sys for console output
from momentum_train.data import DataManager  # Import from package

logger = logging.getLogger("Utils")  # Use a specific logger or the root logger

# Define logs directory (use absolute path or ensure it's relative to project root)
LOGS_DIR = Path("logs")  # Assuming execution from project root
LOGS_DIR.mkdir(exist_ok=True)

# Define handler variables at the module level
training_log_path = LOGS_DIR / "training.log"
# data_processing_log_path = LOGS_DIR / 'data_processing.log' # Add if needed
# Define other log paths as needed


def setup_global_logging(
    log_file_path, 
    root_level=logging.INFO, 
    level_overrides=None,
    max_bytes=1*1024*1024, # Add max_bytes param (e.g., 1MB)
    backup_count=10         # Add backup_count param
    ):
    """Setup global logging to console and a rotating file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)  # Set the minimum level for the root logger

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
    console_handler.setLevel(logging.INFO)  # Console handler level

    # Create Rotating File Handler (mode='a' is default for rotating)
    # Use RotatingFileHandler instead of FileHandler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path, 
        maxBytes=max_bytes, 
        backupCount=backup_count
        )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(root_level)  # File handler level (usually same as root)

    # Add handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.info(f"Logging setup: Console and rotating file {log_file_path} (max: {max_bytes//(1024*1024)}MB, backups: {backup_count})")

    # Apply specific level overrides
    if level_overrides:
        for name, level in level_overrides.items():
            specific_logger = logging.getLogger(name)
            specific_logger.setLevel(level)
            logging.info(
                f"Setting logger '{name}' level to {logging.getLevelName(level)}"
            )

    # No need to return handlers as they are attached to the root logger


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    assert isinstance(seed, int), f"Seed must be an integer, got {type(seed)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Seeds set to {seed}")
    return seed


def get_random_data_file(data_manager: DataManager) -> Path:
    """Get a random training data file using the provided DataManager."""
    assert isinstance(data_manager, DataManager), "Input must be a DataManager instance"
    try:
        # Get a random training file path from the provided manager
        random_file = data_manager.get_random_training_file()
        assert isinstance(random_file, Path), "DataManager did not return a Path object"
        assert (
            random_file.exists()
        ), f"Random file returned by DataManager does not exist: {random_file}"
        assert random_file.is_file(), f"Random file path is not a file: {random_file}"
        return random_file
    except FileNotFoundError as e:
        # Log the error from DataManager if no files are found
        logger.error(f"Error getting random training file: {e}")
        raise  # Re-raise the exception to stop execution
    except Exception as e:
        logger.error(f"Unexpected error in get_random_data_file: {e}")
        raise
