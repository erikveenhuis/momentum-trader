"""Training utilities: seeding and data file helpers."""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from momentum_core.data import DataManager

logger = logging.getLogger("Utils")


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all RNG backends."""
    assert isinstance(seed, int), f"Seed must be an integer, got {type(seed)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Seeds set to {seed}")
    return seed


def get_random_data_file(data_manager: DataManager) -> Path:
    """Get a random training data file using the provided DataManager."""
    assert isinstance(data_manager, DataManager), "Input must be a DataManager instance"
    try:
        random_file = data_manager.get_random_training_file()
        assert isinstance(random_file, Path), "DataManager did not return a Path object"
        assert random_file.exists(), f"Random file returned by DataManager does not exist: {random_file}"
        assert random_file.is_file(), f"Random file path is not a file: {random_file}"
        return random_file
    except FileNotFoundError as e:
        logger.error(f"Error getting random training file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_random_data_file: {e}")
        raise
