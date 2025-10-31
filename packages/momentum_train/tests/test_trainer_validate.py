import pytest
import numpy as np
import logging
import json
import math
from pathlib import Path

# Use absolute imports from src
from momentum_train.trainer import RainbowTrainerModule, logger as trainer_logger # Import logger
from momentum_env import TradingEnv
from momentum_train.data import DataManager
from momentum_train.metrics import calculate_episode_score # Correct import
from momentum_agent.constants import ACCOUNT_STATE_DIM # Import necessary constants, Removed INITIAL_BALANCE

# Note: Fixtures (trainer, mock_data_manager, mock_env, etc.) are in conftest.py

# Mocking removed - Many tests below are removed as they relied heavily on patching.

# --- Test should_validate (Kept as it doesn't use mocking) --- #

@pytest.mark.unittest
@pytest.mark.parametrize(
    "episode, validation_freq, expected",
    [
        (0, 5, False), # Episode 1
        (4, 5, True),  # Episode 5
        (5, 5, False), # Episode 6
        (9, 5, True),  # Episode 10
        (0, 1, True),  # Episode 1
        (1, 1, True),  # Episode 2
    ]
)
def test_should_validate(trainer, episode, validation_freq, expected):
    """Test the simplified should_validate logic based on frequency."""
    trainer.validation_freq = validation_freq
    # The recent_performance argument is no longer used
    assert trainer.should_validate(episode, {}) == expected
