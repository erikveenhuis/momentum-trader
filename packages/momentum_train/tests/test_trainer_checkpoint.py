import pytest
import torch
import logging
import numpy as np
from pathlib import Path

# Use absolute imports from src
from momentum_train.trainer import RainbowTrainerModule # Keep for patching
from momentum_env import TradingEnv
from momentum_agent.constants import ACCOUNT_STATE_DIM
from momentum_train.metrics import calculate_episode_score # Needed for test_train_save_best_checkpoint_on_validation


# Note: Fixtures (trainer, mock_agent, mock_data_manager, etc.) are in conftest.py

# --- Mocking removed, tests relying on patch("torch.save") are removed --- #

