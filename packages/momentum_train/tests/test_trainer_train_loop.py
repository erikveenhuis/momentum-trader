# tests/test_trainer_train_loop.py
import pytest
import torch
import numpy as np
import logging
from pathlib import Path
import warnings # Import warnings for non-float reward test
import math # Import math for isinf check

# Use absolute imports from src
from momentum_train.trainer import RainbowTrainerModule, logger as trainer_logger # Keep for patching, import logger
from momentum_env import TradingEnv
from momentum_train.data import DataManager
from momentum_agent.constants import ACCOUNT_STATE_DIM # REMOVED WINDOW_SIZE
from momentum_train.metrics import calculate_episode_score # Corrected import path

# Helper for creating dummy CSV files in tests that need them
def create_dummy_csv(filepath: Path, rows: int = 20):
    header = "timestamp,open,high,low,close,volume\n"
    rows_data = [f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n" for i in range(rows)]
    csv_content = header + "".join(rows_data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(csv_content)

# Note: Fixtures (trainer, mock_agent, mock_data_manager, etc.) are in conftest.py

# --- Mocking removed, tests below are removed --- #