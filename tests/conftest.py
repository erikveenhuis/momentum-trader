import pytest
import torch
import numpy as np
import os
import shutil
from pathlib import Path

# Use absolute imports from src
from momentum_train import RainbowTrainerModule
from momentum_env import TradingEnv
from momentum_train.data import DataManager
from momentum_agent.constants import ACCOUNT_STATE_DIM
from momentum_agent import RainbowDQNAgent


# Constants for tests
WINDOW_SIZE = 10
INITIAL_BALANCE = 10000
TRANSACTION_FEE = 0.001


# Helper function to create dummy CSV
def create_dummy_csv(filepath: Path, rows: int = 20):
    # Simple header matching expected env columns
    header = "timestamp,open,high,low,close,volume\n"
    rows_data = [
        f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n"
        for i in range(rows)
    ]
    csv_content = header + "".join(rows_data)
    filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    filepath.write_text(csv_content)


# Fixtures
@pytest.fixture(scope="session") # Use session scope for factory-based fixture
def tmp_path_factory_session(tmp_path_factory):
    return tmp_path_factory

@pytest.fixture(scope="function") # Function scope ensures clean state for each test
def tmp_path(tmp_path_factory_session):
    # Use numbered=True to avoid potential conflicts if multiple tests run concurrently
    path = tmp_path_factory_session.mktemp("data", numbered=True)
    yield path
    # Optional: Clean up the directory after the test function finishes
    # shutil.rmtree(path)


@pytest.fixture
def default_config(tmp_path):
    # Minimal config for testing trainer logic
    # Paths are relative to the test execution directory
    return {
        "agent": {
            "network": "rainbow_transformer",
            "window_size": WINDOW_SIZE,
            "n_features": 5, # Needs to match mock data
            "num_actions": 7,
            "lr": 1e-4,
            "gamma": 0.99,
            "replay_buffer_size": 1000, # Changed from buffer_size to replay_buffer_size
            "batch_size": 4,
            "target_update_freq": 10,
            "n_steps": 3,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 100,
            "noisy_std": 0.1,
            "seed": 42,
            # Add missing required parameters
            "hidden_dim": 32,
            "grad_clip_norm": 10.0,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 64,
            "transformer_dropout": 0.0,
        },
        "environment": {
            "initial_balance": INITIAL_BALANCE,
            "reward_scale": 200.0,
            "transaction_fee": TRANSACTION_FEE,
        },
        "trainer": {
            "seed": 42,
            "warmup_steps": 10, # Reduced for faster tests
            "update_freq": 4,
            "log_freq": 5,
            "validation_freq": 2,
            "checkpoint_save_freq": 2,
            "reward_window": 2,
            "early_stopping_patience": 3,
            "min_validation_threshold": 0.0,
        },
        "run": {
            "mode": "train",
            "episodes": 5, # Default number of episodes for tests
            "model_dir": str(tmp_path / "models"), # Use tmp_path
            "resume": False,
        },
    }


# Mocking removed - Fixtures mock_agent, mock_data_manager, mock_env, and trainer are removed.
# Tests relying on these fixtures will need to be updated or removed.

@pytest.fixture
def trainer(default_config, tmp_path):
    """Fixture to provide a minimally configured RainbowTrainerModule instance."""
    # Use the existing fixture data instead of creating new directories
    fixtures_dir = Path("tests/fixtures")
    data_manager = DataManager(str(fixtures_dir))

    # Create a minimal valid RainbowDQNAgent instance
    mock_agent = RainbowDQNAgent(
        config=default_config["agent"],
        device="cpu"
    )

    # Create the trainer instance with a single config dict
    trainer_instance = RainbowTrainerModule(
        agent=mock_agent,
        data_manager=data_manager,
        device=torch.device("cpu"), # Use CPU for tests
        config=default_config
    )
    return trainer_instance
