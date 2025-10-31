# TODO: Add tests for src/utils/checkpoint_utils.py

import pytest
import os
import torch
from pathlib import Path

from momentum_train.checkpoint_utils import find_latest_checkpoint, load_checkpoint

# --- Fixtures ---

@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoint files."""
    chkpt_dir = tmp_path / "checkpoints"
    chkpt_dir.mkdir()
    return chkpt_dir

@pytest.fixture
def sample_checkpoint_data():
    """Valid checkpoint data dictionary including agent state."""
    # NOTE: Ensure this structure matches what _save_checkpoint actually saves!
    return {
        "episode": 100,
        "total_train_steps": 10000,
        "best_validation_metric": 0.75,
        "early_stopping_counter": 2,
        # --- ADDED Agent State ---
        "agent_config": {"lr": 0.0001, "gamma": 0.99}, # Example agent config
        "agent_total_steps": 10000, # Should match total_train_steps for consistency
        "network_state_dict": {"layer1.weight": torch.rand(10, 5), "layer2.bias": torch.rand(10)},
        "target_network_state_dict": {"layer1.weight": torch.rand(10, 5), "layer2.bias": torch.rand(10)},
        "optimizer_state_dict": {"state": {}, "param_groups": []}, # Basic optimizer state
        # --- END ADDED Agent State ---
    }

@pytest.fixture
def incomplete_checkpoint_data():
    """Invalid checkpoint data (missing required keys)."""
    return {
        "episode": 50,
        # Missing total_train_steps, network_state_dict, etc.
    }


# --- Tests for find_latest_checkpoint ---

@pytest.mark.unittest
def test_find_latest_checkpoint_exists(checkpoint_dir):
    """Test finding the latest checkpoint when it exists."""
    prefix = "test_model"
    latest_file = checkpoint_dir / f"{prefix}_latest.pt"
    best_file = checkpoint_dir / f"{prefix}_best.pt"

    latest_file.touch()  # Create the file
    best_file.touch()

    found_path = find_latest_checkpoint(str(checkpoint_dir), prefix)
    assert found_path == str(latest_file)

@pytest.mark.unittest
def test_find_best_checkpoint_when_latest_missing(checkpoint_dir):
    """Test finding the best checkpoint when latest is missing."""
    prefix = "test_model_alt"
    # latest_file = checkpoint_dir / f"{prefix}_latest.pt" # Don't create latest
    best_file = checkpoint_dir / f"{prefix}_best.pt"

    best_file.touch()

    found_path = find_latest_checkpoint(str(checkpoint_dir), prefix)
    assert found_path == str(best_file)

@pytest.mark.unittest
def test_find_no_checkpoint_if_none_exist(checkpoint_dir):
    """Test returning None when no suitable checkpoints exist."""
    prefix = "non_existent_model"
    found_path = find_latest_checkpoint(str(checkpoint_dir), prefix)
    assert found_path is None


# --- Tests for load_checkpoint ---

@pytest.mark.unittest
def test_load_checkpoint_success(checkpoint_dir, sample_checkpoint_data):
    """Test successfully loading a valid checkpoint file."""
    checkpoint_path = checkpoint_dir / "valid_checkpoint.pt"
    torch.save(sample_checkpoint_data, checkpoint_path)

    loaded_data = load_checkpoint(str(checkpoint_path))
    assert loaded_data is not None
    assert loaded_data["episode"] == sample_checkpoint_data["episode"]
    assert "network_state_dict" in loaded_data
    # Check if tensors are loaded correctly (optional, simple check)
    assert torch.is_tensor(loaded_data["network_state_dict"]["layer1.weight"])

@pytest.mark.unittest
def test_load_checkpoint_non_existent_file(checkpoint_dir):
    """Test loading a checkpoint from a non-existent path."""
    non_existent_path = checkpoint_dir / "non_existent.pt"
    loaded_data = load_checkpoint(str(non_existent_path))
    assert loaded_data is None

@pytest.mark.unittest
def test_load_checkpoint_missing_keys(checkpoint_dir, incomplete_checkpoint_data):
    """Test loading a checkpoint file with missing required keys."""
    invalid_path = checkpoint_dir / "invalid_checkpoint.pt"
    torch.save(incomplete_checkpoint_data, invalid_path)

    loaded_data = load_checkpoint(str(invalid_path))
    assert loaded_data is None # Should fail validation

@pytest.mark.unittest
def test_load_checkpoint_empty_path():
    """Test load_checkpoint with an empty path string."""
    loaded_data = load_checkpoint("")
    assert loaded_data is None

@pytest.mark.unittest
def test_load_checkpoint_none_path():
    """Test load_checkpoint with a None path."""
    loaded_data = load_checkpoint(None)
    assert loaded_data is None
