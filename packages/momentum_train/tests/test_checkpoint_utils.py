"""Tests for checkpoint_utils: find_latest_checkpoint and load_checkpoint."""

import pytest
import torch
from momentum_train.utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint

REQUIRED_KEYS = [
    "episode",
    "total_train_steps",
    "network_state_dict",
    "optimizer_state_dict",
    "best_validation_metric",
    "target_network_state_dict",
    "agent_total_steps",
    "early_stopping_counter",
    "agent_config",
]


def _make_valid_checkpoint() -> dict:
    return {key: 0 for key in REQUIRED_KEYS}


# --- find_latest_checkpoint ---


@pytest.mark.unit
def test_find_latest_checkpoint_by_episode_number(tmp_path):
    for ep in (10, 50, 25):
        (tmp_path / f"checkpoint_trainer_latest_20260401_ep{ep}_reward0.50.pt").write_bytes(b"x")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert "ep50" in result


@pytest.mark.unit
def test_find_latest_checkpoint_falls_back_to_latest_pt(tmp_path):
    (tmp_path / "checkpoint_trainer_latest.pt").write_bytes(b"x")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert result.endswith("checkpoint_trainer_latest.pt")


@pytest.mark.unit
def test_find_latest_checkpoint_falls_back_to_best_pt(tmp_path):
    (tmp_path / "checkpoint_trainer_best.pt").write_bytes(b"x")

    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is not None
    assert result.endswith("checkpoint_trainer_best.pt")


@pytest.mark.unit
def test_find_latest_checkpoint_returns_none_empty_dir(tmp_path):
    result = find_latest_checkpoint(model_dir=str(tmp_path))
    assert result is None


# --- load_checkpoint ---


@pytest.mark.unit
def test_load_checkpoint_success(tmp_path):
    ckpt = _make_valid_checkpoint()
    ckpt["best_validation_metric"] = 0.42
    path = tmp_path / "ckpt.pt"
    torch.save(ckpt, path)

    loaded = load_checkpoint(str(path))
    assert loaded is not None
    assert loaded["best_validation_metric"] == pytest.approx(0.42)
    for key in REQUIRED_KEYS:
        assert key in loaded


@pytest.mark.unit
def test_load_checkpoint_missing_keys(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"episode": 1}, path)

    result = load_checkpoint(str(path))
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_file_not_found(tmp_path):
    result = load_checkpoint(str(tmp_path / "nonexistent.pt"))
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_none_path():
    result = load_checkpoint(None)
    assert result is None


@pytest.mark.unit
def test_load_checkpoint_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a valid torch checkpoint")

    result = load_checkpoint(str(path))
    assert result is None
