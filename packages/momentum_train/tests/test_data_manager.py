import random
from pathlib import Path

import pytest
from momentum_train.data import DataManager


def _write_csv(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / name
    file_path.write_text("timestamp,open,high,low,close,volume\n" "2024-01-01T00:00:00Z,100,101,99,100,10\n")
    return file_path


@pytest.mark.unittest
def test_data_manager_organize_and_random_file(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    train_file = _write_csv(processed_dir / "train", "2024-01-01_BTC-USD.csv")
    val_file = _write_csv(processed_dir / "validation", "2024-06-01_ETH-USD.csv")
    test_file = _write_csv(processed_dir / "test", "2025-01-01_BTC-USD.csv")

    dm = DataManager(base_dir=tmp_path, processed_dir_name="processed")
    dm.organize_data()

    assert dm.get_training_files() == [train_file]
    assert dm.get_validation_files() == [val_file]
    assert dm.get_test_files() == [test_file]

    # Ensure random selection returns the only training file
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    assert dm.get_random_training_file() == train_file


@pytest.mark.unittest
def test_data_manager_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataManager(base_dir=tmp_path / "nonexistent", processed_dir_name="processed")
