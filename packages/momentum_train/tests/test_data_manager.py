import random
from pathlib import Path

import numpy as np
import pytest
from momentum_train.data import DataManager


def _write_csv(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / name
    file_path.write_text("timestamp,open,high,low,close,volume\n" "2024-01-01T00:00:00Z,100,101,99,100,10\n")
    return file_path


def _write_npz(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / name
    close_prices = np.array([100.0, 101.0], dtype=np.float32)
    features = np.random.randn(2, 12).astype(np.float32)
    np.savez_compressed(file_path, close_prices=close_prices, features=features)
    return file_path


@pytest.mark.unit
def test_data_manager_prefers_npz(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    _write_csv(processed_dir / "train", "2024-01-01_BTC-USD.csv")
    npz_file = _write_npz(processed_dir / "train", "2024-01-01_BTC-USD.npz")
    _write_npz(processed_dir / "validation", "2024-06-01_ETH-USD.npz")
    _write_npz(processed_dir / "test", "2025-01-01_BTC-USD.npz")

    dm = DataManager(base_dir=tmp_path, processed_dir_name="processed")
    dm.organize_data()

    assert dm.get_training_files() == [npz_file]
    assert all(f.suffix == ".npz" for f in dm.get_validation_files())


@pytest.mark.unit
def test_data_manager_csv_fallback(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    csv_file = _write_csv(processed_dir / "train", "2024-01-01_BTC-USD.csv")
    _write_csv(processed_dir / "validation", "2024-06-01_ETH-USD.csv")
    _write_csv(processed_dir / "test", "2025-01-01_BTC-USD.csv")

    dm = DataManager(base_dir=tmp_path, processed_dir_name="processed")
    dm.organize_data()

    assert dm.get_training_files() == [csv_file]


@pytest.mark.unit
def test_data_manager_curriculum_frac(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    for i in range(10):
        _write_npz(processed_dir / "train", f"2024-01-{i+1:02d}_BTC-USD.npz")
    _write_npz(processed_dir / "validation", "2024-06-01_ETH-USD.npz")
    _write_npz(processed_dir / "test", "2025-01-01_BTC-USD.npz")

    dm = DataManager(base_dir=tmp_path, processed_dir_name="processed")
    dm.organize_data()

    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    result = dm.get_random_training_file(curriculum_frac=0.3)
    assert result in dm.get_training_files()[:3]


@pytest.mark.unit
def test_data_manager_missing_directories(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataManager(base_dir=tmp_path / "nonexistent", processed_dir_name="processed")
