"""Tests for the data processing module."""

import numpy as np
import pandas as pd
import pytest
from momentum_env.data import FEATURE_NAMES, MarketData, MarketDataProcessor, get_observation_at_step


@pytest.fixture
def sample_data():
    """Create sample market data."""
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
            "transactions": [50, 60, 70, 80, 90],
        }
    )
    return data


@pytest.fixture
def data_processor():
    """Create a data processor instance."""
    return MarketDataProcessor(window_size=3)


def test_data_processor_initialization(data_processor):
    """Test data processor initialization."""
    assert data_processor.window_size == 3
    assert data_processor.REQUIRED_COLUMNS == ["open", "high", "low", "close", "volume"]


def test_data_processor_validation(data_processor, sample_data):
    """Test data validation."""
    data_processor._validate_columns(sample_data)

    invalid_data = sample_data.drop("open", axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        data_processor._validate_columns(invalid_data)

    invalid_data = sample_data.copy()
    invalid_data["open"] = "invalid"
    with pytest.raises(ValueError, match="must be numeric"):
        data_processor._validate_columns(invalid_data)


def test_data_processor_load_csv(data_processor, sample_data, tmp_path):
    """Test loading and processing CSV data."""
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)

    market_data = data_processor.load_and_process_data(str(data_path))

    assert isinstance(market_data, MarketData)
    assert market_data.window_size == data_processor.window_size
    assert market_data.num_features == len(FEATURE_NAMES)
    assert market_data.close_prices.dtype == np.float32
    assert market_data.features.dtype == np.float32
    assert market_data.features.shape == (5, len(FEATURE_NAMES))


def test_data_processor_load_npz(data_processor, sample_data, tmp_path):
    """Test loading preprocessed .npz data."""
    close_prices = sample_data["close"].values.astype(np.float32)
    features = np.random.randn(5, len(FEATURE_NAMES)).astype(np.float32)
    npz_path = tmp_path / "test_data.npz"
    np.savez_compressed(npz_path, close_prices=close_prices, features=features)

    market_data = data_processor.load_and_process_data(str(npz_path))

    assert isinstance(market_data, MarketData)
    assert market_data.data_length == 5
    assert market_data.num_features == len(FEATURE_NAMES)
    assert market_data.close_prices.dtype == np.float32
    assert market_data.features.dtype == np.float32


def test_get_observation(data_processor, tmp_path):
    """Test getting observations with window-level z-score normalization."""
    close_prices = np.array([100, 101, 102, 103, 104], dtype=np.float32)
    features = np.random.randn(5, len(FEATURE_NAMES)).astype(np.float32)
    npz_path = tmp_path / "test_data.npz"
    np.savez_compressed(npz_path, close_prices=close_prices, features=features)

    market_data = data_processor.load_and_process_data(str(npz_path))

    observation = get_observation_at_step(
        market_data=market_data,
        step=3,
        position=1.0,
        balance=5000.0,
        initial_balance=10000.0,
        current_price=105.0,
        position_price=100.0,
        bars_in_position=5,
        cumulative_fees_frac=0.5,
    )

    assert isinstance(observation, dict)
    assert "market_data" in observation
    assert "account_state" in observation
    assert observation["market_data"].shape == (3, len(FEATURE_NAMES))
    assert observation["account_state"].shape == (5,)
    assert observation["market_data"].dtype == np.float32
    assert observation["account_state"].dtype == np.float32


def test_get_observation_padding(data_processor, tmp_path):
    """Test that early observations are zero-padded correctly."""
    close_prices = np.array([100, 101, 102, 103, 104], dtype=np.float32)
    features = np.ones((5, len(FEATURE_NAMES)), dtype=np.float32)
    npz_path = tmp_path / "test_data.npz"
    np.savez_compressed(npz_path, close_prices=close_prices, features=features)

    market_data = data_processor.load_and_process_data(str(npz_path))

    observation = get_observation_at_step(
        market_data=market_data,
        step=0,
        position=0.0,
        balance=10000.0,
        initial_balance=10000.0,
        current_price=100.0,
        position_price=0.0,
        bars_in_position=0,
        cumulative_fees_frac=0.0,
    )

    assert observation["market_data"].shape == (3, len(FEATURE_NAMES))
    derived_cols = slice(6, None)
    assert np.all(observation["market_data"][:2, derived_cols] == 0.0)
