"""Tests for the data processing module."""

import numpy as np
import pandas as pd
import pytest
from momentum_env.data import MarketData, MarketDataProcessor, get_observation_at_step


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
    assert data_processor.FEATURE_NAMES == [
        "norm_open",
        "norm_high",
        "norm_low",
        "norm_close",
        "norm_volume",
    ]


def test_data_processor_validation(data_processor, sample_data):
    """Test data validation."""
    # Valid data should not raise
    data_processor._validate_columns(sample_data)

    # Missing column should raise
    invalid_data = sample_data.drop("open", axis=1)
    with pytest.raises(ValueError, match="Missing required columns"):
        data_processor._validate_columns(invalid_data)

    # Non-numeric column should raise
    invalid_data = sample_data.copy()
    invalid_data["open"] = "invalid"
    with pytest.raises(ValueError, match="must be numeric"):
        data_processor._validate_columns(invalid_data)


def test_data_processor_normalization(data_processor, sample_data):
    """Test data normalization."""
    norm_data = data_processor._normalize_data(sample_data)

    assert isinstance(norm_data, pd.DataFrame)
    assert all(col in norm_data.columns for col in data_processor.FEATURE_NAMES)
    assert norm_data.shape[1] == len(data_processor.FEATURE_NAMES)
    assert (norm_data >= 0).all().all()  # All values should be >= 0
    assert (norm_data <= 1).all().all()  # All values should be <= 1


def test_data_processor_load_and_process(data_processor, sample_data, tmp_path):
    """Test loading and processing data."""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)

    # Process data
    market_data = data_processor.load_and_process_data(str(data_path))

    assert isinstance(market_data, MarketData)
    assert market_data.window_size == data_processor.window_size
    assert market_data.num_features == len(data_processor.FEATURE_NAMES)
    assert market_data.close_prices.dtype == np.float32
    assert market_data.normalized_features.dtype == np.float32


def test_get_observation(data_processor, sample_data, tmp_path):
    """Test getting observations."""
    # Save and process sample data
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path, index=False)
    market_data = data_processor.load_and_process_data(str(data_path))

    # Get observation
    observation = get_observation_at_step(
        market_data=market_data,
        step=3,
        position=1.0,
        balance=5000.0,
        initial_balance=10000.0,
        current_price=105.0,
    )

    assert isinstance(observation, dict)
    assert "market_data" in observation
    assert "account_state" in observation
    assert observation["market_data"].shape == (3, 5)  # window_size x num_features
    assert observation["account_state"].shape == (2,)  # [position, balance]
    assert observation["market_data"].dtype == np.float32
    assert observation["account_state"].dtype == np.float32
