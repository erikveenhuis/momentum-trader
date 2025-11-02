"""Data handling for the trading environment."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """Container for market data."""

    # Raw data
    close_prices: np.ndarray
    normalized_features: np.ndarray
    feature_names: List[str]

    # Metadata
    data_length: int
    window_size: int

    @property
    def num_features(self) -> int:
        """Get the number of features."""
        return len(self.feature_names)


class MarketDataProcessor:
    """Process market data for the trading environment."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
    FEATURE_NAMES = [f"norm_{col}" for col in REQUIRED_COLUMNS]

    def __init__(self, window_size: int) -> None:
        """Initialize the data processor.

        Args:
            window_size: Size of the rolling window for normalization
        """
        self.window_size = window_size

    def load_and_process_data(self, data_path: str) -> MarketData:
        """Load and process market data from a CSV file.

        Args:
            data_path: Path to the CSV file containing market data

        Returns:
            Processed market data

        Raises:
            ValueError: If the data is invalid or missing required columns
        """
        # Load data
        data_df = pd.read_csv(data_path).dropna()

        # Validate columns
        self._validate_columns(data_df)

        # Extract close prices
        close_prices = data_df["close"].values.astype(np.float32)

        # Normalize features
        norm_data_df = self._normalize_data(data_df)
        normalized_features = norm_data_df[self.FEATURE_NAMES].values.astype(np.float32)

        return MarketData(
            close_prices=close_prices,
            normalized_features=normalized_features,
            feature_names=self.FEATURE_NAMES,
            data_length=len(normalized_features),
            window_size=self.window_size,
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that the dataframe has the required columns."""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for col in self.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric")

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV data using min-max scaling with forward-fill for initial window."""
        result_df = pd.DataFrame()

        for col in self.REQUIRED_COLUMNS:
            rolling_min = df[col].rolling(window=self.window_size, min_periods=1).min()
            rolling_max = df[col].rolling(window=self.window_size, min_periods=1).max()
            result_df[f"norm_{col}"] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

        if len(result_df) == 0:
            raise ValueError("No data remaining after normalization")

        return result_df


def get_observation_at_step(
    market_data: MarketData,
    step: int,
    position: float,
    balance: float,
    initial_balance: float,
    current_price: float,
) -> Dict[str, np.ndarray]:
    """Get the observation at a specific step.

    Args:
        market_data: Market data container
        step: Current step
        position: Current position size
        balance: Current balance
        initial_balance: Initial balance for normalization
        current_price: Current price for position value calculation

    Returns:
        Dictionary containing market data and account state arrays
    """
    # Get market data window
    observation_step = min(step, market_data.data_length - 1)
    start_index = max(0, observation_step - market_data.window_size + 1)
    market_window = market_data.normalized_features[start_index : observation_step + 1]

    # Pad if necessary
    if len(market_window) < market_data.window_size:
        padding_shape = (market_data.window_size - len(market_window), market_data.num_features)
        padding = np.zeros(padding_shape, dtype=market_window.dtype)
        market_window = np.vstack((padding, market_window))

    # Calculate portfolio value components and normalize account state as fractions
    position_value = max(0.0, position * current_price)
    portfolio_value = max(0.0, balance + position * current_price)
    safe_portfolio_value = max(portfolio_value, 1e-9)

    normalized_position = position_value / safe_portfolio_value
    normalized_balance = balance / safe_portfolio_value

    account_state = np.clip(
        np.array([normalized_position, normalized_balance], dtype=np.float32),
        0.0,
        1.0,
    )

    return {
        "market_data": market_window,
        "account_state": account_state,
    }
