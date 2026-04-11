"""Data handling for the trading environment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_RAW_FEATURES = 6
FEATURE_NAMES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "transactions",
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "realized_vol",
    "volume_ratio",
    "hl_range_ratio",
]


@dataclass
class MarketData:
    """Container for market data."""

    close_prices: np.ndarray
    features: np.ndarray
    feature_names: list[str]

    data_length: int
    window_size: int

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


class MarketDataProcessor:
    """Process market data for the trading environment."""

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    def load_and_process_data(self, data_path: str) -> MarketData:
        """Load market data from .npz (preferred) or CSV (fallback)."""
        if data_path.endswith(".npz"):
            return self._load_npz(data_path)
        return self._load_csv(data_path)

    def _load_npz(self, data_path: str) -> MarketData:
        """Load preprocessed .npz file."""
        data = np.load(data_path)
        close_prices = data["close_prices"].astype(np.float32)
        features = data["features"].astype(np.float32)

        return MarketData(
            close_prices=close_prices,
            features=features,
            feature_names=FEATURE_NAMES[: features.shape[1]],
            data_length=len(close_prices),
            window_size=self.window_size,
        )

    def _load_csv(self, data_path: str) -> MarketData:
        """Fallback: load raw CSV, compute features on the fly."""
        data_df = pd.read_csv(data_path).dropna()
        self._validate_columns(data_df)

        close_prices = data_df["close"].values.astype(np.float32)

        if "transactions" not in data_df.columns:
            data_df["transactions"] = 0.0

        raw_cols = ["open", "high", "low", "close", "volume", "transactions"]
        raw = data_df[raw_cols].values.astype(np.float32)

        from momentum_env.features import compute_derived_features

        derived = compute_derived_features(data_df)
        features = np.concatenate([raw, derived], axis=1).astype(np.float32)

        return MarketData(
            close_prices=close_prices,
            features=features,
            feature_names=FEATURE_NAMES[: features.shape[1]],
            data_length=len(close_prices),
            window_size=self.window_size,
        )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        for col in self.REQUIRED_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' must be numeric")


def get_observation_at_step(
    market_data: MarketData,
    step: int,
    position: float,
    balance: float,
    initial_balance: float,
    current_price: float,
    position_price: float,
    bars_in_position: int,
    cumulative_fees_frac: float,
) -> dict[str, np.ndarray]:
    """Get the observation at a specific step.

    Raw features (columns 0:N_RAW_FEATURES) receive window-level z-score
    normalization so all timesteps share a consistent reference frame.
    Derived features are already approximately stationary and are passed
    through with light clipping.

    Account state is a 5-D vector:
      [0] position_value / portfolio_value
      [1] cash / portfolio_value
      [2] unrealized PnL fraction: (price - cost_basis) / cost_basis
      [3] bars_in_position / 60 (normalized time-in-position)
      [4] cumulative_fees / initial_balance
    """
    observation_step = min(step, market_data.data_length - 1)
    start_index = max(0, observation_step - market_data.window_size + 1)
    market_window = market_data.features[start_index : observation_step + 1].copy()

    if len(market_window) < market_data.window_size:
        padding_shape = (market_data.window_size - len(market_window), market_data.num_features)
        padding = np.zeros(padding_shape, dtype=market_window.dtype)
        market_window = np.vstack((padding, market_window))

    n_raw = min(N_RAW_FEATURES, market_window.shape[1])
    raw_part = market_window[:, :n_raw]
    w_mean = raw_part.mean(axis=0, keepdims=True)
    w_std = raw_part.std(axis=0, keepdims=True) + 1e-8
    market_window[:, :n_raw] = (raw_part - w_mean) / w_std

    if market_window.shape[1] > n_raw:
        derived_part = market_window[:, n_raw:]
        np.clip(derived_part, -10.0, 10.0, out=derived_part)

    position_value = max(0.0, position * current_price)
    portfolio_value = max(0.0, balance + position * current_price)
    safe_portfolio_value = max(portfolio_value, 1e-9)

    normalized_position = position_value / safe_portfolio_value
    normalized_balance = balance / safe_portfolio_value

    unrealized_pnl = 0.0
    if position > 1e-9 and position_price > 1e-9:
        unrealized_pnl = (current_price - position_price) / position_price

    safe_initial = max(initial_balance, 1e-9)

    account_state = np.array(
        [
            np.clip(normalized_position, 0.0, 1.0),
            np.clip(normalized_balance, 0.0, 1.0),
            np.clip(unrealized_pnl, -1.0, 1.0),
            np.clip(bars_in_position / 60.0, 0.0, 1.0),
            np.clip(cumulative_fees_frac / safe_initial, 0.0, 1.0),
        ],
        dtype=np.float32,
    )

    return {
        "market_data": market_window.astype(np.float32),
        "account_state": account_state,
    }
