"""Derived feature computation shared by preprocessing and CSV-fallback loading."""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 20


def compute_derived_features(df: pd.DataFrame) -> np.ndarray:
    """Compute derived features from raw OHLCV+transactions DataFrame.

    Returns float32 array of shape [T, 6] with columns:
      log_return_1, log_return_5, log_return_10,
      realized_vol, volume_ratio, hl_range_ratio.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    T = len(close)

    log_ret_1 = np.zeros(T, dtype=np.float64)
    log_ret_5 = np.zeros(T, dtype=np.float64)
    log_ret_10 = np.zeros(T, dtype=np.float64)

    safe_close = np.where(close > 0, close, 1e-20)
    log_ret_1[1:] = np.log(safe_close[1:] / safe_close[:-1])
    log_ret_5[5:] = np.log(safe_close[5:] / safe_close[:-5])
    log_ret_10[10:] = np.log(safe_close[10:] / safe_close[:-10])

    realized_vol = pd.Series(log_ret_1).rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0).values

    vol_roll_mean = pd.Series(volume).rolling(window=ROLLING_WINDOW, min_periods=1).mean().values
    volume_ratio = volume / np.where(vol_roll_mean > 1e-20, vol_roll_mean, 1e-20)

    hl_range = (high - low) / np.where(safe_close > 0, safe_close, 1e-20)
    hl_roll_mean = pd.Series(hl_range).rolling(window=ROLLING_WINDOW, min_periods=1).mean().values
    hl_range_ratio = hl_range / np.where(hl_roll_mean > 1e-20, hl_roll_mean, 1e-20)

    derived = np.column_stack([log_ret_1, log_ret_5, log_ret_10, realized_vol, volume_ratio, hl_range_ratio])

    np.nan_to_num(derived, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    return derived.astype(np.float32)
