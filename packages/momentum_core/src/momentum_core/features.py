"""Canonical derived-feature computation shared by training and live.

One implementation, two callers: :mod:`momentum_env.features` (preprocess/CSV
fallback) and :mod:`momentum_live.trader` (streaming). Keeping them in sync
matters -- even a tiny numerical divergence z-scores into distribution shift
and invisibly degrades live performance vs. backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ROLLING_WINDOW


def compute_derived_features_np(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Compute the six derived market features from raw OHLCV arrays.

    Returns a ``float32`` array of shape ``[T, 6]`` with columns:
    ``log_return_1, log_return_5, log_return_10, realized_vol,
    volume_ratio, hl_range_ratio``.

    All inputs must be 1-D arrays of equal length ``T``. Non-finite outputs
    are sanitised to ``0.0`` (NaN) / ``±10.0`` (inf) so downstream clipping
    still operates on finite numbers.
    """
    close_f = np.asarray(close, dtype=np.float64)
    high_f = np.asarray(high, dtype=np.float64)
    low_f = np.asarray(low, dtype=np.float64)
    volume_f = np.asarray(volume, dtype=np.float64)
    T = len(close_f)

    log_ret_1 = np.zeros(T, dtype=np.float64)
    log_ret_5 = np.zeros(T, dtype=np.float64)
    log_ret_10 = np.zeros(T, dtype=np.float64)

    safe_close = np.where(close_f > 0, close_f, 1e-20)
    if T > 1:
        log_ret_1[1:] = np.log(safe_close[1:] / safe_close[:-1])
    if T > 5:
        log_ret_5[5:] = np.log(safe_close[5:] / safe_close[:-5])
    if T > 10:
        log_ret_10[10:] = np.log(safe_close[10:] / safe_close[:-10])

    realized_vol = pd.Series(log_ret_1).rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0).values

    vol_roll_mean = pd.Series(volume_f).rolling(window=ROLLING_WINDOW, min_periods=1).mean().values
    volume_ratio = volume_f / np.where(vol_roll_mean > 1e-20, vol_roll_mean, 1e-20)

    hl_range = (high_f - low_f) / np.where(safe_close > 0, safe_close, 1e-20)
    hl_roll_mean = pd.Series(hl_range).rolling(window=ROLLING_WINDOW, min_periods=1).mean().values
    hl_range_ratio = hl_range / np.where(hl_roll_mean > 1e-20, hl_roll_mean, 1e-20)

    derived = np.column_stack([log_ret_1, log_ret_5, log_ret_10, realized_vol, volume_ratio, hl_range_ratio])
    np.nan_to_num(derived, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    return derived.astype(np.float32)


__all__ = ["compute_derived_features_np"]
