"""Derived feature computation shared by preprocessing and CSV-fallback loading.

Thin wrapper around :func:`momentum_core.features.compute_derived_features_np`
so the canonical implementation lives in :mod:`momentum_core` (where both
training and live trading can delegate to it).
"""

import numpy as np
import pandas as pd
from momentum_core.constants import ROLLING_WINDOW  # re-exported for legacy callers
from momentum_core.features import compute_derived_features_np

__all__ = ["ROLLING_WINDOW", "compute_derived_features"]


def compute_derived_features(df: pd.DataFrame) -> np.ndarray:
    """Compute derived features from raw OHLCV+transactions DataFrame.

    Returns float32 array of shape [T, 6] with columns:
      log_return_1, log_return_5, log_return_10,
      realized_vol, volume_ratio, hl_range_ratio.
    """
    return compute_derived_features_np(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        volume=df["volume"].values,
    )
