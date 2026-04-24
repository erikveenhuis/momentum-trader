"""Shared dimensional constants used across :mod:`momentum_env`,
:mod:`momentum_agent`, :mod:`momentum_train`, and :mod:`momentum_live`.

Every size here is a *shape contract* -- changing any of them breaks the
network, the buffer layout, and serialized checkpoints simultaneously, so
they need a single canonical home.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Account-state feature vector
# ---------------------------------------------------------------------------
# [position_frac, cash_frac, unrealized_pnl, bars_in_position / 60,
#  cumulative_fees / initial_balance]
ACCOUNT_STATE_DIM: int = 5

# ---------------------------------------------------------------------------
# Derived market-feature rolling window
# ---------------------------------------------------------------------------
# Rolling window (in bars) used to compute realized_vol, volume_ratio,
# hl_range_ratio. Training env and live trader MUST agree on this value;
# diverging here is a silent distribution-shift bug.
ROLLING_WINDOW: int = 20

# ---------------------------------------------------------------------------
# Canonical market-feature layout
# ---------------------------------------------------------------------------
# 6 raw OHLCV + transactions columns, then 6 derived features. The order is
# load-bearing -- it dictates which column the agent's aux return head reads
# via ``agent.aux_target_feature_index``.
RAW_FEATURE_NAMES: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "transactions",
)

DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "realized_vol",
    "volume_ratio",
    "hl_range_ratio",
)

FEATURE_NAMES: tuple[str, ...] = RAW_FEATURE_NAMES + DERIVED_FEATURE_NAMES

N_RAW_FEATURES: int = len(RAW_FEATURE_NAMES)
N_DERIVED_FEATURES: int = len(DERIVED_FEATURE_NAMES)
N_FEATURES: int = len(FEATURE_NAMES)


__all__ = [
    "ACCOUNT_STATE_DIM",
    "DERIVED_FEATURE_NAMES",
    "FEATURE_NAMES",
    "N_DERIVED_FEATURES",
    "N_FEATURES",
    "N_RAW_FEATURES",
    "RAW_FEATURE_NAMES",
    "ROLLING_WINDOW",
]
