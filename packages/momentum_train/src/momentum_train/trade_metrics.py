"""Re-export of :mod:`momentum_core.trade_metrics` for backward compatibility.

The canonical implementation lives in :mod:`momentum_core.trade_metrics` so it
can also be consumed by ``momentum_live`` (which must not import from
``momentum_train`` per the package boundary DAG). Existing trainer/test code
keeps importing from ``momentum_train.trade_metrics``.
"""

from __future__ import annotations

from momentum_core.trade_metrics import (
    StepRecord,
    TradeRecord,
    aggregate_trade_metrics,
    segment_trades,
)

__all__ = [
    "StepRecord",
    "TradeRecord",
    "segment_trades",
    "aggregate_trade_metrics",
]
