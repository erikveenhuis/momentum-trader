"""Per-trade economics for momentum-sniper KPIs.

This module is the canonical implementation of the per-trade metrics described in
the comprehensive logging plan (Tier 2). The agent is a momentum sniper: it spends
most of its time flat (action 0 = 0% allocation) and only opens a position when
a short-lived edge appears. Portfolio-level Sharpe/MDD are misleading for that
pattern because an idle equity curve dilutes any meaningful trade-level signal.
The metrics here operate on a *trade* — a contiguous span of non-flat positions —
so each trade contributes one observation regardless of how long the agent stayed
flat around it.

The functions are intentionally pure: they take a list of step records (one per
environment step) and return immutable dataclasses / dicts. The trainer wires
them into ``_run_single_evaluation_episode`` (Tier 2b) and the offline analyzer
(Tier 5c).

Definitions
-----------
A *trade* opens at step ``t`` when the position transitions from zero to non-zero,
and closes at step ``t' >= t`` when the position first returns to zero. PnL is
measured against the portfolio value at the open of the trade so the result
captures the combined effect of price moves and position sizing — which is what
the agent is actually optimizing.

* ``pnl_pct`` — ``(exit_portfolio_value / entry_portfolio_value - 1) * 100``
* ``mae_pct`` — minimum (most negative) unrealized PnL % during the trade
* ``mfe_pct`` — maximum (most positive) unrealized PnL % during the trade
* ``per_trade_sharpe`` — ``mean(pnl_pct) / std(pnl_pct)``; primary KPI
* ``hit_rate`` — fraction of trades with ``pnl_pct > 0``
* ``expectancy_pct`` — ``mean(pnl_pct)``
* ``profit_factor`` — ``sum(pnl_pct[pnl > 0]) / sum(|pnl_pct[pnl < 0]|)``
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

# Tolerance below which a position is treated as flat. Prevents tiny residual
# fractions (floating-point noise from partial fills / cost deductions) from
# being treated as live trades.
_POSITION_FLAT_EPS = 1e-9


@dataclass
class StepRecord:
    """Per-step view used by :func:`segment_trades`.

    All fields are required. ``was_greedy`` is ``None`` when the action provenance
    is not tracked (e.g. legacy evaluation paths) and the per-trade greedy share
    will be reported as ``nan``.
    """

    step_index: int
    portfolio_value: float
    position: float
    price: float
    action: int
    transaction_cost: float = 0.0
    was_greedy: bool | None = None


@dataclass
class TradeRecord:
    """One closed trade. All percentages are in percent (×100), not decimals."""

    entry_index: int
    exit_index: int
    duration_steps: int
    entry_price: float
    exit_price: float
    entry_portfolio_value: float
    exit_portfolio_value: float
    pnl_pct: float
    pnl_absolute: float
    max_position: float
    mae_pct: float
    mfe_pct: float
    transaction_cost_total: float
    actions_taken: list[int] = field(default_factory=list)
    pct_greedy_actions: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def _to_step_record(item: object) -> StepRecord:
    """Coerce a step description (StepRecord or mapping) into a StepRecord."""
    if isinstance(item, StepRecord):
        return item
    if isinstance(item, Mapping):
        return StepRecord(
            step_index=int(item.get("step_index", item.get("step", 0))),
            portfolio_value=float(item["portfolio_value"]),
            position=float(item["position"]),
            price=float(item["price"]),
            action=int(item["action"]),
            transaction_cost=float(item.get("transaction_cost", 0.0)),
            was_greedy=item.get("was_greedy"),
        )
    raise TypeError(f"Unsupported step record type: {type(item).__name__}")


def segment_trades(steps: Iterable[object]) -> list[TradeRecord]:
    """Segment a per-step trace into a list of closed trades.

    Open positions at the end of the trace are closed at the final step using the
    final portfolio value as the exit (an "unrealized close"). This keeps the
    aggregate count consistent with the visible behavior in the trace.
    """
    step_list: list[StepRecord] = [_to_step_record(s) for s in steps]
    if not step_list:
        return []

    trades: list[TradeRecord] = []
    in_trade = False
    entry: StepRecord | None = None
    span: list[StepRecord] = []

    def _close(end_step: StepRecord) -> None:
        """Close the currently open trade using ``end_step`` as the exit."""
        nonlocal in_trade, entry, span
        assert entry is not None and in_trade
        path_pv = np.array([s.portfolio_value for s in span] + [end_step.portfolio_value], dtype=np.float64)
        unrealized_pct = (path_pv / entry.portfolio_value - 1.0) * 100.0
        mae = float(unrealized_pct.min()) if unrealized_pct.size else 0.0
        mfe = float(unrealized_pct.max()) if unrealized_pct.size else 0.0
        actions = [s.action for s in span] + [end_step.action]
        # was_greedy is optional; ignore None entries when computing the share.
        greedy_flags = [s.was_greedy for s in span if s.was_greedy is not None]
        pct_greedy = float(np.mean(greedy_flags)) if greedy_flags else float("nan")
        positions_in_trade = [s.position for s in span if s.position > _POSITION_FLAT_EPS]
        max_pos = float(max(positions_in_trade)) if positions_in_trade else float(entry.position)
        cost_total = float(sum(s.transaction_cost for s in span)) + float(end_step.transaction_cost)
        pnl_abs = float(end_step.portfolio_value - entry.portfolio_value)
        pnl_pct = (end_step.portfolio_value / entry.portfolio_value - 1.0) * 100.0 if entry.portfolio_value > 0 else 0.0

        trades.append(
            TradeRecord(
                entry_index=entry.step_index,
                exit_index=end_step.step_index,
                duration_steps=int(end_step.step_index - entry.step_index),
                entry_price=float(entry.price),
                exit_price=float(end_step.price),
                entry_portfolio_value=float(entry.portfolio_value),
                exit_portfolio_value=float(end_step.portfolio_value),
                pnl_pct=pnl_pct,
                pnl_absolute=pnl_abs,
                max_position=max_pos,
                mae_pct=mae,
                mfe_pct=mfe,
                transaction_cost_total=cost_total,
                actions_taken=actions,
                pct_greedy_actions=pct_greedy,
            )
        )

        in_trade = False
        entry = None
        span = []

    for step in step_list:
        is_live = step.position > _POSITION_FLAT_EPS
        if not in_trade:
            if is_live:
                in_trade = True
                entry = step
                span = []  # entry step itself is not appended; close() handles it via path
        else:
            if is_live:
                span.append(step)
            else:
                # Position returned to zero — close the trade *at this step*.
                _close(step)

    if in_trade:
        # Trace ended with the trade still open; close at the last in-trade step.
        # If there is no span (single-step trade still open), close at entry.
        end_step = span[-1] if span else entry
        assert end_step is not None
        _close(end_step)

    return trades


def _safe_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=0))


def aggregate_trade_metrics(trades: Iterable[TradeRecord]) -> dict[str, float]:
    """Aggregate a list of trades into the sniper-KPI dictionary.

    Returns a flat dict so it can be unpacked directly into TensorBoard scalars.
    All percent fields are in percent (×100), not decimals.
    """
    trade_list = list(trades)
    n = len(trade_list)
    if n == 0:
        return {
            "trade_count": 0.0,
            "hit_rate": float("nan"),
            "expectancy_pct": float("nan"),
            "per_trade_sharpe": float("nan"),
            "profit_factor": float("nan"),
            "avg_duration_steps": float("nan"),
            "avg_mae_pct": float("nan"),
            "avg_mfe_pct": float("nan"),
            "worst_mae_pct": float("nan"),
            "best_mfe_pct": float("nan"),
            "median_pnl_pct": float("nan"),
            "p25_pnl_pct": float("nan"),
            "p75_pnl_pct": float("nan"),
            "max_pnl_pct": float("nan"),
            "min_pnl_pct": float("nan"),
            "pct_greedy_actions": float("nan"),
            "total_pnl_absolute": 0.0,
            "total_transaction_cost": 0.0,
        }

    pnl_pct = np.array([t.pnl_pct for t in trade_list], dtype=np.float64)
    pnl_abs = np.array([t.pnl_absolute for t in trade_list], dtype=np.float64)
    durations = np.array([t.duration_steps for t in trade_list], dtype=np.float64)
    mae = np.array([t.mae_pct for t in trade_list], dtype=np.float64)
    mfe = np.array([t.mfe_pct for t in trade_list], dtype=np.float64)
    txn_cost = np.array([t.transaction_cost_total for t in trade_list], dtype=np.float64)

    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    pf_denominator = float(np.abs(losses).sum())
    profit_factor = float(wins.sum() / pf_denominator) if pf_denominator > 0 else (float("inf") if wins.size > 0 else float("nan"))

    std = _safe_std(pnl_pct)
    per_trade_sharpe = float(pnl_pct.mean() / std) if std > 1e-12 else float("nan")

    greedy_shares = [t.pct_greedy_actions for t in trade_list if not np.isnan(t.pct_greedy_actions)]
    pct_greedy = float(np.mean(greedy_shares)) if greedy_shares else float("nan")

    return {
        "trade_count": float(n),
        "hit_rate": float((pnl_pct > 0).mean()),
        "expectancy_pct": float(pnl_pct.mean()),
        "per_trade_sharpe": per_trade_sharpe,
        "profit_factor": profit_factor,
        "avg_duration_steps": float(durations.mean()),
        "avg_mae_pct": float(mae.mean()),
        "avg_mfe_pct": float(mfe.mean()),
        "worst_mae_pct": float(mae.min()),
        "best_mfe_pct": float(mfe.max()),
        "median_pnl_pct": float(np.median(pnl_pct)),
        "p25_pnl_pct": float(np.percentile(pnl_pct, 25)),
        "p75_pnl_pct": float(np.percentile(pnl_pct, 75)),
        "max_pnl_pct": float(pnl_pct.max()),
        "min_pnl_pct": float(pnl_pct.min()),
        "pct_greedy_actions": pct_greedy,
        "total_pnl_absolute": float(pnl_abs.sum()),
        "total_transaction_cost": float(txn_cost.sum()),
    }


__all__ = [
    "StepRecord",
    "TradeRecord",
    "segment_trades",
    "aggregate_trade_metrics",
]
