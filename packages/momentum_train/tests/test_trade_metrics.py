"""Tests for the per-trade economics module (Tier 2a)."""

import numpy as np
import pytest
from momentum_train.trade_metrics import (
    StepRecord,
    TradeRecord,
    aggregate_trade_metrics,
    segment_trades,
)


def _step(idx: int, pv: float, pos: float, price: float, action: int, *, greedy: bool | None = None, cost: float = 0.0) -> StepRecord:
    return StepRecord(
        step_index=idx,
        portfolio_value=pv,
        position=pos,
        price=price,
        action=action,
        transaction_cost=cost,
        was_greedy=greedy,
    )


@pytest.mark.unit
def test_segment_trades_returns_empty_for_empty_input():
    assert segment_trades([]) == []


@pytest.mark.unit
def test_segment_trades_no_trades_when_always_flat():
    steps = [_step(i, 1000.0, 0.0, 100.0, 0) for i in range(10)]
    assert segment_trades(steps) == []


@pytest.mark.unit
def test_segment_trades_single_trade_open_close():
    """A simple long trade: open at step 1, close at step 4 with +5% PnL."""
    steps = [
        _step(0, 1000.0, 0.0, 100.0, 0),
        _step(1, 1000.0, 5.0, 100.0, 5, greedy=True),
        _step(2, 1010.0, 5.0, 102.0, 5, greedy=True),
        _step(3, 1020.0, 5.0, 104.0, 5, greedy=True),
        _step(4, 1050.0, 0.0, 105.0, 0, greedy=True),
        _step(5, 1050.0, 0.0, 105.0, 0),
    ]

    trades = segment_trades(steps)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_index == 1
    assert t.exit_index == 4
    assert t.duration_steps == 3
    assert t.entry_portfolio_value == pytest.approx(1000.0)
    assert t.exit_portfolio_value == pytest.approx(1050.0)
    assert t.pnl_pct == pytest.approx(5.0)
    assert t.pnl_absolute == pytest.approx(50.0)
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(105.0)
    assert t.max_position == pytest.approx(5.0)
    assert t.mfe_pct == pytest.approx(5.0)
    # No drawdown during this trade — MAE should be the smallest unrealized PnL.
    assert t.mae_pct == pytest.approx(1.0)
    assert t.pct_greedy_actions == pytest.approx(1.0)


@pytest.mark.unit
def test_segment_trades_captures_mae_and_mfe():
    """A trade that dips below entry then recovers — MAE must be negative."""
    steps = [
        _step(0, 1000.0, 0.0, 100.0, 0),
        _step(1, 1000.0, 4.0, 100.0, 4),
        _step(2, 980.0, 4.0, 98.0, 4),  # -2% unrealized
        _step(3, 970.0, 4.0, 97.0, 4),  # -3% unrealized (MAE)
        _step(4, 1030.0, 4.0, 103.0, 4),  # +3% unrealized (MFE)
        _step(5, 1010.0, 0.0, 101.0, 0),  # Close at +1%
    ]
    trades = segment_trades(steps)
    assert len(trades) == 1
    t = trades[0]
    assert t.mae_pct == pytest.approx(-3.0)
    assert t.mfe_pct == pytest.approx(3.0)
    assert t.pnl_pct == pytest.approx(1.0)


@pytest.mark.unit
def test_segment_trades_multiple_trades_with_flat_periods():
    steps = [
        _step(0, 1000.0, 0.0, 100.0, 0),
        # Trade 1: +2%
        _step(1, 1000.0, 2.0, 100.0, 2),
        _step(2, 1020.0, 0.0, 102.0, 0),
        _step(3, 1020.0, 0.0, 102.0, 0),
        # Trade 2: -1%
        _step(4, 1020.0, 5.0, 102.0, 5),
        _step(5, 1009.8, 0.0, 100.98, 0),
    ]
    trades = segment_trades(steps)
    assert len(trades) == 2
    assert trades[0].pnl_pct == pytest.approx(2.0)
    assert trades[1].pnl_pct == pytest.approx(-1.0, abs=1e-2)


@pytest.mark.unit
def test_segment_trades_open_at_end_is_closed_unrealized():
    """An open position at end of trace should still produce one trade."""
    steps = [
        _step(0, 1000.0, 0.0, 100.0, 0),
        _step(1, 1000.0, 3.0, 100.0, 3),
        _step(2, 1015.0, 3.0, 101.5, 3),
        _step(3, 1030.0, 3.0, 103.0, 3),
    ]
    trades = segment_trades(steps)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_index == 1
    # Closed at last in-trade step (index 3) with portfolio 1030.
    assert t.exit_index == 3
    assert t.exit_portfolio_value == pytest.approx(1030.0)
    assert t.pnl_pct == pytest.approx(3.0)


@pytest.mark.unit
def test_aggregate_trade_metrics_empty_returns_nan_block():
    metrics = aggregate_trade_metrics([])
    assert metrics["trade_count"] == 0.0
    for key in ("hit_rate", "expectancy_pct", "per_trade_sharpe", "profit_factor"):
        assert np.isnan(metrics[key])


@pytest.mark.unit
def test_aggregate_trade_metrics_computes_sniper_kpis():
    # Three deterministic trades: +3%, -1%, +2%.
    trades = [
        TradeRecord(
            entry_index=0,
            exit_index=10,
            duration_steps=10,
            entry_price=100.0,
            exit_price=103.0,
            entry_portfolio_value=1000.0,
            exit_portfolio_value=1030.0,
            pnl_pct=3.0,
            pnl_absolute=30.0,
            max_position=1.0,
            mae_pct=-0.5,
            mfe_pct=4.0,
            transaction_cost_total=0.5,
            actions_taken=[5, 5, 0],
            pct_greedy_actions=1.0,
        ),
        TradeRecord(
            entry_index=20,
            exit_index=25,
            duration_steps=5,
            entry_price=105.0,
            exit_price=103.95,
            entry_portfolio_value=1030.0,
            exit_portfolio_value=1019.7,
            pnl_pct=-1.0,
            pnl_absolute=-10.3,
            max_position=1.0,
            mae_pct=-2.0,
            mfe_pct=0.5,
            transaction_cost_total=0.4,
            actions_taken=[3, 3, 0],
            pct_greedy_actions=0.5,
        ),
        TradeRecord(
            entry_index=30,
            exit_index=42,
            duration_steps=12,
            entry_price=104.0,
            exit_price=106.08,
            entry_portfolio_value=1019.7,
            exit_portfolio_value=1040.0,
            pnl_pct=2.0,
            pnl_absolute=20.3,
            max_position=1.0,
            mae_pct=-0.3,
            mfe_pct=2.5,
            transaction_cost_total=0.6,
            actions_taken=[5, 5, 0],
            pct_greedy_actions=1.0,
        ),
    ]

    metrics = aggregate_trade_metrics(trades)
    assert metrics["trade_count"] == 3.0
    assert metrics["hit_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["expectancy_pct"] == pytest.approx(np.mean([3.0, -1.0, 2.0]))
    assert metrics["per_trade_sharpe"] == pytest.approx(np.mean([3.0, -1.0, 2.0]) / np.std([3.0, -1.0, 2.0], ddof=0))
    assert metrics["profit_factor"] == pytest.approx(5.0 / 1.0)
    assert metrics["avg_mae_pct"] == pytest.approx(np.mean([-0.5, -2.0, -0.3]))
    assert metrics["avg_mfe_pct"] == pytest.approx(np.mean([4.0, 0.5, 2.5]))
    assert metrics["worst_mae_pct"] == pytest.approx(-2.0)
    assert metrics["best_mfe_pct"] == pytest.approx(4.0)
    assert metrics["max_pnl_pct"] == pytest.approx(3.0)
    assert metrics["min_pnl_pct"] == pytest.approx(-1.0)
    assert metrics["total_pnl_absolute"] == pytest.approx(30.0 - 10.3 + 20.3)
    assert metrics["total_transaction_cost"] == pytest.approx(0.5 + 0.4 + 0.6)
    assert metrics["pct_greedy_actions"] == pytest.approx(np.mean([1.0, 0.5, 1.0]))


@pytest.mark.unit
def test_aggregate_trade_metrics_profit_factor_handles_no_losses():
    trades = [
        TradeRecord(
            entry_index=0,
            exit_index=1,
            duration_steps=1,
            entry_price=100.0,
            exit_price=101.0,
            entry_portfolio_value=1000.0,
            exit_portfolio_value=1010.0,
            pnl_pct=1.0,
            pnl_absolute=10.0,
            max_position=1.0,
            mae_pct=0.0,
            mfe_pct=1.0,
            transaction_cost_total=0.0,
        )
    ]
    metrics = aggregate_trade_metrics(trades)
    assert metrics["profit_factor"] == float("inf")


@pytest.mark.unit
def test_segment_trades_accepts_dict_inputs():
    """Tier 5c will feed segmented step parquet rows as plain dicts."""
    steps = [
        {"step_index": 0, "portfolio_value": 1000.0, "position": 0.0, "price": 100.0, "action": 0},
        {"step_index": 1, "portfolio_value": 1000.0, "position": 1.0, "price": 100.0, "action": 5},
        {"step_index": 2, "portfolio_value": 1010.0, "position": 0.0, "price": 101.0, "action": 0},
    ]
    trades = segment_trades(steps)
    assert len(trades) == 1
    assert trades[0].pnl_pct == pytest.approx(1.0)
