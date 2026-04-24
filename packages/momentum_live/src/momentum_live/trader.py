"""Single-pair live trading orchestration for the momentum agent.

One ``MomentumLiveTrader`` instance owns exactly one symbol + one
``BrokerSubAccountClient`` (one Broker sub-account). Multi-pair fan-out is the
job of :class:`momentum_live.multi_pair_runner.MultiPairRunner`, which builds N
of these traders sharing one ``RainbowDQNAgent`` and one ``CryptoDataStream``.

The agent was trained as a single-asset allocator (one symbol, one cash bucket
per episode), so each trader's observation is computed from its own sub-account
cash + its own position only. There is no shared cash pool here: that bug
(BTC firing 100% drains the pool, ETH then sees cash=0) cannot happen because
each pair has its own balance in Alpaca.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from momentum_agent import RainbowDQNAgent
from momentum_core.constants import ROLLING_WINDOW
from momentum_core.features import compute_derived_features_np
from momentum_core.logging import get_logger
from momentum_core.trade_metrics import (
    StepRecord,
    aggregate_trade_metrics,
    segment_trades,
)
from momentum_env.data import N_RAW_FEATURES
from momentum_env.trading import PortfolioState, TradingLogic

from .config import LiveTradingConfig
from .subaccount_client import BrokerSubAccountClient

if TYPE_CHECKING:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter

LOGGER = get_logger(__name__)

RAW_COLS = ("open", "high", "low", "close", "volume")
N_DERIVED = 6
N_TOTAL = N_RAW_FEATURES + N_DERIVED


@dataclass(slots=True)
class BarData:
    """Normalized representation of an incoming bar."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime

    @classmethod
    def from_alpaca(cls, bar: object) -> BarData:
        """Create ``BarData`` from an ``alpaca-py`` bar object."""

        return cls(
            symbol=bar.symbol,
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(getattr(bar, "volume", 0.0)),
            timestamp=getattr(bar, "timestamp", datetime.now(UTC)),
        )


class LiveFeatureNormalizer:
    """Maintain a sliding window of raw + derived features with window-level z-score."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._raw: deque[np.ndarray] = deque(maxlen=window_size + ROLLING_WINDOW)
        self._close_history: deque[float] = deque(maxlen=window_size + ROLLING_WINDOW)
        self._volume_history: deque[float] = deque(maxlen=window_size + ROLLING_WINDOW)

    @property
    def count(self) -> int:
        return len(self._raw)

    def update(self, bar: BarData) -> bool:
        row = np.array([bar.open, bar.high, bar.low, bar.close, bar.volume, 0.0], dtype=np.float32)
        self._raw.append(row)
        self._close_history.append(bar.close)
        self._volume_history.append(bar.volume)
        return self.count >= self.window_size

    def window(self) -> np.ndarray:
        raw_arr = np.array(list(self._raw), dtype=np.float32)

        # Delegate to the canonical numpy implementation so training and
        # live always see byte-identical derived features given the same
        # OHLCV inputs. (Tier 4.2)
        derived = compute_derived_features_np(
            close=np.array(list(self._close_history), dtype=np.float64),
            high=raw_arr[:, 1].astype(np.float64),
            low=raw_arr[:, 2].astype(np.float64),
            volume=np.array(list(self._volume_history), dtype=np.float64),
        )
        full = np.concatenate([raw_arr, derived], axis=1)

        window = full[-self.window_size :]
        result = np.zeros((self.window_size, N_TOTAL), dtype=np.float32)
        result[-len(window) :] = window

        raw_part = result[:, :N_RAW_FEATURES].copy()
        w_mean = raw_part.mean(axis=0, keepdims=True)
        w_std = raw_part.std(axis=0, keepdims=True) + 1e-8
        result[:, :N_RAW_FEATURES] = (raw_part - w_mean) / w_std
        np.clip(result[:, N_RAW_FEATURES:], -10.0, 10.0, out=result[:, N_RAW_FEATURES:])

        return result


class MomentumLiveTrader:
    """Glue logic between one symbol's bar stream and the agent."""

    TARGET_ALLOCATIONS = {
        0: 0.0,
        1: 0.2,
        2: 0.4,
        3: 0.6,
        4: 0.8,
        5: 1.0,
    }

    def __init__(
        self,
        agent: RainbowDQNAgent,
        symbol: str,
        config: LiveTradingConfig,
        *,
        subaccount_client: BrokerSubAccountClient | None = None,
        writer: SummaryWriter | None = None,
        trades_jsonl_path: Path | None = None,
    ):
        self.agent = agent
        self.symbol = symbol
        self.config = config
        self.subaccount_client: BrokerSubAccountClient | None = subaccount_client
        self.writer = writer if writer is not None else self._maybe_build_writer(config.tb_log_dir)
        self._global_step: int = 0
        self._action_counts = np.zeros(int(getattr(agent, "num_actions", 6)), dtype=np.int64)
        self._trades_jsonl_path: Path | None = trades_jsonl_path
        if self._trades_jsonl_path is None and self.writer is not None and config.tb_log_dir:
            self._trades_jsonl_path = Path(config.tb_log_dir) / "live_trades.jsonl"
        if self._trades_jsonl_path is not None:
            self._trades_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.trading_logic = TradingLogic(
            transaction_fee=config.transaction_fee,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
            drawdown_penalty_lambda=config.drawdown_penalty_lambda,
            slippage_bps=config.slippage_bps,
            opportunity_cost_lambda=config.opportunity_cost_lambda,
            min_trade_value=config.min_trade_value,
            benchmark_allocation_frac=config.benchmark_allocation_frac,
        )

        self.normalizer = LiveFeatureNormalizer(config.window_size)
        # Local position bookkeeping. When ``subaccount_client`` is set, cash is
        # always read from Alpaca per bar and ``portfolio_state.balance`` is
        # informational. Without a sub-account (tests / dry-run), the local
        # balance is the source of truth and starts at ``initial_balance``.
        self.portfolio_state = PortfolioState(
            balance=float(config.initial_balance),
            position=0.0,
            position_price=0.0,
            total_transaction_cost=0.0,
        )
        self.last_price: float = 0.0
        self.bars_in_position: int = 0
        self.step_count: int = 0
        self.step_records: list[StepRecord] = []
        self.closed_trade_count: int = 0
        self.prev_portfolio_value: float = config.initial_balance

        self.min_rebalance_pct = config.min_rebalance_pct

    def set_subaccount_client(self, client: BrokerSubAccountClient) -> None:
        """Attach the per-sub-account trading shim once available."""
        self.subaccount_client = client
        LOGGER.info("[%s] sub-account client attached (account_id=%s)", self.symbol, client.account_id)

    def get_account_cash(self) -> float:
        """Read cash from the sub-account; fall back to local bookkeeping if no client yet."""
        if self.subaccount_client is None:
            return float(self.portfolio_state.balance)
        try:
            return self.subaccount_client.get_account_cash()
        except Exception as exc:
            LOGGER.warning("[%s] failed to read sub-account cash: %s", self.symbol, exc)
            return float(self.portfolio_state.balance)

    def preload_history(self, bars: Iterable[BarData]) -> None:
        """Seed normalizer with historical bars before live streaming."""
        ordered = sorted(bars, key=lambda bar: bar.timestamp)
        for bar in ordered:
            if bar.symbol != self.symbol:
                continue
            self.last_price = float(bar.close)
            self.normalizer.update(bar)
        LOGGER.info(
            "[%s] warmup status | window_count=%d/%d",
            self.symbol,
            self.normalizer.count,
            self.normalizer.window_size,
        )

    def process_bar(self, bar: BarData) -> dict[str, object] | None:
        """Full end-to-end live step: observe → decide → rebalance → record.

        Decomposed into three helpers so each phase can be unit-tested and
        profiled independently:

        * :meth:`_decide_action`   - update normalizer, query agent
        * :meth:`_execute_rebalance` - translate target allocation into an order
        * :meth:`_record_step`      - update bookkeeping, emit TB scalars, build
          the decision dict that callers log/forward.
        """
        if bar.symbol != self.symbol:
            LOGGER.debug("[%s] ignoring bar for %s", self.symbol, bar.symbol)
            return None

        LOGGER.info(
            "[%s] bar received | ts=%s | o=%.2f h=%.2f l=%.2f c=%.2f v=%.6f",
            self.symbol,
            bar.timestamp,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )

        cash_pre = self.get_account_cash()
        position_pre = self.portfolio_state.position
        current_price = float(bar.close)
        portfolio_pre = cash_pre + max(0.0, position_pre * current_price)

        self.last_price = current_price

        decision_inputs = self._decide_action(bar, cash_pre)
        if decision_inputs is None:
            return None
        action_index, was_greedy, target_frac = decision_inputs

        is_valid, order_result = self._execute_rebalance(
            bar=bar,
            action_index=action_index,
            target_frac=target_frac,
            cash_pre=cash_pre,
            position_pre=position_pre,
            current_price=current_price,
        )

        return self._record_step(
            bar=bar,
            action_index=action_index,
            was_greedy=was_greedy,
            target_frac=target_frac,
            current_price=current_price,
            cash_pre=cash_pre,
            position_pre=position_pre,
            portfolio_pre=portfolio_pre,
            is_valid=is_valid,
            order_result=order_result,
        )

    def _decide_action(
        self,
        bar: BarData,
        cash_pre: float,
    ) -> tuple[int, bool, float] | None:
        """Advance the rolling window and ask the agent for an action.

        Returns ``None`` while the normalizer is still warming up; otherwise
        returns ``(action_index, was_greedy, target_allocation_frac)`` and
        emits the per-step action/Q TB scalars as a side-effect.
        """
        ready = self.normalizer.update(bar)
        if not ready:
            LOGGER.debug(
                "[%s] warming up window: %d/%d bars",
                self.symbol,
                self.normalizer.count,
                self.config.window_size,
            )
            return None

        observation = self._observation(cash_pre)

        select_with_provenance = getattr(self.agent, "select_action_with_provenance", None)
        if callable(select_with_provenance):
            action_index, was_greedy = select_with_provenance(observation)
        else:
            action_index = self.agent.select_action(observation)
            was_greedy = True
        target_frac = self.TARGET_ALLOCATIONS[action_index]
        live_q_values = getattr(self.agent, "_last_select_q_values", None)
        self._emit_action_and_q_scalars(action_index, live_q_values)

        LOGGER.info(
            "[%s] action selected | ts=%s | action=%d (%.0f%%) | price=%.2f | cash=%.2f | position=%.6f",
            self.symbol,
            bar.timestamp,
            action_index,
            target_frac * 100,
            float(bar.close),
            cash_pre,
            self.portfolio_state.position,
        )
        return action_index, was_greedy, target_frac

    def _execute_rebalance(
        self,
        *,
        bar: BarData,
        action_index: int,
        target_frac: float,
        cash_pre: float,
        position_pre: float,
        current_price: float,
    ) -> tuple[bool, dict[str, object] | None]:
        """Translate ``target_frac`` into a buy/sell order (or a no-op).

        Returns ``(is_valid, order_result)``. ``is_valid`` is ``False`` only
        when a broker order was attempted and rejected; no-op rebalances
        (below ``min_rebalance_pct``) still count as valid.
        """
        portfolio_value = cash_pre + max(0.0, position_pre * current_price)
        position_value = max(0.0, position_pre * current_price)
        current_frac = position_value / max(portfolio_value, 1e-9)
        delta_frac = target_frac - current_frac
        delta_value = target_frac * portfolio_value - position_value

        is_valid = True
        order_result: dict[str, object] | None = None

        if abs(delta_frac) < self.min_rebalance_pct:
            LOGGER.debug(
                "[%s] delta below min rebalance (%.2f%% < %.2f%%)",
                self.symbol,
                abs(delta_frac) * 100,
                self.min_rebalance_pct * 100,
            )
            return is_valid, order_result

        if delta_value > 0:
            buy_cash = min(delta_value, cash_pre)
            if buy_cash / max(portfolio_value, 1e-9) < self.min_rebalance_pct:
                LOGGER.debug("[%s] buy cash below min notional, skipping", self.symbol)
            elif self.subaccount_client is not None:
                fraction = buy_cash / max(cash_pre, 1e-9)
                order_result = self._execute_order("buy", fraction, current_price)
                if order_result:
                    self._apply_order_to_local_state(order_result)
                else:
                    is_valid = False
            else:
                temp = PortfolioState(
                    balance=cash_pre,
                    position=position_pre,
                    position_price=self.portfolio_state.position_price,
                    total_transaction_cost=self.portfolio_state.total_transaction_cost,
                )
                fraction = buy_cash / max(cash_pre, 1e-9)
                is_valid, new_ps = self.trading_logic.handle_buy(temp, current_price, fraction)
                if is_valid:
                    self.portfolio_state = new_ps
        else:
            sell_value = abs(delta_value)
            if position_value < 1e-9 or sell_value / max(portfolio_value, 1e-9) < self.min_rebalance_pct:
                LOGGER.debug("[%s] sell value below min notional or no position, skipping", self.symbol)
            elif self.subaccount_client is not None:
                sell_fraction = min(sell_value / position_value, 1.0)
                order_result = self._execute_order("sell", sell_fraction, current_price)
                if order_result:
                    self._apply_order_to_local_state(order_result)
                else:
                    is_valid = False
            else:
                temp = PortfolioState(
                    balance=cash_pre,
                    position=position_pre,
                    position_price=self.portfolio_state.position_price,
                    total_transaction_cost=self.portfolio_state.total_transaction_cost,
                )
                sell_fraction = min(sell_value / position_value, 1.0)
                is_valid, new_ps = self.trading_logic.handle_sell(temp, current_price, sell_fraction)
                if is_valid:
                    self.portfolio_state = new_ps

        return is_valid, order_result

    def _record_step(
        self,
        *,
        bar: BarData,
        action_index: int,
        was_greedy: bool,
        target_frac: float,
        current_price: float,
        cash_pre: float,
        position_pre: float,
        portfolio_pre: float,
        is_valid: bool,
        order_result: dict[str, object] | None,
    ) -> dict[str, object]:
        """Finalise bookkeeping after the rebalance phase and emit the decision dict."""
        if self.portfolio_state.position > 1e-9:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        cash_post = self.get_account_cash()
        position_post = self.portfolio_state.position
        portfolio_post = cash_post + max(0.0, position_post * current_price)
        self.prev_portfolio_value = portfolio_post

        self._emit_step_state(action_index, was_greedy, portfolio_post)
        self._maybe_emit_trade_close(position_pre)
        self._global_step += 1

        decision: dict[str, object] = {
            "symbol": self.symbol,
            "timestamp": bar.timestamp,
            "action_index": action_index,
            "target_allocation": target_frac,
            "price": current_price,
            "portfolio_value": portfolio_post,
            "cash": cash_post,
            "valid": is_valid,
            "order": order_result,
        }

        order_info = ""
        if order_result:
            order_info = (
                " | order_id="
                f"{order_result.get('order_id')} status={order_result.get('status')}"
                f" qty={order_result.get('quantity', 0.0):.6f}"
                f" filled={order_result.get('filled_quantity', 0.0):.6f}"
            )

        action_label = f"target_{int(target_frac * 100)}pct"
        LOGGER.info(
            "[%s] decision summary | ts=%s | action=%s | price=%.2f | cash=%.2f->%.2f | position=%.6f->%.6f | "
            "portfolio=%.2f->%.2f | valid=%s%s",
            self.symbol,
            bar.timestamp,
            action_label,
            current_price,
            cash_pre,
            cash_post,
            position_pre,
            position_post,
            portfolio_pre,
            portfolio_post,
            is_valid,
            order_info,
        )
        return decision

    def _observation(self, cash: float) -> dict[str, np.ndarray]:
        return {
            "market_data": self.normalizer.window(),
            "account_state": self._account_state(cash),
        }

    def _account_state(self, cash: float) -> np.ndarray:
        price = max(self.last_price, 1e-9)
        position_value = max(0.0, self.portfolio_state.position * price)
        balance = max(0.0, cash)
        portfolio_value = max(balance + position_value, 1e-9)

        normalized_position = float(np.clip(position_value / portfolio_value, 0.0, 1.0))
        normalized_balance = float(np.clip(balance / portfolio_value, 0.0, 1.0))

        unrealized_pnl = 0.0
        if self.portfolio_state.position > 1e-9 and self.portfolio_state.position_price > 1e-9:
            unrealized_pnl = (price - self.portfolio_state.position_price) / self.portfolio_state.position_price

        # Match training's feature[4] exactly: cumulative_fees are normalized
        # against the constant initial balance, not the current portfolio value.
        # Using the live portfolio value here would silently shift the feature
        # distribution vs. what the agent saw during training (see
        # `get_observation_at_step` in momentum_env/data.py).
        safe_initial = max(float(self.config.initial_balance), 1e-9)

        return np.array(
            [
                normalized_position,
                normalized_balance,
                np.clip(unrealized_pnl, -1.0, 1.0),
                np.clip(self.bars_in_position / 60.0, 0.0, 1.0),
                np.clip(self.portfolio_state.total_transaction_cost / safe_initial, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _execute_order(self, action: str, fraction: float, price: float) -> dict[str, object] | None:
        """Execute a real order on the sub-account and return order details."""
        client = self.subaccount_client
        if client is None:
            return None

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            account = client.get_account()
            available_cash = float(account.cash)

            LOGGER.info(
                "[%s] order sizing | action=%s | fraction=%.2f | price=%.2f | cash=%.2f",
                self.symbol,
                action,
                fraction,
                price,
                available_cash,
            )

            if action == "buy":
                if available_cash <= 0:
                    LOGGER.warning("[%s] zero spendable cash; skipping buy", self.symbol)
                    return None
                max_spend = available_cash * fraction
                if max_spend < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "[%s] order notional too small for buy: %.2f < min_trade_value=%.2f",
                        self.symbol,
                        max_spend,
                        self.trading_logic.min_trade_value,
                    )
                    return None
                quantity = max_spend / price
            else:
                positions = client.get_all_positions()
                current_position = 0.0
                sanitized = self.symbol.replace("/", "")
                for pos in positions:
                    pos_symbol = getattr(pos, "symbol", "")
                    if pos_symbol in (self.symbol, sanitized):
                        current_position = float(pos.qty)
                        break
                quantity = current_position * fraction
                if quantity < 0.00001:
                    LOGGER.warning(
                        "[%s] insufficient position for sell: %.6f * %.2f = %.6f",
                        self.symbol,
                        current_position,
                        fraction,
                        quantity,
                    )
                    return None
                notional = quantity * price
                if notional < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "[%s] order notional too small for sell: %.2f < min_trade_value=%.2f",
                        self.symbol,
                        notional,
                        self.trading_logic.min_trade_value,
                    )
                    return None

            if action == "buy":
                notional = quantity * price
                if notional < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "[%s] order notional too small for buy after sizing: %.2f < %.2f",
                        self.symbol,
                        notional,
                        self.trading_logic.min_trade_value,
                    )
                    return None

            side = OrderSide.BUY if action == "buy" else OrderSide.SELL
            trade_symbol = self.symbol.replace("/", "") if "/" in self.symbol else self.symbol
            order_request = MarketOrderRequest(
                symbol=trade_symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,
            )

            LOGGER.info(
                "[%s] submitting %s | qty=%.6f | notional~=%.2f",
                self.symbol,
                action.upper(),
                quantity,
                quantity * price,
            )
            order = client.submit_order(order_request)
            LOGGER.info(
                "[%s] order submitted | %s",
                self.symbol,
                self._order_debug_snapshot(order),
            )

            completed_order = self._wait_for_order_completion(order.id)
            if completed_order is not None:
                order = completed_order
                LOGGER.info(
                    "[%s] order completion | %s",
                    self.symbol,
                    self._order_debug_snapshot(order),
                )

            return self._build_order_result(
                order=order,
                trade_symbol=trade_symbol,
                side=action,
                requested_quantity=quantity,
                reference_price=price,
            )

        except Exception as exc:
            LOGGER.error("[%s] failed to execute %s order: %s", self.symbol, action, exc)
            return None

    def _apply_order_to_local_state(self, order_result: dict[str, object]) -> None:
        """Update local position bookkeeping from an executed order's fill."""
        side = order_result.get("side")
        filled_qty = float(order_result.get("filled_quantity", 0.0) or 0.0)
        filled_price = float(order_result.get("filled_avg_price", 0.0) or 0.0)
        if filled_qty <= 0.0 or filled_price <= 0.0:
            return

        fee = filled_qty * filled_price * self.config.transaction_fee
        if side == "buy":
            current_position = self.portfolio_state.position
            current_cost = current_position * self.portfolio_state.position_price
            new_position = current_position + filled_qty
            new_position_price = (current_cost + filled_qty * filled_price) / new_position if new_position > 0 else 0.0
            self.portfolio_state = PortfolioState(
                balance=0.0,
                position=new_position,
                position_price=new_position_price,
                total_transaction_cost=self.portfolio_state.total_transaction_cost + fee,
            )
        elif side == "sell":
            new_position = max(0.0, self.portfolio_state.position - filled_qty)
            self.portfolio_state = PortfolioState(
                balance=0.0,
                position=new_position,
                position_price=self.portfolio_state.position_price,
                total_transaction_cost=self.portfolio_state.total_transaction_cost + fee,
            )

    def _wait_for_order_completion(self, order_id, timeout: float = 15.0, interval: float = 0.5):
        client = self.subaccount_client
        if client is None:
            return None
        deadline = time.monotonic() + max(timeout, interval)
        last_snapshot = None
        last_status = None
        last_filled = None
        while time.monotonic() < deadline:
            try:
                last_snapshot = client.get_order_by_id(order_id)
            except Exception as exc:
                LOGGER.warning("[%s] unable to refresh order %s: %s", self.symbol, order_id, exc)
                return last_snapshot
            status_text = self._extract_status_text(getattr(last_snapshot, "status", None))
            status_lower = status_text.lower()
            filled_qty = self._to_float(getattr(last_snapshot, "filled_qty", None))
            avg_price = self._to_float(getattr(last_snapshot, "filled_avg_price", None), default=0.0)
            if status_lower != last_status or filled_qty != last_filled:
                LOGGER.info(
                    "[%s] order %s | status=%s | filled_qty=%.6f | avg_fill=%.2f",
                    self.symbol,
                    order_id,
                    status_text or "unknown",
                    filled_qty,
                    avg_price,
                )
                last_status = status_lower
                last_filled = filled_qty
            if status_lower in {"filled", "partially_filled", "canceled", "expired", "done_for_day", "rejected"}:
                return last_snapshot
            time.sleep(interval)
        return last_snapshot

    def _build_order_result(
        self,
        order,
        trade_symbol: str,
        side: str,
        requested_quantity: float,
        reference_price: float,
    ) -> dict[str, object]:
        order_id = getattr(order, "id", None)
        status_text = self._extract_status_text(getattr(order, "status", None))
        status_lower = status_text.lower() if status_text else ""

        filled_qty = self._to_float(getattr(order, "filled_qty", None), 0.0)
        filled_avg_price = self._to_float(getattr(order, "filled_avg_price", None), reference_price)
        notional = self._to_float(getattr(order, "notional", None), requested_quantity * reference_price)

        return {
            "order_id": order_id,
            "symbol": self.symbol,
            "alpaca_symbol": trade_symbol,
            "side": side,
            "quantity": requested_quantity,
            "filled_quantity": filled_qty,
            "price": reference_price,
            "filled_avg_price": filled_avg_price,
            "notional": notional,
            "status": status_text,
            "status_lower": status_lower,
            "submitted_at": getattr(order, "submitted_at", None),
            "filled_at": getattr(order, "filled_at", None),
            "updated_at": getattr(order, "updated_at", None),
        }

    @staticmethod
    def _extract_status_text(status_raw) -> str:
        if status_raw is None:
            return ""
        if hasattr(status_raw, "value"):
            text = str(status_raw.value)
        elif hasattr(status_raw, "name"):
            text = str(status_raw.name)
        else:
            text = str(status_raw)
        if text.startswith("OrderStatus."):
            text = text.split(".", 1)[1]
        return text

    def _order_debug_snapshot(self, order) -> dict[str, object]:
        if order is None:
            return {}
        attrs = (
            "id",
            "client_order_id",
            "symbol",
            "qty",
            "notional",
            "filled_qty",
            "filled_avg_price",
            "side",
            "type",
            "time_in_force",
            "status",
            "status_message",
            "failed_reason",
            "submitted_at",
            "filled_at",
            "canceled_at",
            "expired_at",
            "created_at",
            "updated_at",
        )
        return {attr: self._serialize_order_value(getattr(order, attr, None)) for attr in attrs}

    @staticmethod
    def _serialize_order_value(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, (int, float, str, bool)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
                return str(value)
        return str(value)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _maybe_build_writer(tb_log_dir: str | None):
        if not tb_log_dir:
            return None
        try:  # pragma: no cover - import-time path varies by env
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not import SummaryWriter (%s); live TB disabled", exc)
            return None
        # flush_secs=1 caps how long a scalar can sit in the background queue
        # before being written to disk. The live trader is long-running and
        # any unclean shutdown (kernel freeze, power loss, OOM on the host)
        # will otherwise lose up to ~2 min of Live/* scalars.
        return SummaryWriter(log_dir=tb_log_dir, flush_secs=1)

    def _emit_action_and_q_scalars(self, action_index: int, q_values: np.ndarray | None) -> None:
        if self.writer is None:
            return
        step = self._global_step
        if 0 <= action_index < self._action_counts.size:
            self._action_counts[action_index] += 1
        total = max(int(self._action_counts.sum()), 1)
        for k in range(self._action_counts.size):
            self.writer.add_scalar(f"Live/Action Rate/{k}", float(self._action_counts[k]) / total, step)
        self.writer.add_scalar(f"Live/Action/{self.symbol}", action_index, step)

        if q_values is not None and q_values.size > 0:
            q = np.asarray(q_values, dtype=np.float32).reshape(-1)
            self.writer.add_scalar("Live/Q/Mean", float(q.mean()), step)
            self.writer.add_scalar("Live/Q/MaxAcrossActions", float(q.max()), step)
            self.writer.add_scalar("Live/Q/MinAcrossActions", float(q.min()), step)
            top2 = np.partition(q, -2)[-2:] if q.size >= 2 else q
            margin = float(top2[-1] - top2[0]) if q.size >= 2 else 0.0
            self.writer.add_scalar("Live/Q/ActionMargin", margin, step)
            if 0 <= action_index < q.size:
                self.writer.add_scalar("Live/Q/Selected", float(q[action_index]), step)

    def _emit_step_state(self, action_index: int, was_greedy: bool, portfolio_value: float) -> None:
        if self.writer is not None:
            step = self._global_step
            self.writer.add_scalar(f"Live/PortfolioValue/{self.symbol}", float(portfolio_value), step)
            self.writer.add_scalar(f"Live/Position/{self.symbol}", float(self.portfolio_state.position), step)

        self.step_records.append(
            StepRecord(
                step_index=self.step_count,
                portfolio_value=float(portfolio_value),
                position=float(self.portfolio_state.position),
                price=float(self.last_price),
                action=int(action_index),
                transaction_cost=0.0,
                was_greedy=bool(was_greedy),
            )
        )
        self.step_count += 1

    def _maybe_emit_trade_close(self, pre_position: float) -> None:
        post_position = self.portfolio_state.position
        if pre_position <= 1e-9 or post_position > 1e-9:
            return
        trades = segment_trades(self.step_records)
        if not trades:
            return
        new_trades = trades[self.closed_trade_count :]
        if not new_trades:
            return
        self.closed_trade_count = len(trades)

        if self.writer is None and self._trades_jsonl_path is None:
            return

        agg = aggregate_trade_metrics(trades)
        step = self._global_step
        if self.writer is not None:
            for tag, key in (
                ("Live/Trade/Count", "trade_count"),
                ("Live/Trade/HitRate", "hit_rate"),
                ("Live/Trade/Expectancy", "expectancy_pct"),
                ("Live/Trade/PerTradeSharpe", "per_trade_sharpe"),
                ("Live/Trade/ProfitFactor", "profit_factor"),
                ("Live/Trade/AvgDuration", "avg_duration_steps"),
                ("Live/Trade/AvgMAE", "avg_mae_pct"),
                ("Live/Trade/AvgMFE", "avg_mfe_pct"),
                ("Live/Trade/PctGreedy", "pct_greedy_actions"),
                ("Live/Trade/TotalPnLAbs", "total_pnl_absolute"),
                ("Live/Trade/TotalTxnCost", "total_transaction_cost"),
            ):
                value = agg.get(key, float("nan"))
                if value is None or (isinstance(value, float) and not np.isfinite(value)):
                    continue
                self.writer.add_scalar(f"{tag}/{self.symbol}", float(value), step)

        if self._trades_jsonl_path is not None:
            with self._trades_jsonl_path.open("a", encoding="utf-8") as fp:
                for trade in new_trades:
                    payload = trade.to_dict()
                    payload["symbol"] = self.symbol
                    payload["closed_at_step"] = step
                    fp.write(json.dumps(payload, default=float) + "\n")


__all__ = ["BarData", "LiveFeatureNormalizer", "MomentumLiveTrader"]
