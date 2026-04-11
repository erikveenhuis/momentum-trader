"""Live trading orchestration for the momentum agent."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger
from momentum_env.data import N_RAW_FEATURES
from momentum_env.trading import PortfolioState, TradingLogic

from .config import LiveTradingConfig

LOGGER = get_logger("momentum_live.trader")

RAW_COLS = ("open", "high", "low", "close", "volume")
N_DERIVED = 6
N_TOTAL = N_RAW_FEATURES + N_DERIVED
ROLLING_WINDOW = 20


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
            symbol=getattr(bar, "symbol"),
            open=float(getattr(bar, "open")),
            high=float(getattr(bar, "high")),
            low=float(getattr(bar, "low")),
            close=float(getattr(bar, "close")),
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
        T = len(raw_arr)

        closes = np.array(list(self._close_history), dtype=np.float64)
        volumes = np.array(list(self._volume_history), dtype=np.float64)

        safe_closes = np.where(closes > 0, closes, 1e-20)
        log_ret_1 = np.zeros(T, dtype=np.float32)
        log_ret_5 = np.zeros(T, dtype=np.float32)
        log_ret_10 = np.zeros(T, dtype=np.float32)
        if T > 1:
            log_ret_1[1:] = np.log(safe_closes[1:] / safe_closes[:-1]).astype(np.float32)
        if T > 5:
            log_ret_5[5:] = np.log(safe_closes[5:] / safe_closes[:-5]).astype(np.float32)
        if T > 10:
            log_ret_10[10:] = np.log(safe_closes[10:] / safe_closes[:-10]).astype(np.float32)

        realized_vol = pd.Series(log_ret_1).rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0).values.astype(np.float32)
        vol_mean = pd.Series(volumes).rolling(ROLLING_WINDOW, min_periods=1).mean().values
        volume_ratio = (volumes / np.where(vol_mean > 1e-20, vol_mean, 1e-20)).astype(np.float32)
        hl_range = (raw_arr[:, 1] - raw_arr[:, 2]) / np.where(raw_arr[:, 3] > 0, raw_arr[:, 3], 1e-20)
        hl_mean = pd.Series(hl_range).rolling(ROLLING_WINDOW, min_periods=1).mean().values
        hl_range_ratio = (hl_range / np.where(hl_mean > 1e-20, hl_mean, 1e-20)).astype(np.float32)

        derived = np.column_stack([log_ret_1, log_ret_5, log_ret_10, realized_vol, volume_ratio, hl_range_ratio])
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


@dataclass(slots=True)
class SymbolState:
    """Per-symbol tracking: feature normalizer, portfolio position, and account-state computation."""

    symbol: str
    normalizer: LiveFeatureNormalizer
    trading_logic: TradingLogic
    portfolio_state: PortfolioState
    prev_portfolio_value: float
    last_price: float = 0.0

    def observation(self, shared_balance: float) -> dict[str, np.ndarray]:
        account_state = self._account_state(shared_balance)
        market_window = self.normalizer.window()
        return {
            "market_data": market_window,
            "account_state": account_state,
        }

    def update_price(self, price: float) -> None:
        self.last_price = float(price)

    bars_in_position: int = 0

    def _account_state(self, shared_balance: float) -> np.ndarray:
        price = max(self.last_price, 1e-9)
        position_value = max(0.0, self.portfolio_state.position * price)
        balance = max(0.0, shared_balance)
        portfolio_value = max(balance + position_value, 1e-9)

        normalized_position = np.clip(position_value / portfolio_value, 0.0, 1.0)
        normalized_balance = np.clip(balance / portfolio_value, 0.0, 1.0)

        unrealized_pnl = 0.0
        if self.portfolio_state.position > 1e-9 and self.portfolio_state.position_price > 1e-9:
            unrealized_pnl = (price - self.portfolio_state.position_price) / self.portfolio_state.position_price

        safe_initial = max(self.prev_portfolio_value, 1e-9)

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


class MomentumLiveTrader:
    """Glue logic between the live data stream and the agent."""

    TARGET_ALLOCATIONS = {
        0: 0.0,
        1: 0.2,
        2: 0.4,
        3: 0.6,
        4: 0.8,
        5: 1.0,
    }

    def __init__(self, agent: RainbowDQNAgent, config: LiveTradingConfig):
        self.agent = agent
        self.config = config
        self.trading_logic = TradingLogic(
            transaction_fee=config.transaction_fee,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
            drawdown_penalty_lambda=config.drawdown_penalty_lambda,
            slippage_bps=config.slippage_bps,
            opportunity_cost_lambda=config.opportunity_cost_lambda,
            min_trade_value=config.min_trade_value,
        )
        self.symbol_states: dict[str, SymbolState] = {}
        self.trading_client = None  # Will be set by AlpacaStreamRunner

        # Shared account balance across all symbols
        self.shared_balance = config.initial_balance
        self.shared_portfolio_value = config.initial_balance

        # Minimum notional amount for live orders to avoid micro-trades
        self.min_rebalance_pct = config.min_rebalance_pct

        for symbol in config.symbols:
            normalizer = LiveFeatureNormalizer(config.window_size)
            # Each symbol tracks only its own position, balance comes from shared pool
            portfolio_state = PortfolioState(
                balance=0.0,  # Individual symbols don't have their own balance
                position=0.0,
                position_price=0.0,
                total_transaction_cost=0.0,
            )
            self.symbol_states[symbol] = SymbolState(
                symbol=symbol,
                normalizer=normalizer,
                trading_logic=self.trading_logic,
                portfolio_state=portfolio_state,
                prev_portfolio_value=config.initial_balance,
            )

    def set_trading_client(self, client) -> None:
        """Set the Alpaca trading client for order execution."""
        self.trading_client = client
        LOGGER.info("Trading client configured for live order execution")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.symbol_states.keys())

    def preload_history(self, bars: Iterable[BarData]) -> None:
        """Seed normalizers with historical bars before live streaming."""

        ordered_bars = sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))
        if not ordered_bars:
            LOGGER.info("No historical bars provided for warmup")
            return

        for bar in ordered_bars:
            symbol_state = self.symbol_states.get(bar.symbol)
            if symbol_state is None:
                LOGGER.debug("Skipping warmup bar for untracked symbol %s", bar.symbol)
                continue

            symbol_state.update_price(bar.close)
            symbol_state.normalizer.update(bar)

        for symbol, state in self.symbol_states.items():
            LOGGER.info(
                "Warmup status | symbol=%s | window_count=%d/%d",
                symbol,
                state.normalizer.count,
                state.normalizer.window_size,
            )

    def process_bar(self, bar: BarData) -> dict[str, object] | None:
        symbol_state = self.symbol_states.get(bar.symbol)
        if symbol_state is None:
            LOGGER.debug("Received bar for untracked symbol %s", bar.symbol)
            return None

        LOGGER.info(
            "Bar received | symbol=%s | ts=%s | open=%.2f | high=%.2f | low=%.2f | close=%.2f | volume=%.6f",
            bar.symbol,
            bar.timestamp,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )

        pre_shared_balance = self.shared_balance
        pre_portfolio_value = self.shared_portfolio_value
        pre_position = symbol_state.portfolio_state.position

        symbol_state.update_price(bar.close)
        ready = symbol_state.normalizer.update(bar)
        if not ready:
            LOGGER.debug("Waiting for %s window: %s/%s bars", bar.symbol, symbol_state.normalizer.count, self.config.window_size)
            return None

        observation = symbol_state.observation(self.shared_balance)
        LOGGER.info(
            "Observation ready | symbol=%s | balance=%.2f | position=%.6f | window_count=%d",
            bar.symbol,
            pre_shared_balance,
            pre_position,
            symbol_state.normalizer.count,
        )

        action_index = self.agent.select_action(observation)
        target_frac = self.TARGET_ALLOCATIONS[action_index]

        current_price = float(bar.close)

        LOGGER.info(
            "Action selected | symbol=%s | ts=%s | target_alloc=%.0f%% | index=%d | price=%.2f",
            bar.symbol,
            bar.timestamp,
            target_frac * 100,
            action_index,
            current_price,
        )

        portfolio_value = self.shared_balance + max(0.0, symbol_state.portfolio_state.position * current_price)
        current_position_value = max(0.0, symbol_state.portfolio_state.position * current_price)
        current_frac = current_position_value / max(portfolio_value, 1e-9)
        delta_frac = target_frac - current_frac
        delta_value = target_frac * portfolio_value - current_position_value

        is_valid = True

        if abs(delta_frac) < self.min_rebalance_pct:
            LOGGER.debug("Delta below min rebalance (%.2f%% < %.2f%%)", abs(delta_frac) * 100, self.min_rebalance_pct * 100)
        elif delta_value > 0:
            buy_cash = min(delta_value, self.shared_balance)
            if buy_cash / portfolio_value < self.min_rebalance_pct:
                LOGGER.debug("Buy cash below min notional, skipping")
            elif self.trading_client:
                order_result = self._execute_order(bar.symbol, "buy", buy_cash / max(self.shared_balance, 1e-9), current_price)
                if order_result:
                    self._update_shared_portfolio_from_order(order_result, 1, buy_cash / max(self.shared_balance, 1e-9), current_price)
                else:
                    is_valid = False
            else:
                temp_ps = PortfolioState(
                    balance=self.shared_balance,
                    position=symbol_state.portfolio_state.position,
                    position_price=symbol_state.portfolio_state.position_price,
                    total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost,
                )
                buy_fraction = buy_cash / max(self.shared_balance, 1e-9)
                is_valid, new_ps = symbol_state.trading_logic.handle_buy(temp_ps, current_price, buy_fraction)
                if is_valid:
                    self.shared_balance = new_ps.balance
                    symbol_state.portfolio_state = PortfolioState(
                        balance=0.0,
                        position=new_ps.position,
                        position_price=new_ps.position_price,
                        total_transaction_cost=new_ps.total_transaction_cost,
                    )
        else:
            sell_value = abs(delta_value)
            if current_position_value < 1e-9 or sell_value / portfolio_value < self.min_rebalance_pct:
                LOGGER.debug("Sell value below min notional or no position, skipping")
            elif self.trading_client:
                sell_fraction = min(sell_value / current_position_value, 1.0)
                order_result = self._execute_order(bar.symbol, "sell", sell_fraction, current_price)
                if order_result:
                    self._update_shared_portfolio_from_order(order_result, 2, sell_fraction, current_price)
                else:
                    is_valid = False
            else:
                temp_ps = PortfolioState(
                    balance=self.shared_balance,
                    position=symbol_state.portfolio_state.position,
                    position_price=symbol_state.portfolio_state.position_price,
                    total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost,
                )
                sell_fraction = min(sell_value / current_position_value, 1.0)
                is_valid, new_ps = symbol_state.trading_logic.handle_sell(temp_ps, current_price, sell_fraction)
                if is_valid:
                    self.shared_balance = new_ps.balance
                    symbol_state.portfolio_state = PortfolioState(
                        balance=0.0,
                        position=new_ps.position,
                        position_price=new_ps.position_price,
                        total_transaction_cost=new_ps.total_transaction_cost,
                    )

        if symbol_state.portfolio_state.position > 1e-9:
            symbol_state.bars_in_position += 1
        else:
            symbol_state.bars_in_position = 0

        total_position_value = sum(max(0.0, state.portfolio_state.position * current_price) for state in self.symbol_states.values())
        self.shared_portfolio_value = self.shared_balance + total_position_value
        symbol_state.prev_portfolio_value = self.shared_portfolio_value

        post_shared_balance = self.shared_balance
        post_position = symbol_state.portfolio_state.position
        post_portfolio_value = self.shared_portfolio_value

        order_result = None
        decision = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp,
            "action_index": action_index,
            "target_allocation": target_frac,
            "price": current_price,
            "portfolio_value": self.shared_portfolio_value,
            "shared_balance": self.shared_balance,
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
            "Decision summary | symbol=%s | ts=%s | action=%s | price=%.2f | balance=%.2f->%.2f | "
            "position=%.6f->%.6f | portfolio=%.2f->%.2f | valid=%s%s",
            bar.symbol,
            bar.timestamp,
            action_label,
            current_price,
            pre_shared_balance,
            post_shared_balance,
            pre_position,
            post_position,
            pre_portfolio_value,
            post_portfolio_value,
            is_valid,
            order_info,
        )

        return decision

    def _execute_order(self, symbol: str, action: str, fraction: float, price: float) -> dict[str, object] | None:
        """Execute a real order on Alpaca and return order details."""
        if not self.trading_client:
            return None

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            # Get account information to determine available cash
            account = self.trading_client.get_account()
            available_cash = float(account.cash)
            spendable_cash = min(self.shared_balance, available_cash)

            LOGGER.info(
                "Account status | trading_blocked=%s | transfers_blocked=%s | account_blocked=%s | shorting_enabled=%s | status=%s",
                getattr(account, "trading_blocked", "n/a"),
                getattr(account, "transfers_blocked", "n/a"),
                getattr(account, "account_blocked", "n/a"),
                getattr(account, "shorting_enabled", "n/a"),
                getattr(account, "status", "n/a"),
            )

            LOGGER.info(
                "Order sizing | symbol=%s | action=%s | fraction=%.2f | price=%.2f | shared_balance=%.2f | account_cash=%.2f | spendable_cash=%.2f",
                symbol,
                action,
                fraction,
                price,
                self.shared_balance,
                available_cash,
                spendable_cash,
            )

            # Calculate order quantity based on action
            if action == "buy":
                if spendable_cash <= 0:
                    LOGGER.warning(
                        "Skipping %s buy due to zero spendable cash | shared_balance=%.2f | account_cash=%.2f",
                        symbol,
                        self.shared_balance,
                        available_cash,
                    )
                    return None

                # Calculate how much we can buy with available/shared cash
                max_spend = spendable_cash * fraction
                if max_spend < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "Order notional too small for %s buy: spend=%.2f < min_trade_value=%.2f",
                        symbol,
                        max_spend,
                        self.trading_logic.min_trade_value,
                    )
                    return None
                quantity = max_spend / price
                LOGGER.info(
                    "Buy sizing | symbol=%s | max_spend=%.2f | quantity=%.6f",
                    symbol,
                    max_spend,
                    quantity,
                )
                if quantity < 0.00001:  # Minimum crypto order size
                    LOGGER.warning(f"Order too small for {symbol}: ${max_spend:.2f} at ${price:.2f}")
                    return None
            else:  # sell
                # Get current position for this symbol
                positions = self.trading_client.get_all_positions()
                current_position = 0.0
                sanitized_symbol = symbol.replace("/", "")
                for position in positions:
                    position_symbol = getattr(position, "symbol", "")
                    if position_symbol == symbol or position_symbol == sanitized_symbol:
                        current_position = float(position.qty)
                        break

                quantity = current_position * fraction
                LOGGER.info(
                    "Sell sizing | symbol=%s | current_position=%.6f | quantity=%.6f",
                    symbol,
                    current_position,
                    quantity,
                )
                if quantity < 0.00001:  # Minimum crypto order size
                    LOGGER.warning(f"Insufficient position for {symbol} sell: {current_position} * {fraction} = {quantity}")
                    return None

                notional = quantity * price
                if notional < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "Order notional too small for %s sell: notional=%.2f < min_trade_value=%.2f",
                        symbol,
                        notional,
                        self.trading_logic.min_trade_value,
                    )
                    return None

            if action == "buy":
                notional = quantity * price
                if notional < self.trading_logic.min_trade_value:
                    LOGGER.warning(
                        "Order notional too small for %s buy after sizing: notional=%.2f < min_trade_value=%.2f",
                        symbol,
                        notional,
                        self.trading_logic.min_trade_value,
                    )
                    return None

            # Create and submit market order
            side = OrderSide.BUY if action == "buy" else OrderSide.SELL

            trade_symbol = symbol.replace("/", "") if "/" in symbol else symbol

            order_request = MarketOrderRequest(
                symbol=trade_symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.GTC,  # Good 'til cancelled for crypto
            )

            LOGGER.info(
                "Submitting %s order | symbol=%s | qty=%.6f | notional~=%.2f",
                action.upper(),
                trade_symbol,
                quantity,
                quantity * price,
            )

            order = self.trading_client.submit_order(order_request)

            LOGGER.info(f"Submitted {action.upper()} order for {quantity:.6f} {symbol} at market price")
            LOGGER.info(
                "Order submitted snapshot | %s",
                self._order_debug_snapshot(order),
            )

            completed_order = self._wait_for_order_completion(order.id)
            if completed_order is not None:
                order = completed_order
                LOGGER.info(
                    "Order completion snapshot | %s",
                    self._order_debug_snapshot(order),
                )

            return self._build_order_result(
                order=order,
                submitted_symbol=symbol,
                trade_symbol=trade_symbol,
                side=action,
                requested_quantity=quantity,
                reference_price=price,
            )

        except Exception as e:
            LOGGER.error(f"Failed to execute {action} order for {symbol}: {e}")
            return None

    def _update_shared_portfolio_from_order(
        self, order_result: dict[str, object], trade_type: int, fraction: float, current_price: float
    ) -> bool:
        """Update shared portfolio state based on executed order."""
        try:
            symbol = order_result["symbol"]
            action = order_result["side"]
            order_id = order_result["order_id"]

            status_value = order_result.get("status_lower") or str(order_result.get("status", "")).lower()
            filled_quantity = float(order_result.get("filled_quantity", 0.0) or 0.0)
            filled_price = float(order_result.get("filled_avg_price", current_price) or current_price)

            if filled_quantity <= 0.0:
                LOGGER.info(
                    "Order %s not filled yet (status=%s); skipping portfolio update",
                    order_id,
                    status_value or "unknown",
                )
                return False

            LOGGER.info(
                "Order %s status=%s – applying filled quantity %.6f at price %.2f",
                order_id,
                status_value,
                filled_quantity,
                filled_price,
            )

            if action == "buy":
                cost = filled_quantity * filled_price
                fee = cost * self.config.transaction_fee
                total_cost = cost + fee

                if self.shared_balance + 1e-9 >= total_cost:
                    self.shared_balance -= total_cost

                    symbol_state = self.symbol_states[symbol]
                    current_position = symbol_state.portfolio_state.position
                    current_position_value = current_position * symbol_state.portfolio_state.position_price

                    new_position = current_position + filled_quantity
                    new_position_price = (current_position_value + cost) / new_position if new_position > 0 else 0.0

                    symbol_state.portfolio_state = PortfolioState(
                        balance=0.0,
                        position=new_position,
                        position_price=new_position_price,
                        total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost + fee,
                    )

                    LOGGER.info(f"Updated position for {symbol}: {current_position:.6f} -> {new_position:.6f}")
                else:
                    LOGGER.warning(
                        "Insufficient balance for %s buy fill: need $%.2f, have $%.2f",
                        symbol,
                        total_cost,
                        self.shared_balance,
                    )
                    return False

            elif action == "sell":
                symbol_state = self.symbol_states[symbol]
                current_position = symbol_state.portfolio_state.position

                if current_position + 1e-9 >= filled_quantity:
                    proceeds = filled_quantity * filled_price
                    fee = proceeds * self.config.transaction_fee
                    net_proceeds = proceeds - fee

                    self.shared_balance += net_proceeds

                    new_position = max(0.0, current_position - filled_quantity)

                    symbol_state.portfolio_state = PortfolioState(
                        balance=0.0,
                        position=new_position,
                        position_price=symbol_state.portfolio_state.position_price,
                        total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost + fee,
                    )

                    LOGGER.info(f"Updated position for {symbol}: {current_position:.6f} -> {new_position:.6f}")
                else:
                    LOGGER.warning(
                        "Insufficient position for %s sell fill: need %.6f, have %.6f",
                        symbol,
                        filled_quantity,
                        current_position,
                    )
                    return False

            return True

        except Exception as e:
            LOGGER.error(f"Error updating shared portfolio from order: {e}")
            return False

    def _wait_for_order_completion(self, order_id: str, timeout: float = 15.0, interval: float = 0.5):
        """Poll Alpaca for order completion up to a timeout."""
        if not self.trading_client:
            return None

        deadline = time.monotonic() + max(timeout, interval)
        last_snapshot = None
        last_status = None
        last_filled = None

        while time.monotonic() < deadline:
            try:
                last_snapshot = self.trading_client.get_order_by_id(order_id)
            except Exception as exc:
                LOGGER.warning("Unable to refresh order %s: %s", order_id, exc)
                return last_snapshot

            status_text = self._extract_status_text(getattr(last_snapshot, "status", None))
            status_lower = status_text.lower()
            filled_qty = self._to_float(getattr(last_snapshot, "filled_qty", None))
            avg_price = self._to_float(getattr(last_snapshot, "filled_avg_price", None), default=0.0)
            failed_reason = getattr(last_snapshot, "failed_reason", None)
            status_message = getattr(last_snapshot, "status_message", None)

            if status_lower != last_status or filled_qty != last_filled:
                LOGGER.info(
                    "Order %s status update | status=%s | filled_qty=%.6f | avg_fill=%.2f | failed_reason=%s | status_message=%s",
                    order_id,
                    status_text or "unknown",
                    filled_qty,
                    avg_price,
                    failed_reason or "",
                    status_message or "",
                )
                last_status = status_lower
                last_filled = filled_qty

            if status_lower in {"filled", "partially_filled", "canceled", "expired", "done_for_day", "rejected"}:
                return last_snapshot

            time.sleep(interval)

        if last_snapshot is not None:
            status_text = self._extract_status_text(getattr(last_snapshot, "status", None)) or "unknown"
            LOGGER.info(
                "Order %s still %s after %.1fs timeout; continuing with latest snapshot | snapshot=%s",
                order_id,
                status_text,
                timeout,
                self._order_debug_snapshot(last_snapshot),
            )

        return last_snapshot

    def _build_order_result(
        self,
        order,
        submitted_symbol: str,
        trade_symbol: str,
        side: str,
        requested_quantity: float,
        reference_price: float,
    ) -> dict[str, object]:
        """Normalize an Alpaca order object into a serializable dict."""

        order_id = getattr(order, "id", None)
        status_text = self._extract_status_text(getattr(order, "status", None))
        status_lower = status_text.lower() if status_text else ""

        filled_qty = self._to_float(getattr(order, "filled_qty", None), 0.0)
        filled_avg_price = self._to_float(getattr(order, "filled_avg_price", None), reference_price)
        notional = self._to_float(getattr(order, "notional", None), requested_quantity * reference_price)
        limit_price = self._to_float(getattr(order, "limit_price", None), 0.0)
        stop_price = self._to_float(getattr(order, "stop_price", None), 0.0)

        return {
            "order_id": order_id,
            "symbol": submitted_symbol,
            "alpaca_symbol": trade_symbol,
            "side": side,
            "quantity": requested_quantity,
            "filled_quantity": filled_qty,
            "price": reference_price,
            "filled_avg_price": filled_avg_price,
            "notional": notional,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "status": status_text,
            "status_lower": status_lower,
            "failed_reason": getattr(order, "failed_reason", None),
            "status_message": getattr(order, "status_message", None),
            "client_order_id": getattr(order, "client_order_id", None),
            "submitted_at": getattr(order, "submitted_at", None),
            "filled_at": getattr(order, "filled_at", None),
            "updated_at": getattr(order, "updated_at", None),
        }

    @staticmethod
    def _extract_status_text(status_raw) -> str:
        """Convert various Alpaca status representations into a plain string."""
        if status_raw is None:
            return ""
        if hasattr(status_raw, "value"):
            status_text = str(status_raw.value)
        elif hasattr(status_raw, "name"):
            status_text = str(status_raw.name)
        else:
            status_text = str(status_raw)

        if status_text.startswith("OrderStatus."):
            status_text = status_text.split(".", 1)[1]

        return status_text

    def _order_debug_snapshot(self, order) -> dict[str, object]:
        if order is None:
            return {}

        attributes = [
            "id",
            "client_order_id",
            "symbol",
            "qty",
            "notional",
            "filled_qty",
            "filled_avg_price",
            "side",
            "type",
            "order_type",
            "time_in_force",
            "limit_price",
            "stop_price",
            "status",
            "status_message",
            "failed_reason",
            "submitted_at",
            "filled_at",
            "canceled_at",
            "expired_at",
            "created_at",
            "updated_at",
        ]

        snapshot: dict[str, object] = {}
        for attr in attributes:
            value = getattr(order, attr, None)
            snapshot[attr] = self._serialize_order_value(value)

        return snapshot

    @staticmethod
    def _serialize_order_value(value):
        if value is None:
            return None
        if isinstance(value, (datetime,)):
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
            except Exception:  # pragma: no cover - defensive
                return str(value)
        return str(value)

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _map_action(self, action_index: int) -> tuple[str, float]:
        if action_index not in self.TARGET_ALLOCATIONS:
            raise ValueError(f"Received unsupported action index {action_index}")
        target = self.TARGET_ALLOCATIONS[action_index]
        return f"target_{int(target * 100)}pct", target


__all__ = ["BarData", "LiveFeatureNormalizer", "MomentumLiveTrader"]
