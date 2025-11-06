"""Live trading orchestration for the momentum agent."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, Optional

import numpy as np
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger
from momentum_env.trading import PortfolioState, TradingLogic

from .config import LiveTradingConfig

LOGGER = get_logger("momentum_live.trader")

FEATURE_NAMES = ("open", "high", "low", "close", "volume")


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
    def from_alpaca(cls, bar: object) -> "BarData":
        """Create ``BarData`` from an ``alpaca-py`` bar object."""

        return cls(
            symbol=getattr(bar, "symbol"),
            open=float(getattr(bar, "open")),
            high=float(getattr(bar, "high")),
            low=float(getattr(bar, "low")),
            close=float(getattr(bar, "close")),
            volume=float(getattr(bar, "volume", 0.0)),
            timestamp=getattr(bar, "timestamp", datetime.utcnow()),
        )


class LiveFeatureNormalizer:
    """Maintain a sliding window of normalized OHLCV features."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._raw_history: Dict[str, Deque[float]] = {name: deque(maxlen=window_size) for name in FEATURE_NAMES}
        self._normalized: Deque[np.ndarray] = deque(maxlen=window_size)

    @property
    def count(self) -> int:
        return len(self._normalized)

    def update(self, bar: BarData) -> bool:
        for feature in FEATURE_NAMES:
            value = float(getattr(bar, feature))
            self._raw_history[feature].append(value)

        normalized_row = []
        for feature in FEATURE_NAMES:
            history = self._raw_history[feature]
            current_value = history[-1]
            min_value = min(history)
            max_value = max(history)
            denom = max(max_value - min_value, 1e-8)
            normalized = np.clip((current_value - min_value) / denom, 0.0, 1.0)
            normalized_row.append(normalized)

        self._normalized.append(np.asarray(normalized_row, dtype=np.float32))
        return self.count >= self.window_size

    def window(self) -> np.ndarray:
        data = list(self._normalized)
        result = np.zeros((self.window_size, len(FEATURE_NAMES)), dtype=np.float32)
        if data:
            result[-len(data) :, :] = np.stack(data)
        return result


@dataclass(slots=True)
class SymbolState:
    symbol: str
    normalizer: LiveFeatureNormalizer
    trading_logic: TradingLogic
    portfolio_state: PortfolioState
    prev_portfolio_value: float
    last_price: float = 0.0

    def observation(self, shared_balance: float) -> Dict[str, np.ndarray]:
        account_state = self._account_state(shared_balance)
        market_window = self.normalizer.window()
        return {
            "market_data": market_window,
            "account_state": account_state,
        }

    def update_price(self, price: float) -> None:
        self.last_price = float(price)

    def _account_state(self, shared_balance: float) -> np.ndarray:
        price = max(self.last_price, 1e-9)
        position_value = max(0.0, self.portfolio_state.position * price)
        balance = max(0.0, shared_balance)
        portfolio_value = max(balance + position_value, 1e-9)

        normalized_position = position_value / portfolio_value
        normalized_balance = balance / portfolio_value
        return np.asarray([normalized_position, normalized_balance], dtype=np.float32)


class MomentumLiveTrader:
    """Glue logic between the live data stream and the agent."""

    ACTION_SPACE = {
        0: ("hold", 0.0),
        1: ("buy", 0.10),
        2: ("buy", 0.25),
        3: ("buy", 0.50),
        4: ("sell", 0.10),
        5: ("sell", 0.25),
        6: ("sell", 1.00),
    }

    def __init__(self, agent: RainbowDQNAgent, config: LiveTradingConfig):
        self.agent = agent
        self.config = config
        self.trading_logic = TradingLogic(
            transaction_fee=config.transaction_fee,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
        )
        self.symbol_states: Dict[str, SymbolState] = {}
        self.trading_client = None  # Will be set by AlpacaStreamRunner

        # Shared account balance across all symbols
        self.shared_balance = config.initial_balance
        self.shared_portfolio_value = config.initial_balance

        # Minimum notional amount for live orders to avoid micro-trades
        self.min_notional = float(getattr(config, "min_notional", 10.0))

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

    def process_bar(self, bar: BarData) -> Optional[Dict[str, object]]:
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
        action_label, fraction = self._map_action(action_index)

        LOGGER.info(
            "Action selected | symbol=%s | ts=%s | action=%s | index=%d | fraction=%.2f | price=%.2f",
            bar.symbol,
            bar.timestamp,
            action_label,
            action_index,
            fraction,
            bar.close,
        )

        current_price = float(bar.close)
        trade_type = 0 if action_label == "hold" else 1 if action_label == "buy" else 2

        # Execute real order if trading client is available and action is not hold
        order_result = None
        if self.trading_client and action_label != "hold":
            if action_label == "sell" and symbol_state.portfolio_state.position <= 0.0:
                LOGGER.warning(
                    "Skipping live SELL for %s due to zero position | position=%.6f",
                    bar.symbol,
                    symbol_state.portfolio_state.position,
                )
                is_valid = False
            elif action_label == "buy" and self.shared_balance <= 0.0:
                LOGGER.warning(
                    "Skipping live BUY for %s due to zero balance | shared_balance=%.2f",
                    bar.symbol,
                    self.shared_balance,
                )
                is_valid = False
            elif action_label == "sell" and symbol_state.portfolio_state.position * current_price < self.min_notional:
                LOGGER.warning(
                    "Skipping live SELL for %s due to insufficient notional | position=%.6f | min_notional=%.2f",
                    bar.symbol,
                    symbol_state.portfolio_state.position,
                    self.min_notional,
                )
                is_valid = False
            elif action_label == "buy" and (self.shared_balance * fraction) < self.min_notional:
                LOGGER.warning(
                    "Skipping live BUY for %s due to insufficient notional | balance=%.2f | fraction=%.2f | min_notional=%.2f",
                    bar.symbol,
                    self.shared_balance,
                    fraction,
                    self.min_notional,
                )
                is_valid = False
            else:
                LOGGER.info(
                    "Submitting live order | symbol=%s | action=%s | fraction=%.2f | price=%.2f",
                    bar.symbol,
                    action_label,
                    fraction,
                    current_price,
                )
                order_result = self._execute_order(bar.symbol, action_label, fraction, current_price)
                if order_result:
                    # Update shared balance and symbol position based on executed order
                    executed = self._update_shared_portfolio_from_order(order_result, trade_type, fraction, current_price)
                    is_valid = executed
                else:
                    # Order failed, mark as invalid
                    is_valid = False
        else:
            # Simulate trade for validation or when no trading client
            # For simulation, create a temporary portfolio state with shared balance
            temp_portfolio_state = PortfolioState(
                balance=self.shared_balance,
                position=symbol_state.portfolio_state.position,
                position_price=symbol_state.portfolio_state.position_price,
                total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost,
            )
            new_portfolio_state, is_valid = symbol_state.trading_logic.apply_trade(
                temp_portfolio_state,
                current_price=current_price,
                action=trade_type,
                action_value=fraction,
            )
            # Update shared balance and symbol position
            self.shared_balance = new_portfolio_state.balance
            symbol_state.portfolio_state = PortfolioState(
                balance=0.0,  # Symbol doesn't own balance
                position=new_portfolio_state.position,
                position_price=new_portfolio_state.position_price,
                total_transaction_cost=new_portfolio_state.total_transaction_cost,
            )
            LOGGER.info(
                "Simulated trade | symbol=%s | action=%s | fraction=%.2f | new_balance=%.2f | new_position=%.6f",
                bar.symbol,
                action_label,
                fraction,
                self.shared_balance,
                symbol_state.portfolio_state.position,
            )

        # Update shared portfolio value
        total_position_value = sum(max(0.0, state.portfolio_state.position * current_price) for state in self.symbol_states.values())
        self.shared_portfolio_value = self.shared_balance + total_position_value
        symbol_state.prev_portfolio_value = self.shared_portfolio_value

        post_shared_balance = self.shared_balance
        post_position = symbol_state.portfolio_state.position
        post_portfolio_value = self.shared_portfolio_value

        decision = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp,
            "action_index": action_index,
            "action": action_label,
            "fraction": fraction,
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

        LOGGER.info(
            "Decision summary | symbol=%s | ts=%s | action=%s | fraction=%.2f | price=%.2f | balance=%.2f->%.2f | "
            "position=%.6f->%.6f | portfolio=%.2f->%.2f | valid=%s%s",
            bar.symbol,
            bar.timestamp,
            action_label,
            fraction,
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

    def _execute_order(self, symbol: str, action: str, fraction: float, price: float) -> Optional[Dict[str, object]]:
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
                if max_spend < self.min_notional:
                    LOGGER.warning(
                        "Order notional too small for %s buy: spend=%.2f < min_notional=%.2f",
                        symbol,
                        max_spend,
                        self.min_notional,
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
                if notional < self.min_notional:
                    LOGGER.warning(
                        "Order notional too small for %s sell: notional=%.2f < min_notional=%.2f",
                        symbol,
                        notional,
                        self.min_notional,
                    )
                    return None

            if action == "buy":
                notional = quantity * price
                if notional < self.min_notional:
                    LOGGER.warning(
                        "Order notional too small for %s buy after sizing: notional=%.2f < min_notional=%.2f",
                        symbol,
                        notional,
                        self.min_notional,
                    )
                    return None

            # Create and submit market order
            side = OrderSide.BUY if action == "buy" else OrderSide.SELL

            trade_symbol = symbol.replace("/", "") if "/" in symbol else symbol

            order_request = MarketOrderRequest(
                symbol=trade_symbol, qty=quantity, side=side, time_in_force=TimeInForce.GTC  # Good 'til cancelled for crypto
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

            completed_order = self._wait_for_order_completion(order.id)
            if completed_order is not None:
                order = completed_order

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
        self, order_result: Dict[str, object], trade_type: int, fraction: float, current_price: float
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

        while time.monotonic() < deadline:
            try:
                last_snapshot = self.trading_client.get_order_by_id(order_id)
            except Exception as exc:
                LOGGER.warning("Unable to refresh order %s: %s", order_id, exc)
                return last_snapshot

            status_text = self._extract_status_text(getattr(last_snapshot, "status", None))
            status_lower = status_text.lower()

            if status_lower in {"filled", "partially_filled", "canceled", "expired", "done_for_day", "rejected"}:
                return last_snapshot

            time.sleep(interval)

        if last_snapshot is not None:
            status_text = self._extract_status_text(getattr(last_snapshot, "status", None)) or "unknown"
            LOGGER.info(
                "Order %s still %s after %.1fs timeout; continuing with latest snapshot",
                order_id,
                status_text,
                timeout,
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
    ) -> Dict[str, object]:
        """Normalize an Alpaca order object into a serializable dict."""

        order_id = getattr(order, "id", None)
        status_text = self._extract_status_text(getattr(order, "status", None))
        status_lower = status_text.lower() if status_text else ""

        def _to_float(value, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        filled_qty = _to_float(getattr(order, "filled_qty", None), 0.0)
        filled_avg_price = _to_float(getattr(order, "filled_avg_price", None), reference_price)

        return {
            "order_id": order_id,
            "symbol": submitted_symbol,
            "alpaca_symbol": trade_symbol,
            "side": side,
            "quantity": requested_quantity,
            "filled_quantity": filled_qty,
            "price": reference_price,
            "filled_avg_price": filled_avg_price,
            "status": status_text,
            "status_lower": status_lower,
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

    def _update_portfolio_from_order(
        self, portfolio_state: PortfolioState, order_result: Dict[str, object], trade_type: int, fraction: float, current_price: float
    ) -> tuple[PortfolioState, bool]:
        """Legacy method - kept for compatibility but should not be used."""
        # This method is deprecated in favor of _update_shared_portfolio_from_order
        return portfolio_state, False

    def _map_action(self, action_index: int) -> tuple[str, float]:
        if action_index not in self.ACTION_SPACE:
            raise ValueError(f"Received unsupported action index {action_index}")
        return self.ACTION_SPACE[action_index]


__all__ = ["BarData", "LiveFeatureNormalizer", "MomentumLiveTrader"]
