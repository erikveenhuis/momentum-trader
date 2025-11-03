"""Live trading orchestration for the momentum agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional

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

    def process_bar(self, bar: BarData) -> Optional[Dict[str, object]]:
        symbol_state = self.symbol_states.get(bar.symbol)
        if symbol_state is None:
            LOGGER.debug("Received bar for untracked symbol %s", bar.symbol)
            return None

        symbol_state.update_price(bar.close)
        ready = symbol_state.normalizer.update(bar)
        if not ready:
            LOGGER.debug("Waiting for %s window: %s/%s bars", bar.symbol, symbol_state.normalizer.count, self.config.window_size)
            return None

        observation = symbol_state.observation(self.shared_balance)
        action_index = self.agent.select_action(observation)
        action_label, fraction = self._map_action(action_index)

        current_price = float(bar.close)
        trade_type = 0 if action_label == "hold" else 1 if action_label == "buy" else 2

        # Execute real order if trading client is available and action is not hold
        order_result = None
        if self.trading_client and action_label != "hold":
            order_result = self._execute_order(bar.symbol, action_label, fraction, current_price)
            if order_result:
                # Update shared balance and symbol position based on executed order
                self._update_shared_portfolio_from_order(order_result, trade_type, fraction, current_price)
                is_valid = True
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

        # Update shared portfolio value
        total_position_value = sum(max(0.0, state.portfolio_state.position * current_price) for state in self.symbol_states.values())
        self.shared_portfolio_value = self.shared_balance + total_position_value
        symbol_state.prev_portfolio_value = self.shared_portfolio_value

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

        order_info = f" | Order: {order_result['order_id']}" if order_result else ""
        LOGGER.info(
            "%s %s %.0f%% at %.2f | Shared PV %.2f (Balance: %.2f) | valid=%s%s",
            bar.symbol,
            action_label.upper(),
            fraction * 100,
            current_price,
            self.shared_portfolio_value,
            self.shared_balance,
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

            # Calculate order quantity based on action
            if action == "buy":
                # Calculate how much we can buy with available cash
                max_spend = available_cash * fraction
                quantity = max_spend / price
                if quantity < 0.00001:  # Minimum crypto order size
                    LOGGER.warning(f"Order too small for {symbol}: ${max_spend:.2f} at ${price:.2f}")
                    return None
            else:  # sell
                # Get current position for this symbol
                positions = self.trading_client.get_all_positions()
                current_position = 0.0
                for position in positions:
                    if position.symbol == symbol:
                        current_position = float(position.qty)
                        break

                quantity = current_position * fraction
                if quantity < 0.00001:  # Minimum crypto order size
                    LOGGER.warning(f"Insufficient position for {symbol} sell: {current_position} * {fraction} = {quantity}")
                    return None

            # Create and submit market order
            side = OrderSide.BUY if action == "buy" else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol, qty=quantity, side=side, time_in_force=TimeInForce.GTC  # Good 'til cancelled for crypto
            )

            order = self.trading_client.submit_order(order_request)

            LOGGER.info(f"Submitted {action.upper()} order for {quantity:.6f} {symbol} at market price")

            return {
                "order_id": order.id,
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "price": price,
                "status": order.status,
            }

        except Exception as e:
            LOGGER.error(f"Failed to execute {action} order for {symbol}: {e}")
            return None

    def _update_shared_portfolio_from_order(
        self, order_result: Dict[str, object], trade_type: int, fraction: float, current_price: float
    ) -> None:
        """Update shared portfolio state based on executed order."""
        try:
            symbol = order_result["symbol"]
            action = order_result["side"]
            quantity = order_result["quantity"]
            order_id = order_result["order_id"]

            if order_result["status"] == "accepted":
                LOGGER.info(f"Order {order_id} accepted for execution")

                # For simplicity, assume the order executes at the requested price
                # In production, you'd wait for actual fills
                if action == "buy":
                    cost = quantity * current_price
                    fee = cost * self.config.transaction_fee
                    total_cost = cost + fee

                    if self.shared_balance >= total_cost:
                        self.shared_balance -= total_cost

                        # Update symbol position
                        symbol_state = self.symbol_states[symbol]
                        current_position = symbol_state.portfolio_state.position
                        current_position_value = current_position * symbol_state.portfolio_state.position_price

                        new_position = current_position + quantity
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
                            f"Insufficient balance for {symbol} buy order: need ${total_cost:.2f}, have ${self.shared_balance:.2f}"
                        )

                elif action == "sell":
                    # Update symbol position
                    symbol_state = self.symbol_states[symbol]
                    current_position = symbol_state.portfolio_state.position

                    if current_position >= quantity:
                        proceeds = quantity * current_price
                        fee = proceeds * self.config.transaction_fee
                        net_proceeds = proceeds - fee

                        self.shared_balance += net_proceeds

                        new_position = current_position - quantity

                        symbol_state.portfolio_state = PortfolioState(
                            balance=0.0,
                            position=new_position,
                            position_price=symbol_state.portfolio_state.position_price,  # Keep existing price
                            total_transaction_cost=symbol_state.portfolio_state.total_transaction_cost + fee,
                        )

                        LOGGER.info(f"Updated position for {symbol}: {current_position:.6f} -> {new_position:.6f}")
                    else:
                        LOGGER.warning(f"Insufficient position for {symbol} sell order: need {quantity:.6f}, have {current_position:.6f}")

            else:
                LOGGER.warning(f"Order {order_id} status: {order_result['status']}")

        except Exception as e:
            LOGGER.error(f"Error updating shared portfolio from order: {e}")

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
