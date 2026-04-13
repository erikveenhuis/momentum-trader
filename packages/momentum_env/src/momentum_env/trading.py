"""Trading logic for the trading environment."""

from dataclasses import dataclass


@dataclass
class PortfolioState:
    """Container for portfolio state."""

    balance: float
    position: float
    position_price: float
    total_transaction_cost: float

    def portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value at current price."""
        return max(0, self.balance + self.position * current_price)


class TradingLogic:
    """Handle trading logic and portfolio management."""

    def __init__(
        self,
        transaction_fee: float,
        reward_scale: float,
        invalid_action_penalty: float,
        drawdown_penalty_lambda: float,
        slippage_bps: float,
        opportunity_cost_lambda: float,
        min_trade_value: float,
        benchmark_allocation_frac: float,
    ) -> None:
        self.transaction_fee = transaction_fee
        self.reward_scale = reward_scale
        self.invalid_action_penalty = invalid_action_penalty
        self.drawdown_penalty_lambda = drawdown_penalty_lambda
        self.slippage_bps = slippage_bps
        self.opportunity_cost_lambda = opportunity_cost_lambda
        self.min_trade_value: float = min_trade_value
        self.benchmark_allocation_frac = benchmark_allocation_frac
        self.peak_portfolio_value: float = 0.0

    def handle_buy(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action_value: float,
    ) -> tuple[bool, PortfolioState]:
        """Handle buy action logic.

        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action_value: Fraction of balance to use for buying (0 to 1)

        Returns:
            Tuple of (is_valid, new_portfolio_state)
        """
        if action_value > 1.0 or action_value < 0.0:
            return False, portfolio_state

        if portfolio_state.balance <= 1e-9 or current_price <= 1e-20:
            return False, portfolio_state

        gross_transaction_value_cash = portfolio_state.balance * action_value

        if gross_transaction_value_cash < self.min_trade_value:
            return False, portfolio_state

        transaction_cost = gross_transaction_value_cash * self.transaction_fee
        slippage_cost = gross_transaction_value_cash * self.slippage_bps * 1e-4
        total_cost = transaction_cost + slippage_cost

        net_transaction_value_cash = gross_transaction_value_cash - total_cost
        if net_transaction_value_cash <= 0:
            return False, portfolio_state
        position_change = net_transaction_value_cash / current_price

        new_balance = portfolio_state.balance - gross_transaction_value_cash
        new_position = portfolio_state.position + position_change
        new_total_cost = portfolio_state.total_transaction_cost + total_cost

        # Calculate position price as weighted average (cost basis)
        if portfolio_state.position <= 1e-9:
            # If no existing position, just use current price
            new_position_price = current_price
        else:
            # Calculate weighted average based on old and new position sizes
            new_position_price = (
                (portfolio_state.position * portfolio_state.position_price) + (position_change * current_price)
            ) / new_position

        return True, PortfolioState(
            balance=new_balance,
            position=new_position,
            position_price=new_position_price,
            total_transaction_cost=new_total_cost,
        )

    def handle_sell(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action_value: float,
    ) -> tuple[bool, PortfolioState]:
        """Handle sell action logic.

        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action_value: Fraction of position to sell (0 to 1)

        Returns:
            Tuple of (is_valid, new_portfolio_state)
        """
        if action_value > 1.0 or action_value < 0.0:
            return False, portfolio_state

        if portfolio_state.position <= 1e-9:
            return False, portfolio_state

        sell_amount_shares = portfolio_state.position * action_value
        gross_transaction_value_cash = sell_amount_shares * current_price

        if gross_transaction_value_cash < self.min_trade_value:
            return False, portfolio_state

        transaction_cost = gross_transaction_value_cash * self.transaction_fee
        slippage_cost = gross_transaction_value_cash * self.slippage_bps * 1e-4
        total_cost = transaction_cost + slippage_cost
        net_transaction_value_cash = gross_transaction_value_cash - total_cost

        if net_transaction_value_cash < 0:
            return False, portfolio_state

        new_balance = portfolio_state.balance + net_transaction_value_cash
        new_position = portfolio_state.position - sell_amount_shares
        new_total_cost = portfolio_state.total_transaction_cost + total_cost

        return True, PortfolioState(
            balance=new_balance,
            position=new_position,
            position_price=portfolio_state.position_price if new_position > 1e-9 else 0.0,
            total_transaction_cost=new_total_cost,
        )

    def apply_trade(
        self,
        portfolio_state: PortfolioState,
        current_price: float,
        action: int,
        action_value: float,
    ) -> tuple[PortfolioState, bool]:
        """Apply a trading action.

        Args:
            portfolio_state: Current portfolio state
            current_price: Current asset price
            action: Action type (0=Hold, 1=Buy, 2=Sell)
            action_value: Action value (0 to 1)

        Returns:
            Tuple of (New portfolio state, is_valid flag)
        """
        is_valid: bool = True
        new_state: PortfolioState = portfolio_state

        if action == 0:  # Hold
            # Hold is always valid, state doesn't change
            new_state = portfolio_state
            is_valid = True

        elif action == 1:  # Buy
            is_valid, temp_state = self.handle_buy(portfolio_state, current_price, action_value)
            # Only update state if the action was valid
            new_state = temp_state if is_valid else portfolio_state

        elif action == 2:  # Sell
            is_valid, temp_state = self.handle_sell(portfolio_state, current_price, action_value)
            # Only update state if the action was valid
            new_state = temp_state if is_valid else portfolio_state

        else:  # Unknown action
            is_valid = False  # Treat unknown actions as invalid
            new_state = portfolio_state  # State does not change

        return new_state, is_valid  # Return the state and validity

    def reset_peak(self, initial_value: float) -> None:
        """Reset peak portfolio value for drawdown tracking (call on env reset)."""
        self.peak_portfolio_value = initial_value

    def calculate_reward(
        self,
        prev_portfolio_value: float,
        pre_trade_portfolio_value: float,
        post_trade_portfolio_value: float,
        is_valid: bool,
        price_return: float,
        position_fraction: float,
    ) -> float:
        """Benchmark-relative reward: rewards excess return over a fixed-allocation baseline.

        Args:
            price_return: The asset's price return this step (close_t / close_{t-1} - 1).
            position_fraction: Fraction of portfolio in the asset (0 = all cash, 1 = fully invested).
        """
        if not is_valid:
            return self.invalid_action_penalty

        if prev_portfolio_value > 1e-9 and pre_trade_portfolio_value > 1e-9:
            market_return = (pre_trade_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            market_return = 0.0

        if pre_trade_portfolio_value > 1e-9 and post_trade_portfolio_value > 1e-9:
            trade_return = (post_trade_portfolio_value - pre_trade_portfolio_value) / pre_trade_portfolio_value
        else:
            trade_return = 0.0

        benchmark_return = self.benchmark_allocation_frac * price_return
        excess_return = (market_return + trade_return) - benchmark_return
        pnl_reward = self.reward_scale * excess_return

        self.peak_portfolio_value = max(self.peak_portfolio_value, post_trade_portfolio_value)
        drawdown = 0.0
        if self.peak_portfolio_value > 1e-9:
            drawdown = (self.peak_portfolio_value - post_trade_portfolio_value) / self.peak_portfolio_value

        cash_fraction = max(0.0, 1.0 - position_fraction)
        opportunity_cost = self.opportunity_cost_lambda * abs(price_return) * cash_fraction

        reward = pnl_reward - self.drawdown_penalty_lambda * drawdown - opportunity_cost
        return reward
