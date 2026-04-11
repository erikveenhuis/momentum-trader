"""Tests for the trading logic module."""

import pytest
from momentum_env.trading import PortfolioState, TradingLogic


@pytest.fixture
def portfolio_state():
    """Create a portfolio state instance."""
    return PortfolioState(
        balance=10000.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.0,
    )


@pytest.fixture
def trading_logic():
    """Create a trading logic instance matching production config."""
    return TradingLogic(
        transaction_fee=0.001,
        reward_scale=1.0,
        invalid_action_penalty=-0.1,
        drawdown_penalty_lambda=0.5,
        slippage_bps=5.0,
        opportunity_cost_lambda=0.1,
        min_trade_value=1.0,
    )


def test_portfolio_state_initialization(portfolio_state):
    """Test portfolio state initialization."""
    assert portfolio_state.balance == 10000.0
    assert portfolio_state.position == 0.0
    assert portfolio_state.position_price == 0.0
    assert portfolio_state.total_transaction_cost == 0.0


def test_portfolio_value(portfolio_state):
    """Test portfolio value calculation."""
    # Initial state
    assert portfolio_state.portfolio_value(current_price=100.0) == 10000.0

    # With position
    portfolio_state.position = 2.0
    portfolio_state.position_price = 100.0
    assert portfolio_state.portfolio_value(current_price=120.0) == 10240.0  # 10000 + 2 * (120-100)


def test_handle_buy(trading_logic, portfolio_state):
    """Test buy action handling with fee + slippage."""
    current_price = 100.0
    action_value = 0.5
    fee = trading_logic.transaction_fee
    slip = trading_logic.slippage_bps * 1e-4

    is_valid, new_state = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )

    gross = 10000.0 * 0.5
    total_cost = gross * (fee + slip)
    net = gross - total_cost
    expected_position = net / current_price

    assert is_valid
    assert new_state.balance == pytest.approx(5000.0)
    assert new_state.position == pytest.approx(expected_position)
    assert new_state.position_price == current_price
    assert new_state.total_transaction_cost == pytest.approx(total_cost)


def test_handle_sell(trading_logic, portfolio_state):
    """Test sell action handling with fee + slippage."""
    portfolio_state.position = 1.0
    portfolio_state.position_price = 90.0
    current_price = 100.0
    action_value = 0.5
    fee = trading_logic.transaction_fee
    slip = trading_logic.slippage_bps * 1e-4

    is_valid, new_state = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )

    gross = 0.5 * current_price
    total_cost = gross * (fee + slip)
    net = gross - total_cost

    assert is_valid
    assert new_state.balance == pytest.approx(10000.0 + net)
    assert new_state.position == pytest.approx(0.5)
    assert new_state.position_price == 90.0
    assert new_state.total_transaction_cost == pytest.approx(total_cost)


def test_invalid_actions(trading_logic, portfolio_state):
    """Test invalid action handling."""
    current_price = 100.0

    # Invalid buy (action_value > 1)
    is_valid, _ = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=1.5,
    )
    assert not is_valid

    # Invalid sell (no position)
    is_valid, _ = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=0.5,
    )
    assert not is_valid


def test_apply_trade(trading_logic, portfolio_state):
    """Test trade application."""
    current_price = 100.0

    # Test hold action
    new_state, is_valid = trading_logic.apply_trade(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action=0,  # Hold
        action_value=0.0,
    )
    assert is_valid  # Hold should always be valid
    assert new_state == portfolio_state  # State should be unchanged

    # Test buy action
    new_state, is_valid = trading_logic.apply_trade(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action=1,  # Buy
        action_value=0.5,
    )
    assert is_valid  # Buy should be valid here
    assert new_state.balance < portfolio_state.balance
    assert new_state.position > portfolio_state.position

    # Test sell action (after setting up a position)
    portfolio_state_with_pos = PortfolioState(
        balance=portfolio_state.balance,  # Start with original balance
        position=1.0,
        position_price=90.0,
        total_transaction_cost=0.0,
    )
    new_state, is_valid = trading_logic.apply_trade(
        portfolio_state=portfolio_state_with_pos,  # Use state with position
        current_price=current_price,
        action=2,  # Sell
        action_value=0.5,
    )
    assert is_valid  # Sell should be valid here
    assert new_state.balance > portfolio_state_with_pos.balance
    assert new_state.position < portfolio_state_with_pos.position


def test_calculate_reward_pnl(trading_logic):
    """Test PnL component of reward."""
    trading_logic.reset_peak(10000.0)

    reward = trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=10500.0,
        post_trade_portfolio_value=10500.0,
        is_valid=True,
        price_return=0.05,
        position_fraction=1.0,
    )
    pnl = trading_logic.reward_scale * 0.05
    assert reward > 0
    assert reward == pytest.approx(pnl, abs=0.01)


def test_calculate_reward_invalid(trading_logic):
    """Invalid actions return the penalty regardless of values."""
    trading_logic.reset_peak(10000.0)

    reward = trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=10000.0,
        post_trade_portfolio_value=10000.0,
        is_valid=False,
        price_return=0.0,
        position_fraction=0.0,
    )
    assert reward == trading_logic.invalid_action_penalty


def test_calculate_reward_drawdown_penalty(trading_logic):
    """Drawdown penalty fires when portfolio drops below peak."""
    trading_logic.reset_peak(10000.0)

    trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=10000.0,
        post_trade_portfolio_value=10000.0,
        is_valid=True,
        price_return=0.0,
        position_fraction=0.5,
    )

    reward = trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=9500.0,
        post_trade_portfolio_value=9500.0,
        is_valid=True,
        price_return=-0.05,
        position_fraction=0.5,
    )
    assert reward < 0


def test_calculate_reward_opportunity_cost(trading_logic):
    """Holding cash while market moves incurs opportunity cost."""
    trading_logic.reset_peak(10000.0)

    reward_in_cash = trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=10000.0,
        post_trade_portfolio_value=10000.0,
        is_valid=True,
        price_return=0.01,
        position_fraction=0.0,
    )

    trading_logic.reset_peak(10000.0)

    reward_fully_invested = trading_logic.calculate_reward(
        prev_portfolio_value=10000.0,
        pre_trade_portfolio_value=10100.0,
        post_trade_portfolio_value=10100.0,
        is_valid=True,
        price_return=0.01,
        position_fraction=1.0,
    )

    assert reward_in_cash < reward_fully_invested
    expected_opp_cost = trading_logic.opportunity_cost_lambda * 0.01 * 1.0
    assert reward_in_cash < 0
    assert abs(reward_in_cash) >= expected_opp_cost * 0.9


def test_position_price_calculation(trading_logic, portfolio_state):
    """Test position price calculation and updates."""
    current_price = 100.0

    # Test buying with no existing position
    is_valid, new_state = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=0.2,  # Buy 20% of available balance
    )
    assert is_valid
    assert new_state.position_price == current_price

    # Test buying more at a different price (should update to weighted average)
    portfolio_state = PortfolioState(
        balance=8000.0,
        position=20.0,
        position_price=100.0,
        total_transaction_cost=0.0,
    )
    new_price = 110.0
    is_valid, new_state = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=new_price,
        action_value=0.25,  # Buy 25% of available balance
    )
    assert is_valid

    fee = trading_logic.transaction_fee
    slip = trading_logic.slippage_bps * 1e-4
    gross = 8000.0 * 0.25
    net = gross * (1 - fee - slip)
    expected_shares = net / new_price
    expected_price = ((20.0 * 100.0) + (expected_shares * new_price)) / (20.0 + expected_shares)
    assert new_state.position_price == pytest.approx(expected_price, rel=1e-5)

    # Test selling part of position (position price should remain unchanged)
    portfolio_state = PortfolioState(
        balance=8000.0,
        position=20.0,
        position_price=100.0,
        total_transaction_cost=0.0,
    )
    sell_price = 120.0
    is_valid, new_state = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=sell_price,
        action_value=0.5,  # Sell 50% of position
    )
    assert is_valid
    assert new_state.position_price == 100.0  # Should remain unchanged

    # Test selling entire position (position price should be reset to 0)
    portfolio_state = PortfolioState(
        balance=8000.0,
        position=20.0,
        position_price=100.0,
        total_transaction_cost=0.0,
    )
    is_valid, new_state = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=sell_price,
        action_value=1.0,  # Sell 100% of position
    )
    assert is_valid
    assert new_state.position_price == 0.0  # Should be reset to 0
