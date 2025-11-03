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
    """Create a trading logic instance."""
    return TradingLogic(transaction_fee=0.001, reward_scale=500.0, invalid_action_penalty=-1.0)


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
    """Test buy action handling."""
    current_price = 100.0
    action_value = 0.5  # Buy 50% of available balance

    # Execute buy
    is_valid, new_state = trading_logic.handle_buy(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )

    assert is_valid
    assert new_state.balance == pytest.approx(5000.0, rel=1e-10)  # 10000 * 0.5
    assert new_state.position == pytest.approx(49.95, rel=1e-10)  # (10000 * 0.5 * (1 - 0.001)) / 100
    assert new_state.position_price == current_price
    assert new_state.total_transaction_cost == pytest.approx(5.0, rel=1e-10)  # 5000 * 0.001


def test_handle_sell(trading_logic, portfolio_state):
    """Test sell action handling."""
    # Setup initial position
    portfolio_state.position = 1.0
    portfolio_state.position_price = 90.0
    current_price = 100.0
    action_value = 0.5  # Sell 50% of position

    # Execute sell
    is_valid, new_state = trading_logic.handle_sell(
        portfolio_state=portfolio_state,
        current_price=current_price,
        action_value=action_value,
    )

    assert is_valid
    expected_sell_value = 50.0  # 100 * 0.5
    expected_transaction_cost = expected_sell_value * 0.001

    assert new_state.balance == pytest.approx(10000.0 + expected_sell_value - expected_transaction_cost, rel=1e-10)
    assert new_state.position == pytest.approx(0.5, rel=1e-10)
    assert new_state.position_price == 90.0  # Unchanged
    assert new_state.total_transaction_cost == pytest.approx(expected_transaction_cost, rel=1e-10)


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


def test_calculate_reward(trading_logic):
    """Test reward calculation with market and trade components."""
    reward_scale = trading_logic.reward_scale

    # No market move, no trade impact -> zero reward
    prev_value = 10000.0
    pre_trade_value = 10000.0
    post_trade_value = 10000.0
    reward = trading_logic.calculate_reward(prev_value, pre_trade_value, post_trade_value, True)
    assert reward == 0.0

    # Positive market move before trade should be rewarded
    prev_value = 10000.0
    pre_trade_value = 10500.0  # 5% gain before action
    post_trade_value = 10500.0
    expected = ((pre_trade_value - prev_value) / prev_value) * reward_scale
    reward = trading_logic.calculate_reward(prev_value, pre_trade_value, post_trade_value, True)
    assert reward == pytest.approx(expected, rel=1e-9)

    # Negative market move before trade should be penalized
    prev_value = 10000.0
    pre_trade_value = 9500.0  # 5% loss before action
    post_trade_value = 9500.0
    expected = ((pre_trade_value - prev_value) / prev_value) * reward_scale
    reward = trading_logic.calculate_reward(prev_value, pre_trade_value, post_trade_value, True)
    assert reward == pytest.approx(expected, rel=1e-9)

    # Trade impact (e.g., transaction cost) should be reflected after market move
    prev_value = 10000.0
    pre_trade_value = 10000.0
    post_trade_value = 9980.0  # -0.2% from trade cost/slippage
    expected = ((pre_trade_value - prev_value) / prev_value + (post_trade_value - pre_trade_value) / pre_trade_value) * reward_scale
    reward = trading_logic.calculate_reward(prev_value, pre_trade_value, post_trade_value, True)
    assert reward == pytest.approx(expected, rel=1e-9)

    # Invalid actions should return the configured penalty regardless of values
    reward = trading_logic.calculate_reward(prev_value, pre_trade_value, post_trade_value, False)
    assert reward == trading_logic.invalid_action_penalty

    # Guard against extremely small portfolio values (should fall back to zero reward)
    tiny_value = 5e-10
    reward = trading_logic.calculate_reward(tiny_value, tiny_value, 1.0, True)
    assert reward == 0.0


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

    # Initial position: 20 shares at $100
    # New purchase: ~18.16 shares at $110 (2000 * 0.999 / 110)
    # Expected weighted avg: (20*100 + 18.16*110) / (20+18.16)
    expected_shares = 2000 * 0.999 / 110
    expected_price = ((20.0 * 100.0) + (expected_shares * 110.0)) / (20.0 + expected_shares)
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
