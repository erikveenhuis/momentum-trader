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


def test_calculate_reward(trading_logic, portfolio_state):
    """Test reward calculation."""
    # Define prices for the step
    previous_price = 100.0  # Assume price at the start of the step
    current_price = 100.0  # Default price at the end, override in specific cases
    reward_scale = trading_logic.reward_scale  # 500.0

    # Test reward for hold action (no change) - Initial state (no position)
    prev_portfolio_value = portfolio_state.portfolio_value(previous_price)
    cur_portfolio_value = portfolio_state.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value,
        cur_portfolio_value=cur_portfolio_value,
        is_valid=True,
    )
    assert reward == 0.0  # log(1) * scale = 0

    # Setup state with a position for subsequent hold tests
    state_with_position = PortfolioState(
        balance=5000.0,
        position=50.0,  # Holding 50 units
        position_price=100.0,  # Acquired at price 100
        total_transaction_cost=5.0,
    )

    # Test reward for hold action when price goes UP
    price_up = 110.0
    prev_portfolio_value_up = state_with_position.portfolio_value(previous_price)
    cur_portfolio_value_up = state_with_position.portfolio_value(price_up)
    reward_hold_up = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_up,
        cur_portfolio_value=cur_portfolio_value_up,
        is_valid=True,
    )
    expected_reward_hold_up = ((cur_portfolio_value_up - prev_portfolio_value_up) / prev_portfolio_value_up) * reward_scale
    assert reward_hold_up == pytest.approx(expected_reward_hold_up, rel=1e-9)

    # Test reward for hold action when price goes DOWN
    price_down = 90.0
    prev_portfolio_value_down = state_with_position.portfolio_value(previous_price)
    cur_portfolio_value_down = state_with_position.portfolio_value(price_down)
    reward_hold_down = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_down,
        cur_portfolio_value=cur_portfolio_value_down,
        is_valid=True,
    )
    expected_reward_hold_down = ((cur_portfolio_value_down - prev_portfolio_value_down) / prev_portfolio_value_down) * reward_scale
    assert reward_hold_down == pytest.approx(expected_reward_hold_down, rel=1e-9)

    # Test reward for invalid action
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value,
        cur_portfolio_value=cur_portfolio_value,
        is_valid=False,
    )
    assert reward == trading_logic.invalid_action_penalty  # -1.0

    # Test reward for a trade that results in the same value from start to end
    # (e.g., selling a position at the start price, ignoring fees for setup simplicity)
    old_state_same = PortfolioState(
        balance=10000.0,
        position=1.0,
        position_price=90.0,
        total_transaction_cost=0.0,
    )
    new_state_same = PortfolioState(  # Simulating selling the position for 100 cash
        balance=10100.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.1,
    )
    prev_portfolio_value_same = old_state_same.portfolio_value(previous_price)
    cur_portfolio_value_same = new_state_same.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_same,
        cur_portfolio_value=cur_portfolio_value_same,
        is_valid=True,
    )
    expected_reward_same = ((cur_portfolio_value_same - prev_portfolio_value_same) / prev_portfolio_value_same) * reward_scale
    assert reward == pytest.approx(expected_reward_same, rel=1e-9)

    # Test reward for profitable trade (portfolio value increases)
    old_state_gain = PortfolioState(
        balance=10000.0,
        position=1.0,
        position_price=90.0,
        total_transaction_cost=0.0,
    )
    new_state_gain = PortfolioState(  # Simulating selling for 110 cash
        balance=10110.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.1,
    )
    prev_portfolio_value_gain = old_state_gain.portfolio_value(previous_price)
    cur_portfolio_value_gain = new_state_gain.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_gain,
        cur_portfolio_value=cur_portfolio_value_gain,
        is_valid=True,
    )
    expected_reward_gain = ((cur_portfolio_value_gain - prev_portfolio_value_gain) / prev_portfolio_value_gain) * reward_scale
    assert reward == pytest.approx(expected_reward_gain, rel=1e-9)

    # Test reward for loss-making trade (portfolio value decreases)
    old_state_loss = PortfolioState(
        balance=10000.0,
        position=1.0,
        position_price=110.0,
        total_transaction_cost=0.0,
    )
    new_state_loss = PortfolioState(  # Simulating selling for 90 cash
        balance=10090.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.1,
    )
    prev_portfolio_value_loss = old_state_loss.portfolio_value(previous_price)
    cur_portfolio_value_loss = new_state_loss.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_loss,
        cur_portfolio_value=cur_portfolio_value_loss,
        is_valid=True,
    )
    expected_reward_loss = ((cur_portfolio_value_loss - prev_portfolio_value_loss) / prev_portfolio_value_loss) * reward_scale
    assert reward == pytest.approx(expected_reward_loss, rel=1e-9)

    # Test reward for very small portfolio value (should return 0 if no change)
    small_value_state = PortfolioState(
        balance=1e-9,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.0,
    )
    prev_portfolio_value_small = small_value_state.portfolio_value(previous_price)
    cur_portfolio_value_small = small_value_state.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_small,
        cur_portfolio_value=cur_portfolio_value_small,
        is_valid=True,
    )
    assert reward == 0.0  # log(1) * scale = 0

    # Test reward where portfolio value decreases slightly
    old_state_slight_loss = PortfolioState(
        balance=10000.0,
        position=0.1,
        position_price=95.0,
        total_transaction_cost=0.0,
    )
    new_state_slight_loss = PortfolioState(  # Simulating value dropping slightly
        balance=10005.0,
        position=0.0,
        position_price=0.0,
        total_transaction_cost=0.5,
    )
    prev_portfolio_value_slight_loss = old_state_slight_loss.portfolio_value(previous_price)
    cur_portfolio_value_slight_loss = new_state_slight_loss.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_slight_loss,
        cur_portfolio_value=cur_portfolio_value_slight_loss,
        is_valid=True,
    )
    expected_reward_slight_loss = (
        (cur_portfolio_value_slight_loss - prev_portfolio_value_slight_loss) / prev_portfolio_value_slight_loss
    ) * reward_scale
    assert reward == pytest.approx(expected_reward_slight_loss, rel=1e-9)

    # Test reward when start portfolio value is near zero (avoid log(inf))
    zero_start_state = PortfolioState(balance=1e-15, position=0.0, position_price=0.0, total_transaction_cost=0.0)
    non_zero_end_state = PortfolioState(balance=1.0, position=0.0, position_price=0.0, total_transaction_cost=0.0)
    prev_portfolio_value_zero = zero_start_state.portfolio_value(previous_price)
    cur_portfolio_value_zero = non_zero_end_state.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_zero,
        cur_portfolio_value=cur_portfolio_value_zero,
        is_valid=True,
    )
    assert reward == 0.0  # Should return 0 due to near-zero start value guard

    # Test reward when end portfolio value is near zero (avoid log(0))
    non_zero_start_state = PortfolioState(balance=1.0, position=0.0, position_price=0.0, total_transaction_cost=0.0)
    zero_end_state = PortfolioState(balance=1e-15, position=0.0, position_price=0.0, total_transaction_cost=0.0)
    prev_portfolio_value_zero_end = non_zero_start_state.portfolio_value(previous_price)
    cur_portfolio_value_zero_end = zero_end_state.portfolio_value(current_price)
    reward = trading_logic.calculate_reward(
        prev_portfolio_value=prev_portfolio_value_zero_end,
        cur_portfolio_value=cur_portfolio_value_zero_end,
        is_valid=True,
    )
    assert reward == 0.0  # Should return 0 due to near-zero end value guard


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
