"""Tests for the trading environment module."""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from momentum_env.config import TradingEnvConfig
from momentum_env.environment import TradingEnv


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(90, 110, 100),
            "high": np.random.uniform(100, 120, 100),
            "low": np.random.uniform(80, 100, 100),
            "close": np.random.uniform(90, 110, 100),
            "volume": np.random.uniform(1000, 2000, 100),
        },
        index=dates,
    )
    return data


@pytest.fixture
def profitable_trading_env(tmp_path):
    """Create an environment with deterministic profitable price movement."""
    data = pd.DataFrame(
        {
            "open": [100.0, 200.0, 200.0],
            "high": [100.0, 200.0, 200.0],
            "low": [100.0, 200.0, 200.0],
            "close": [100.0, 200.0, 200.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )

    data_path = tmp_path / "profit_data.csv"
    data.to_csv(data_path, index=False)

    config = TradingEnvConfig(
        data_path=str(data_path),
        window_size=2,
        initial_balance=1000.0,
        transaction_fee=0.001,
        reward_scale=500.0,
        render_mode=None,
    )

    return TradingEnv(config=config)


@pytest.fixture
def trading_env(sample_data, tmp_path):
    """Create a trading environment instance."""
    # Save sample data to temporary file
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path)

    config = TradingEnvConfig(
        data_path=str(data_path),
        window_size=10,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_scale=500.0,
        render_mode=None,
    )

    env = TradingEnv(config=config)
    return env


@pytest.fixture
def invalid_buy_trading_env(sample_data, tmp_path):
    """Create a trading environment instance with near-zero balance."""
    data_path = tmp_path / "test_data_low_bal.csv"
    sample_data.to_csv(data_path)

    config = TradingEnvConfig(
        data_path=str(data_path),
        window_size=10,
        initial_balance=1e-10,  # Near-zero balance
        transaction_fee=0.001,
        reward_scale=500.0,
        render_mode=None,
    )

    env = TradingEnv(config=config)
    return env


def test_environment_initialization(trading_env):
    """Test environment initialization."""
    assert isinstance(trading_env, gym.Env)
    assert trading_env.config.window_size == 10
    assert trading_env.config.initial_balance == 10000.0
    assert trading_env.config.transaction_fee == 0.001
    assert trading_env.config.reward_scale == 500.0


def test_action_space(trading_env):
    """Test action space configuration."""
    assert isinstance(trading_env.action_space, gym.spaces.Discrete)
    assert trading_env.action_space.n == 7  # Hold, Buy25%, Buy50%, Buy100%, Sell25%, Sell50%, Sell100%


def test_observation_space(trading_env):
    """Test observation space configuration."""
    assert isinstance(trading_env.observation_space, gym.spaces.Dict)

    # Market data features
    assert isinstance(trading_env.observation_space["market_data"], gym.spaces.Box)
    assert trading_env.observation_space["market_data"].shape == (10, 5)  # window_size x features

    # Account state features
    assert isinstance(trading_env.observation_space["account_state"], gym.spaces.Box)
    assert trading_env.observation_space["account_state"].shape == (2,)  # position, balance


def test_reset(trading_env):
    """Test environment reset."""
    observation, info = trading_env.reset()

    # Check observation structure
    assert isinstance(observation, dict)
    assert "market_data" in observation
    assert "account_state" in observation

    # Check observation shapes
    assert observation["market_data"].shape == (10, 5)  # window_size x features
    assert observation["account_state"].shape == (2,)  # position, balance

    # Check initial portfolio state
    assert info["balance"] == 10000.0  # Initial balance
    assert info["position"] == 0.0  # Initial position


def test_step(trading_env):
    """Test environment step."""
    trading_env.reset()

    # Test hold action
    observation, reward, terminated, truncated, info = trading_env.step(0)
    assert not terminated
    assert isinstance(reward, float)
    assert isinstance(observation, dict)
    assert isinstance(info, dict)

    # Test buy action
    observation, reward, terminated, truncated, info = trading_env.step(1)  # Buy 25%
    assert not terminated
    assert isinstance(reward, float)
    assert info["balance"] < 10000.0  # Balance should decrease
    assert info["position"] > 0.0  # Position should increase


def test_invalid_actions(trading_env):
    """Test invalid action handling."""
    trading_env.reset()
    initial_balance = trading_env.config.initial_balance  # Capture initial balance for comparison

    # Test selling with no position (Action 4: Sell 10%)
    observation, reward, terminated, truncated, info = trading_env.step(4)
    assert reward == trading_env.config.invalid_action_penalty
    assert info["balance"] == initial_balance  # Balance should be unchanged
    assert info["position"] == 0.0  # Position should be unchanged
    assert not terminated and not truncated  # Should not terminate/truncate on invalid action

    # Test selling with no position (Action 5: Sell 25%)
    observation, reward, terminated, truncated, info = trading_env.step(5)
    assert reward == trading_env.config.invalid_action_penalty
    assert info["balance"] == initial_balance  # Balance should be unchanged
    assert info["position"] == 0.0  # Position should be unchanged
    assert not terminated and not truncated

    # Test selling with no position (Action 6: Sell 100%)
    observation, reward, terminated, truncated, info = trading_env.step(6)
    assert reward == trading_env.config.invalid_action_penalty
    assert info["balance"] == initial_balance  # Balance should be unchanged
    assert info["position"] == 0.0  # Position should be unchanged
    assert not terminated and not truncated


def test_step_action_outcomes(trading_env):
    """Test the quantitative outcome of each valid buy/sell action."""
    config = trading_env.config
    fee = config.transaction_fee

    # --- Test Buy Actions ---
    obs, info = trading_env.reset()
    initial_balance = info["balance"]
    price_step0 = info["price"]

    # Action 1: Buy 10%
    trading_env.reset()  # Reset for clean state
    obs, reward, term, trunc, info = trading_env.step(1)  # Buy 10%
    buy_amount_gross_1 = initial_balance * 0.10  # Use 10% for action 1
    cost_1 = buy_amount_gross_1 * fee
    buy_amount_net_1 = buy_amount_gross_1 - cost_1
    expected_pos_1 = buy_amount_net_1 / price_step0
    expected_bal_1 = initial_balance - buy_amount_gross_1
    assert info["balance"] == pytest.approx(expected_bal_1)
    assert info["position"] == pytest.approx(expected_pos_1)
    assert info["step_transaction_cost"] == pytest.approx(cost_1)

    # Action 2: Buy 25%
    trading_env.reset()  # Reset for clean state
    obs, reward, term, trunc, info = trading_env.step(2)  # Buy 25%
    buy_amount_gross_2 = initial_balance * 0.25  # Use 25% for action 2
    cost_2 = buy_amount_gross_2 * fee
    buy_amount_net_2 = buy_amount_gross_2 - cost_2
    expected_pos_2 = buy_amount_net_2 / price_step0
    expected_bal_2 = initial_balance - buy_amount_gross_2
    assert info["balance"] == pytest.approx(expected_bal_2)
    assert info["position"] == pytest.approx(expected_pos_2)
    assert info["step_transaction_cost"] == pytest.approx(cost_2)

    # Action 3: Buy 50%
    trading_env.reset()  # Reset for clean state
    obs, reward, term, trunc, info = trading_env.step(3)  # Buy 50%
    buy_amount_gross_3 = initial_balance * 0.50  # Use 50% for action 3
    cost_3 = buy_amount_gross_3 * fee
    buy_amount_net_3 = buy_amount_gross_3 - cost_3
    expected_pos_3 = buy_amount_net_3 / price_step0
    expected_bal_3 = initial_balance - buy_amount_gross_3
    assert info["balance"] == pytest.approx(expected_bal_3)
    assert info["position"] == pytest.approx(expected_pos_3)
    assert info["step_transaction_cost"] == pytest.approx(cost_3)

    # --- Setup for Sell Actions ---
    # Perform a 100% buy to get a position to sell from
    trading_env.reset()
    obs, reward, term, trunc, info_buy = trading_env.step(3)
    bal_after_buy = info_buy["balance"]
    pos_after_buy = info_buy["position"]
    # Price used for the buy was price_step0 = trading_env.market_data.close_prices[0]
    # State is now at the beginning of step 1

    # --- Test Sell Actions ---

    # Action 4: Sell 10% (occurs during step 1)
    price_for_step1 = trading_env.market_data.close_prices[1]
    sell_amount_shares_4 = pos_after_buy * 0.10  # Use 10% for action 4
    sell_amount_gross_4 = sell_amount_shares_4 * price_for_step1
    cost_4 = sell_amount_gross_4 * fee
    sell_amount_net_4 = sell_amount_gross_4 - cost_4
    expected_pos_4 = pos_after_buy - sell_amount_shares_4
    expected_bal_4 = bal_after_buy + sell_amount_net_4

    obs, reward, term, trunc, info_sell4 = trading_env.step(4)
    assert info_sell4["balance"] == pytest.approx(expected_bal_4)
    assert info_sell4["position"] == pytest.approx(expected_pos_4)
    assert info_sell4["step_transaction_cost"] == pytest.approx(cost_4)
    # State is now at the beginning of step 2
    bal_after_sell4 = info_sell4["balance"]
    pos_after_sell4 = info_sell4["position"]

    # Action 5: Sell 25% (of remaining, occurs during step 2)
    price_for_step2 = trading_env.market_data.close_prices[2]
    sell_amount_shares_5 = pos_after_sell4 * 0.25  # Use 25% for action 5
    sell_amount_gross_5 = sell_amount_shares_5 * price_for_step2
    cost_5 = sell_amount_gross_5 * fee
    sell_amount_net_5 = sell_amount_gross_5 - cost_5
    expected_pos_5 = pos_after_sell4 - sell_amount_shares_5
    expected_bal_5 = bal_after_sell4 + sell_amount_net_5

    obs, reward, term, trunc, info_sell5 = trading_env.step(5)
    assert info_sell5["balance"] == pytest.approx(expected_bal_5)
    assert info_sell5["position"] == pytest.approx(expected_pos_5)
    assert info_sell5["step_transaction_cost"] == pytest.approx(cost_5)
    # State is now at the beginning of step 3
    bal_after_sell5 = info_sell5["balance"]
    pos_after_sell5 = info_sell5["position"]

    # Action 6: Sell 100% (of remaining, occurs during step 3)
    price_for_step3 = trading_env.market_data.close_prices[3]
    sell_amount_shares_6 = pos_after_sell5 * 1.00  # Use 100% for action 6
    sell_amount_gross_6 = sell_amount_shares_6 * price_for_step3
    cost_6 = sell_amount_gross_6 * fee
    sell_amount_net_6 = sell_amount_gross_6 - cost_6
    # expected_pos_6 = pos_after_sell5 - sell_amount_shares_6
    expected_bal_6 = bal_after_sell5 + sell_amount_net_6

    obs, reward, term, trunc, info_sell6 = trading_env.step(6)
    # Position might not be exactly zero due to float precision
    assert info_sell6["balance"] == pytest.approx(expected_bal_6)
    assert info_sell6["position"] == pytest.approx(0.0, abs=1e-9)
    assert info_sell6["step_transaction_cost"] == pytest.approx(cost_6)


def test_episode_termination(trading_env):
    """Test episode termination conditions."""
    trading_env.reset()

    # Run until portfolio depletion or dataset exhaustion
    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated) and step_count < 1000:
        _, _, terminated, truncated, _ = trading_env.step(0)  # Hold action
        step_count += 1

    assert truncated  # Dataset exhaustion should trigger truncation
    assert not terminated  # No portfolio depletion when continuously holding


def test_invalid_buy_actions(invalid_buy_trading_env):
    """Test invalid buy action handling (near-zero balance)."""
    env = invalid_buy_trading_env
    env.reset()
    initial_balance = env.config.initial_balance

    # Test buying with near-zero balance (Action 1: Buy 10%)
    observation, reward, terminated, truncated, info = env.step(1)
    assert reward == env.config.invalid_action_penalty
    assert info["balance"] == initial_balance
    assert info["position"] == 0.0
    assert not terminated and not truncated

    # Test buying with near-zero balance (Action 2: Buy 25%)
    observation, reward, terminated, truncated, info = env.step(2)
    assert reward == env.config.invalid_action_penalty
    assert info["balance"] == initial_balance
    assert info["position"] == 0.0
    assert not terminated and not truncated

    # Test buying with near-zero balance (Action 3: Buy 50%)
    observation, reward, terminated, truncated, info = env.step(3)
    assert reward == env.config.invalid_action_penalty
    assert info["balance"] == initial_balance
    assert info["position"] == 0.0
    assert not terminated and not truncated


def test_account_state_within_bounds(profitable_trading_env):
    """Ensure account_state observations stay within declared Box bounds."""
    env = profitable_trading_env
    account_space = env.observation_space["account_state"]

    observation, info = env.reset()
    assert account_space.contains(observation["account_state"])

    observation, reward, terminated, truncated, info = env.step(3)  # Buy 50%
    assert not terminated and not truncated
    assert account_space.contains(observation["account_state"])

    observation, reward, terminated, truncated, info = env.step(6)  # Sell 100%
    assert not terminated and not truncated
    assert info["balance"] > env.config.initial_balance
    assert account_space.contains(observation["account_state"])
