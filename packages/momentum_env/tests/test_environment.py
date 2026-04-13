"""Tests for the trading environment module."""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from momentum_env.config import TradingEnvConfig
from momentum_env.environment import NUM_ACTIONS, TARGET_ALLOCATIONS, TradingEnv


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="h")
    np.random.seed(42)
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
        invalid_action_penalty=-0.1,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        benchmark_allocation_frac=0.5,
        min_rebalance_pct=0.02,
        min_trade_value=1.0,
    )

    return TradingEnv(config=config)


@pytest.fixture
def trading_env(sample_data, tmp_path):
    data_path = tmp_path / "test_data.csv"
    sample_data.to_csv(data_path)

    config = TradingEnvConfig(
        data_path=str(data_path),
        window_size=10,
        initial_balance=10000.0,
        transaction_fee=0.001,
        reward_scale=500.0,
        invalid_action_penalty=-0.1,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        benchmark_allocation_frac=0.5,
        min_rebalance_pct=0.02,
        min_trade_value=1.0,
    )

    return TradingEnv(config=config)


def test_environment_initialization(trading_env):
    assert isinstance(trading_env, gym.Env)
    assert trading_env.config.window_size == 10
    assert trading_env.config.initial_balance == 10000.0


def test_action_space(trading_env):
    assert isinstance(trading_env.action_space, gym.spaces.Discrete)
    assert trading_env.action_space.n == NUM_ACTIONS


def test_observation_space(trading_env):
    assert isinstance(trading_env.observation_space, gym.spaces.Dict)
    assert isinstance(trading_env.observation_space["market_data"], gym.spaces.Box)
    assert trading_env.observation_space["market_data"].shape[0] == 10
    assert isinstance(trading_env.observation_space["account_state"], gym.spaces.Box)
    assert trading_env.observation_space["account_state"].shape == (5,)


def test_reset(trading_env):
    observation, info = trading_env.reset()

    assert isinstance(observation, dict)
    assert "market_data" in observation
    assert "account_state" in observation
    assert observation["market_data"].shape[0] == 10
    assert observation["account_state"].shape == (5,)
    assert info["balance"] == 10000.0
    assert info["position"] == 0.0


def test_step_hold(trading_env):
    """Action 0 = target 0% = stay in cash, no trade."""
    trading_env.reset()
    obs, reward, terminated, truncated, info = trading_env.step(0)
    assert not terminated
    assert isinstance(reward, float)
    assert info["position"] == 0.0
    assert info["balance"] == pytest.approx(10000.0)


def test_step_buy_target(trading_env):
    """Action 5 = target 100% = buy with all cash."""
    trading_env.reset()
    obs, reward, term, trunc, info = trading_env.step(5)
    assert not term
    assert info["balance"] < 10000.0
    assert info["position"] > 0.0


def test_step_sell_back(trading_env):
    """Go to 100% then back to 0% = sell everything."""
    trading_env.reset()

    obs, reward, term, trunc, info_buy = trading_env.step(5)
    assert info_buy["position"] > 0.0

    obs, reward, term, trunc, info_sell = trading_env.step(0)
    assert info_sell["position"] == pytest.approx(0.0, abs=1e-9)
    assert info_sell["balance"] > 0.0


def test_target_allocation_adjusts_position(trading_env):
    """Going from 0% to 60% then to 20% should reduce position."""
    trading_env.reset()

    obs, _, _, _, info_60 = trading_env.step(3)
    pos_60 = info_60["position"]
    assert pos_60 > 0.0

    obs, _, _, _, info_20 = trading_env.step(1)
    pos_20 = info_20["position"]
    assert pos_20 < pos_60


def test_all_actions_valid(trading_env):
    """Every target allocation action should be valid (no invalid action penalty)."""
    for action in range(NUM_ACTIONS):
        trading_env.reset()
        obs, reward, term, trunc, info = trading_env.step(action)
        assert not info.get("invalid_action", False), f"Action {action} ({TARGET_ALLOCATIONS[action] * 100}%) was invalid"


def test_hold_target_no_trade(trading_env):
    """Selecting 0% target when already at 0% should result in no trade cost."""
    trading_env.reset()

    _, _, _, _, info = trading_env.step(0)
    cost = info["step_transaction_cost"]

    assert cost == pytest.approx(0.0, abs=1e-6)


def test_episode_termination(trading_env):
    trading_env.reset()
    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated) and step_count < 1000:
        _, _, terminated, truncated, _ = trading_env.step(0)
        step_count += 1

    assert truncated
    assert not terminated


def test_account_state_within_bounds(profitable_trading_env):
    env = profitable_trading_env
    account_space = env.observation_space["account_state"]

    observation, info = env.reset()
    assert account_space.contains(observation["account_state"])

    observation, reward, terminated, truncated, info = env.step(5)
    assert not terminated and not truncated
    assert account_space.contains(observation["account_state"])

    observation, reward, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert info["balance"] > env.config.initial_balance
    assert account_space.contains(observation["account_state"])


def test_profitable_roundtrip(profitable_trading_env):
    """Buy at 100, price goes to 200, sell -- should profit."""
    env = profitable_trading_env
    env.reset()

    obs, reward, term, trunc, info_buy = env.step(5)
    assert info_buy["position"] > 0.0

    obs, reward, term, trunc, info_sell = env.step(0)
    assert info_sell["balance"] > env.config.initial_balance
