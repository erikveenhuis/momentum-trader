"""Demo script that runs a random agent in the trading environment."""

import argparse
import os

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from momentum_core.logging import get_logger

from momentum_env.config import TradingEnvConfig

logger = get_logger(__name__)


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Run a random agent in the trading environment.")
    parser.add_argument("--skip-env-check", action="store_true", help="Skip the gymnasium environment check.")
    args = parser.parse_args()

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "tests", "fixtures", "2017-06-27_BTC-USD.csv")

    # Register and create environment
    gym.register(id="momentum-env-v0", entry_point="momentum_env.environment:TradingEnv")
    env = gym.make(
        "momentum-env-v0",
        config=TradingEnvConfig(
            data_path=data_path,
            window_size=60,
            initial_balance=10000.0,
            transaction_fee=0.001,
            reward_scale=1.0,
            invalid_action_penalty=-0.1,
            drawdown_penalty_lambda=0.5,
            slippage_bps=5.0,
            opportunity_cost_lambda=0.1,
            benchmark_allocation_frac=0.5,
            min_rebalance_pct=0.02,
            min_trade_value=1.0,
        ),
    )

    # Conditionally run check_env
    if not args.skip_env_check:
        logger.info("Checking environment validity...")
        check_env(env.unwrapped)
        logger.info("Environment check passed!")

    observation, info = env.reset()
    env.render()

    step = 0
    while True:
        step += 1
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)

        if not (terminated or truncated):
            env.render()

        if terminated or truncated:
            logger.info("Episode finished after %d steps!", step)
            logger.info("Final portfolio value: $%.2f", info["portfolio_value"])
            logger.info("Total transaction cost: $%.2f", info["transaction_cost"])
            break

    env.close()


if __name__ == "__main__":
    main()
