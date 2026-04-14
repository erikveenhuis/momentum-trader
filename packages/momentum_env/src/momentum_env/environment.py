import gymnasium as gym
import numpy as np
from gymnasium import spaces

from momentum_env.config import TradingEnvConfig
from momentum_env.data import MarketDataProcessor, get_observation_at_step
from momentum_env.trading import PortfolioState, TradingLogic

TARGET_ALLOCATIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
NUM_ACTIONS = len(TARGET_ALLOCATIONS)


class TradingEnv(gym.Env):
    """Trading environment with target-allocation actions.

    Each action sets the desired portfolio exposure:
        0 = 0%   (all cash)
        1 = 20%
        2 = 40%
        3 = 60%
        4 = 80%
        5 = 100% (fully invested)

    The environment computes the buy/sell needed to reach the target.
    Every action is always valid -- no masking required.
    """

    def __init__(self, config: TradingEnvConfig) -> None:
        super().__init__()

        self.config = config

        self.data_processor = MarketDataProcessor(window_size=config.window_size)
        self.trading_logic = TradingLogic(
            transaction_fee=config.transaction_fee,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
            drawdown_penalty_lambda=config.drawdown_penalty_lambda,
            slippage_bps=config.slippage_bps,
            opportunity_cost_lambda=config.opportunity_cost_lambda,
            min_trade_value=config.min_trade_value,
            benchmark_allocation_frac=config.benchmark_allocation_frac,
        )
        self.min_rebalance_pct = config.min_rebalance_pct

        self.market_data = self.data_processor.load_and_process_data(self.config.data_path)

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "market_data": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(config.window_size, self.market_data.num_features),
                    dtype=np.float32,
                ),
                "account_state": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(5,),
                    dtype=np.float32,
                ),
            }
        )

        self.state = None
        self.portfolio_state = None
        self.prev_portfolio_value: float | None = None
        self.bars_in_position: int = 0
        self.reset()

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)

        if options and "data_path" in options:
            new_path = str(options["data_path"])
            new_market_data = self.data_processor.load_and_process_data(new_path)
            if new_market_data.num_features != self.market_data.num_features:
                raise ValueError(
                    f"Feature count mismatch: expected {self.market_data.num_features}, got {new_market_data.num_features} from {new_path}"
                )
            self.market_data = new_market_data

        self.state = {"current_step": 0}
        self.portfolio_state = PortfolioState(
            balance=self.config.initial_balance,
            position=0.0,
            position_price=0.0,
            total_transaction_cost=0.0,
        )
        self.bars_in_position = 0

        observation = self._get_observation()
        info = self._get_info()

        self.prev_portfolio_value = info["portfolio_value"]
        self.trading_logic.reset_peak(info["portfolio_value"])

        return observation, info

    def _get_info(self, step_override: int | None = None) -> dict[str, int | float]:
        target_step = self.state["current_step"] if step_override is None else step_override
        target_step = min(target_step, self.market_data.data_length - 1)
        current_price = self.market_data.close_prices[target_step]
        portfolio_value = self.portfolio_state.portfolio_value(current_price)

        return {
            "step": target_step,
            "price": current_price,
            "balance": self.portfolio_state.balance,
            "position": self.portfolio_state.position,
            "portfolio_value": portfolio_value,
            "transaction_cost": self.portfolio_state.total_transaction_cost,
            "action": None,
            "step_transaction_cost": 0.0,
        }

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        """Execute one step. Action is a target allocation index (0-5)."""
        if self.state["current_step"] >= self.market_data.data_length:
            raise RuntimeError("Episode is done, call reset() first")

        current_price = self.market_data.close_prices[self.state["current_step"]]

        old_portfolio_state = self.portfolio_state
        pre_trade_portfolio_value = old_portfolio_state.portfolio_value(current_price)

        target_frac = TARGET_ALLOCATIONS[action]
        self.portfolio_state, is_valid = self._execute_target_allocation(target_frac, current_price, pre_trade_portfolio_value)

        post_trade_portfolio_value = self.portfolio_state.portfolio_value(current_price)

        prev_step = max(0, self.state["current_step"] - 1)
        prev_price = self.market_data.close_prices[prev_step]
        price_return = (current_price / prev_price - 1.0) if prev_price > 1e-20 else 0.0

        position_value = max(0.0, old_portfolio_state.position * current_price)
        position_fraction = position_value / max(pre_trade_portfolio_value, 1e-9)

        reward = float(
            self.trading_logic.calculate_reward(
                prev_portfolio_value=self.prev_portfolio_value,
                pre_trade_portfolio_value=pre_trade_portfolio_value,
                post_trade_portfolio_value=post_trade_portfolio_value,
                is_valid=is_valid,
                price_return=price_return,
                position_fraction=position_fraction,
            )
        )

        next_step = self.state["current_step"] + 1
        terminated, truncated = self._check_termination(post_trade_portfolio_value, next_step)

        observation = self._get_observation(step_override=next_step)
        info = self._get_info(step_override=next_step)
        info["action"] = action
        info["target_allocation"] = target_frac
        info["invalid_action"] = not is_valid
        info["step_transaction_cost"] = self.portfolio_state.total_transaction_cost - old_portfolio_state.total_transaction_cost

        if next_step >= self.market_data.data_length:
            self.state["current_step"] = self.market_data.data_length
        else:
            self.state["current_step"] = next_step

        self.prev_portfolio_value = post_trade_portfolio_value

        if self.portfolio_state.position > 1e-9:
            self.bars_in_position += 1
        else:
            self.bars_in_position = 0

        return observation, reward, terminated, truncated, info

    def _execute_target_allocation(self, target_frac: float, current_price: float, portfolio_value: float) -> tuple[PortfolioState, bool]:
        """Adjust position to reach the target allocation fraction.

        Returns (new_portfolio_state, is_valid). All targets are always valid;
        is_valid=False only if a trade is needed but below min notional.
        """
        if current_price <= 1e-20 or portfolio_value <= 1e-9:
            return self.portfolio_state, True

        current_position_value = max(0.0, self.portfolio_state.position * current_price)
        current_frac = current_position_value / portfolio_value
        delta_frac = target_frac - current_frac

        if abs(delta_frac) < self.min_rebalance_pct:
            return self.portfolio_state, True

        target_position_value = target_frac * portfolio_value
        delta_value = target_position_value - current_position_value

        if delta_frac > 0:
            buy_cash = min(delta_value, self.portfolio_state.balance)
            if buy_cash / portfolio_value < self.min_rebalance_pct:
                return self.portfolio_state, True
            buy_fraction = buy_cash / max(self.portfolio_state.balance, 1e-9)
            is_valid, new_state = self.trading_logic.handle_buy(self.portfolio_state, current_price, buy_fraction)
            return new_state, is_valid
        else:
            sell_value = abs(delta_value)
            if current_position_value < 1e-9:
                return self.portfolio_state, True
            sell_fraction = min(sell_value / current_position_value, 1.0)
            is_valid, new_state = self.trading_logic.handle_sell(self.portfolio_state, current_price, sell_fraction)
            return new_state, is_valid

    def _get_observation(self, step_override: int | None = None) -> dict[str, np.ndarray]:
        target_step = self.state["current_step"] if step_override is None else step_override
        target_step = min(target_step, self.market_data.data_length - 1)
        current_price = self.market_data.close_prices[target_step]
        return get_observation_at_step(
            market_data=self.market_data,
            step=target_step,
            position=self.portfolio_state.position,
            balance=self.portfolio_state.balance,
            initial_balance=self.config.initial_balance,
            current_price=current_price,
            position_price=self.portfolio_state.position_price,
            bars_in_position=self.bars_in_position,
            cumulative_fees_frac=self.portfolio_state.total_transaction_cost,
        )

    def _check_termination(self, current_portfolio_value: float, next_step: int) -> tuple[bool, bool]:
        data_finished = next_step >= self.market_data.data_length
        portfolio_depleted = current_portfolio_value < self.config.initial_balance * 0.01

        terminated = bool(portfolio_depleted)
        truncated = bool(data_finished and not terminated)

        return terminated, truncated
