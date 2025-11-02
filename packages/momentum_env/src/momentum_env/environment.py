from typing import Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from momentum_env.config import TradingEnvConfig
from momentum_env.data import MarketDataProcessor, get_observation_at_step
from momentum_env.trading import PortfolioState, TradingLogic
from momentum_env.visualization import TradingVisualizer


class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning.

    Features:
    - Normalized OHLCV observations with configurable window size
    - Discrete action space: 0=Hold, 1=Buy10%, 2=Buy25%, 3=Buy50%, 4=Sell10%, 5=Sell25%, 6=Sell100%
    - Account state information (position and balance)
    - Configurable transaction fees and initial balance
    - Customizable reward function based on PnL and trading costs
    """

    metadata = {"render_modes": ["human", "terminal"], "render_fps": 60}

    def __init__(self, config: TradingEnvConfig, render_mode: Optional[str] = None) -> None:
        """Initialize the trading environment.

        Args:
            config: Environment configuration
            render_mode: The rendering mode (ignored if set in config)
        """
        super().__init__()

        # Store configuration
        self.config = config
        # Prioritize render_mode from config, otherwise use the argument (or None)
        # Check if config has render_mode attribute and it's not None
        if hasattr(config, "render_mode") and config.render_mode is not None:
            self.render_mode = config.render_mode
        else:
            self.render_mode = render_mode

        # Initialize components
        self.data_processor = MarketDataProcessor(window_size=config.window_size)
        self.trading_logic = TradingLogic(
            transaction_fee=config.transaction_fee,
            reward_scale=config.reward_scale,
            invalid_action_penalty=config.invalid_action_penalty,
        )

        # Load and process market data
        self.market_data = self.data_processor.load_and_process_data(self.config.data_path)

        # Initialize visualizer if needed
        self.visualizer = None
        if self.render_mode == "human":
            self.visualizer = TradingVisualizer(self.market_data)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)  # 0=Hold, 1=Buy10%, 2=Buy25%, 3=Buy50%, 4=Sell10%, 5=Sell25%, 6=Sell100%
        self.observation_space = spaces.Dict(
            {
                "market_data": spaces.Box(
                    low=0,  # Normalized features are in [0,1]
                    high=1,
                    shape=(config.window_size, self.market_data.num_features),
                    dtype=np.float32,
                ),
                "account_state": spaces.Box(
                    low=0,
                    high=1,
                    shape=(2,),
                    dtype=np.float32,  # Normalized position and balance are in [0,1]
                ),
            }
        )

        # Initialize state variables
        self.state = None
        self.portfolio_state = None
        self.prev_portfolio_value: Optional[float] = None
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset state
        self.state = {
            "current_step": 0,
        }

        # Reset portfolio state
        self.portfolio_state = PortfolioState(
            balance=self.config.initial_balance,
            position=0.0,
            position_price=0.0,
            total_transaction_cost=0.0,
        )

        # Get initial observation and info
        observation = self._get_observation()
        info = self._get_info()

        # Initialize previous portfolio value
        self.prev_portfolio_value = info["portfolio_value"]

        return observation, info

    def _get_info(self, step_override: Optional[int] = None) -> Dict[str, Union[int, float]]:
        """Get additional information about the specified step (defaults to current state)."""
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
            "action": None,  # Will be set in step method
            "step_transaction_cost": 0.0,  # Will be set in step method
        }

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment.

        Args:
            action: Trading action to execute (0=Hold, 1=Buy10%, 2=Buy25%, 3=Buy50%, 4=Sell10%, 5=Sell25%, 6=Sell100%)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.state["current_step"] >= self.market_data.data_length:
            raise RuntimeError("Episode is done, call reset() first")

        # Get current price
        current_price = self.market_data.close_prices[self.state["current_step"]]

        # Store old portfolio state for transaction cost info calculation
        old_portfolio_state = self.portfolio_state

        # Map action to type and value
        if action == 0:  # Hold
            action_type = 0
            action_value = 0.0
        elif 1 <= action <= 3:  # Buy (1=10%, 2=25%, 3=50%)
            action_type = 1
            action_value = {1: 0.10, 2: 0.25, 3: 0.50}[action]
        else:  # Sell (4=10%, 5=25%, 6=100%)
            action_type = 2
            action_value = {4: 0.10, 5: 0.25, 6: 1.00}[action]

        # Apply trading action and get validity flag
        new_portfolio_state, is_valid = self.trading_logic.apply_trade(
            portfolio_state=self.portfolio_state,
            current_price=current_price,
            action=action_type,
            action_value=action_value,
        )

        # Update portfolio state
        self.portfolio_state = new_portfolio_state

        # Calculate current portfolio value (after action, using current price)
        cur_portfolio_value = self.portfolio_state.portfolio_value(current_price)

        # Calculate reward using previous and current portfolio values and the validity flag
        reward = float(
            self.trading_logic.calculate_reward(
                prev_portfolio_value=self.prev_portfolio_value,
                cur_portfolio_value=cur_portfolio_value,
                is_valid=is_valid,
            )
        )

        # Determine next step index and done flags
        next_step = self.state["current_step"] + 1
        terminated, truncated = self._check_termination(cur_portfolio_value, next_step)

        # Build next observation and info from the upcoming step (clamped if beyond data end)
        observation = self._get_observation(step_override=next_step)
        info = self._get_info(step_override=next_step)
        info["action"] = action
        info["invalid_action"] = not is_valid
        info["step_transaction_cost"] = self.portfolio_state.total_transaction_cost - old_portfolio_state.total_transaction_cost

        # Advance pointer (clamp to end-of-data once exhausted)
        if next_step >= self.market_data.data_length:
            self.state["current_step"] = self.market_data.data_length
        else:
            self.state["current_step"] = next_step

        # Update previous portfolio value for the next step
        self.prev_portfolio_value = cur_portfolio_value

        return observation, reward, terminated, truncated, info

    def _get_observation(self, step_override: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Get the observation at the specified step (defaults to current state)."""
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
        )

    def _check_termination(self, current_portfolio_value: float, next_step: int) -> Tuple[bool, bool]:
        """Determine termination (portfolio depleted or data exhausted) and truncation flags."""
        data_finished = next_step >= self.market_data.data_length
        portfolio_depleted = current_portfolio_value < self.config.initial_balance * 0.01

        terminated = portfolio_depleted or data_finished
        truncated = data_finished and not portfolio_depleted

        return terminated, truncated

    def render(self) -> None:
        """Render the environment."""
        if self.render_mode is None:
            return

        info = self._get_info()

        if self.render_mode == "human":
            self.visualizer.update(info)
        elif self.render_mode == "terminal":
            print(f"Step {info['step']}: Price=${info['price']:.2f}, PV=${info['portfolio_value']:.2f}")

    def close(self) -> None:
        """Clean up environment resources."""
        if self.visualizer is not None:
            self.visualizer.close()
