import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
from .utils.logging_config import get_logger

# Get logger instance
logger = get_logger("Metrics")


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio of returns."""
    assert isinstance(returns, list), "Input returns must be a list"
    assert all(
        isinstance(r, (float, np.float32, np.float64)) for r in returns
    ), "All elements in returns must be floats"
    assert (
        isinstance(risk_free_rate, float) and 0.0 <= risk_free_rate < 1.0
    ), "Risk-free rate must be a float between 0 and 1"
    if not returns:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)
    if len(excess_returns) < 2:
        return 0.0

    std_dev = np.std(excess_returns)
    assert std_dev >= 0, "Standard deviation cannot be negative"

    # Handle cases with zero or near-zero standard deviation
    if std_dev < 1e-9:  # If std dev is effectively zero
        return 0.0  # Sharpe ratio is undefined or zero

    # Calculate Sharpe ratio (original logic with epsilon is fine here since std_dev > 0)
    sharpe = np.mean(excess_returns) / (std_dev + 1e-8)
    assert not np.isnan(sharpe) and not np.isinf(
        sharpe
    ), f"Sharpe ratio calculation resulted in NaN/Inf: {sharpe}"
    return sharpe


def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate the maximum drawdown from portfolio values."""
    assert isinstance(portfolio_values, list), "Input portfolio_values must be a list"
    assert all(
        isinstance(v, (float, np.float32, np.float64)) for v in portfolio_values
    ), "All elements in portfolio_values must be floats"
    if not portfolio_values:
        return 0.0

    portfolio_values = np.array(portfolio_values)
    assert np.all(portfolio_values >= 0), "Portfolio values cannot be negative"
    peak = np.maximum.accumulate(portfolio_values)
    # Add epsilon to peak to prevent division by zero if peak is 0
    drawdown = (peak - portfolio_values) / (peak + 1e-9)
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
    assert (
        0.0 <= max_dd <= 1.0
    ), f"Max drawdown calculation resulted in value outside [0, 1]: {max_dd}"
    return max_dd


def calculate_avg_trade_return(returns: List[float]) -> float:
    """Calculate the average return per trade."""
    assert isinstance(returns, list), "Input returns must be a list"
    assert all(
        isinstance(r, (float, np.float32, np.float64)) for r in returns
    ), "All elements in returns must be floats"
    if not returns:
        return 0.0

    avg_return = np.mean(returns)
    assert not np.isnan(avg_return) and not np.isinf(
        avg_return
    ), f"Avg trade return calculation resulted in NaN/Inf: {avg_return}"
    return avg_return


def calculate_episode_score(metrics: Dict[str, float]) -> float:
    """
    Calculate a normalized episode-level score focusing on risk-adjusted return and drawdown.

    Metrics (Sharpe Ratio, Total Return) are normalized using a sigmoid function
    to map them to the (0, 1) range before weighting.
    This score is designed for evaluating a single episode (e.g., one backtest run)
    and excludes metrics like win_rate that focus on intra-episode step performance.
    """
    assert isinstance(metrics, dict), "Input metrics must be a dictionary"
    if not metrics:
        return 0.0

    # Define weights for each metric
    weights = {
        "sharpe_ratio": 0.40,  # Weight for Sharpe Ratio
        "total_return": 0.20,  # Weight for Total Return
        "max_drawdown": 0.40,  # Weight for (1 - Max Drawdown)
    }
    # Ensure weights sum to 1 (or very close due to floating point)
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

    score = 0.0
    sharpe = metrics.get("sharpe_ratio", 0.0)
    # Convert total_return from percentage to decimal
    total_return_pct = metrics.get("total_return", 0.0) / 100.0
    # Default max_drawdown to 1.0 (worst case) if not provided
    max_drawdown = metrics.get("max_drawdown", 1.0)

    # --- Normalization using Sigmoid: 1 / (1 + exp(-x)) ---
    # Introduce scaling factors for Sharpe and Total Return to increase sensitivity
    k_sharpe = 10.0  # Scaling factor for Sharpe Ratio
    k_return = 30.0  # Scaling factor for Total Return

    # Maps values to (0, 1). 0 -> 0.5, positive -> >0.5, negative -> <0.5
    normalized_sharpe = 1.0 / (1.0 + np.exp(-k_sharpe * sharpe))
    normalized_total_return = 1.0 / (1.0 + np.exp(-k_return * total_return_pct))
    # (1 - max_drawdown) is already effectively normalized in [0, 1]
    normalized_drawdown_complement = 1.0 - max_drawdown

    # --- Weighted sum calculation using normalized values ---
    score += normalized_sharpe * weights["sharpe_ratio"]
    score += normalized_total_return * weights["total_return"]
    score += normalized_drawdown_complement * weights["max_drawdown"]

    # Clamp score to handle potential floating point inaccuracies near 0 or 1
    score = np.clip(score, 0.0, 1.0)

    # Final score will be between 0 and 1 because inputs are normalized and weights sum to 1
    # --- Separated Assertions for Debugging --- 
    assert isinstance(score, (float, np.floating)), f"Score type is not float or np.floating: {type(score)}"
    assert not np.isnan(score), f"Score calculation resulted in NaN: {score}"
    assert not np.isinf(score), f"Score calculation resulted in Inf: {score}"
    assert 0.0 <= score <= 1.0, f"Score calculation out of range [0,1]: {score}"
    return score


class PerformanceTracker:
    """Track performance metrics over time."""

    def __init__(self, window_size: int = 100):
        assert (
            isinstance(window_size, int) and window_size > 0
        ), "Window size must be a positive integer"
        self.window_size = window_size
        self.portfolio_values = []
        self.returns = []
        self.actions = []
        self.transaction_costs = []
        self.rewards = []

    def add_initial_value(self, initial_value: float):
        """Adds the initial portfolio value before the first step."""
        assert (
            isinstance(initial_value, (float, np.float32, np.float64))
            and initial_value >= 0
        ), "Invalid initial portfolio value"
        if not self.portfolio_values:  # Only add if list is empty
            self.portfolio_values.append(initial_value)
        else:
            logger.warning(
                "Attempted to add initial value when portfolio history already exists."
            )

    def update(
        self,
        portfolio_value: float,
        action: float,
        reward: float,
        transaction_cost: float = 0.0,
    ):
        """Update tracker with new values."""
        assert (
            isinstance(portfolio_value, (float, np.float32, np.float64))
            and portfolio_value >= 0
        ), "Invalid portfolio value"
        assert isinstance(
            action, (int, float, np.integer, np.float32, np.float64)
        ), "Invalid action type"
        assert isinstance(
            reward, (float, np.float32, np.float64)
        ), "Invalid reward type"
        assert (
            isinstance(transaction_cost, (float, np.float32, np.float64))
            and transaction_cost >= 0
        ), "Invalid transaction cost"
        assert not np.isnan(reward) and not np.isinf(reward), "Reward is NaN or Inf"

        self.portfolio_values.append(portfolio_value)
        self.actions.append(action)
        self.rewards.append(reward)
        self.transaction_costs.append(transaction_cost)

        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2]
            assert prev_value >= 0, "Previous portfolio value was negative"
            ret = portfolio_value / (prev_value + 1e-9) - 1
            assert not np.isnan(ret) and not np.isinf(
                ret
            ), "Return calculation resulted in NaN/Inf"
            self.returns.append(ret)

    def get_action_counts(self) -> Dict[int, int]:
        """Counts the occurrences of each discrete action taken."""
        counts = {i: 0 for i in range(7)} # Assuming 7 discrete actions (0-6)
        for action in self.actions:
            action_int = int(action) # Ensure action is integer index
            if 0 <= action_int < 7:
                counts[action_int] += 1
            else:
                logger.warning(f"Encountered invalid action index {action} in tracker.")
        return counts

    def get_metrics(self) -> Dict[str, float]:
        """Calculate all metrics from current data."""
        if not self.portfolio_values:
            return {}

        assert (
            len(self.portfolio_values) > 0
        ), "Cannot get metrics with empty portfolio history"
        assert (
            self.portfolio_values[0] > 0
        ), "Initial portfolio value must be positive to calculate return"

        metrics = {
            "initial_portfolio_value": self.portfolio_values[0],
            "portfolio_value": self.portfolio_values[-1],
            "total_return": (self.portfolio_values[-1] / self.portfolio_values[0] - 1)
            * 100,
            "sharpe_ratio": calculate_sharpe_ratio(self.returns),
            "max_drawdown": calculate_max_drawdown(self.portfolio_values),
            "avg_trade_return": calculate_avg_trade_return(self.returns),
            "transaction_costs": sum(self.transaction_costs) if self.transaction_costs else 0.0,
            "avg_reward": np.mean(self.rewards) if self.rewards else 0.0,
            "action_counts": self.get_action_counts(),
        }
        assert all(
            isinstance(v, (float, np.float32, np.float64, dict)) for v in metrics.values()
        ), "Not all calculated metrics are floats or dict"
        assert 0.0 <= metrics["max_drawdown"] <= 1.0, "Max drawdown out of range [0, 1]"
        # Check for NaN/Inf only in numeric values
        assert not any(
            np.isnan(v) or np.isinf(v) 
            for k, v in metrics.items() 
            if isinstance(v, (float, np.number)) # Check only numeric types
        ), "NaN or Inf found in calculated numeric metrics"
        return metrics

    def get_recent_metrics(self) -> Dict[str, float]:
        """Calculate metrics using only recent data."""
        if len(self.portfolio_values) < 2:
            return self.get_metrics()

        effective_window = min(self.window_size, len(self.portfolio_values))
        if effective_window < 2:
            return self.get_metrics()

        recent_portfolio = self.portfolio_values[-effective_window:]
        recent_returns = self.returns[-(effective_window - 1) :]
        recent_actions = self.actions[-effective_window:]
        recent_rewards = self.rewards[-effective_window:]
        recent_costs = self.transaction_costs[-effective_window:]

        assert (
            len(recent_portfolio) == effective_window
        ), "Recent portfolio length mismatch"
        assert (
            len(recent_returns) == effective_window - 1
        ), "Recent returns length mismatch"
        assert (
            recent_portfolio[0] > 0
        ), "Initial value in recent portfolio window must be positive"

        metrics = {
            "portfolio_value": recent_portfolio[-1],
            "total_return": (recent_portfolio[-1] / recent_portfolio[0] - 1) * 100,
            "sharpe_ratio": calculate_sharpe_ratio(recent_returns),
            "max_drawdown": calculate_max_drawdown(recent_portfolio),
            "avg_trade_return": calculate_avg_trade_return(recent_returns),
            "transaction_costs": sum(recent_costs) if recent_costs else 0.0,
            "avg_reward": np.mean(recent_rewards),
        }
        assert all(
            isinstance(v, (float, np.float32, np.float64)) for v in metrics.values()
        ), "Not all calculated recent metrics are floats"
        assert (
            0.0 <= metrics["max_drawdown"] <= 1.0
        ), "Recent max drawdown out of range [0, 1]"
        assert not any(
            np.isnan(v) or np.isinf(v) for v in metrics.values()
        ), "NaN or Inf found in calculated recent metrics"
        return metrics

    def get_improvement_rate(self) -> float:
        """Calculate the rate of improvement in recent performance using linear regression slope."""
        if len(self.portfolio_values) < 2:
            return 0.0

        recent_values = np.array(self.portfolio_values[-self.window_size :])
        if len(recent_values) < 2:
            return 0.0

        x = np.arange(len(recent_values))
        try:
            slope, intercept = np.polyfit(x, recent_values, 1)
        except (np.linalg.LinAlgError, ValueError):
            logger.warning("Could not fit linear regression for improvement rate.")
            return 0.0

        initial_val = recent_values[0]
        if initial_val <= 1e-9:
            return 0.0

        improvement_rate = slope / initial_val
        assert not np.isnan(improvement_rate) and not np.isinf(
            improvement_rate
        ), "Improvement rate calculation resulted in NaN/Inf"
        return improvement_rate

    def get_stability(self) -> float:
        """Calculate the stability of recent performance (lower standard deviation of returns is more stable)."""
        if len(self.returns) < 2:
            return 0.0

        recent_returns = np.array(self.returns[-self.window_size :])
        if len(recent_returns) < 2:
            return 0.0

        std_dev = np.std(recent_returns)
        assert std_dev >= 0, "Standard deviation of returns cannot be negative"
        stability = 1.0 / (1.0 + std_dev + 1e-9)
        assert (
            0.0 <= stability <= 1.0
        ), f"Stability calculation resulted in value outside [0, 1]: {stability}"
        return stability
