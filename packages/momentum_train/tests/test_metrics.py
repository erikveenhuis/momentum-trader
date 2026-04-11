"""Tests for the metrics module."""

import numpy as np
import pytest
from momentum_train.metrics import (
    PerformanceTracker,
    calculate_avg_trade_return,
    calculate_episode_score,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


class TestSharpeRatio:
    def test_positive_returns(self):
        returns = [0.01, 0.02, 0.015, 0.005, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_negative_returns(self):
        returns = [-0.01, -0.02, -0.015, -0.005, -0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_empty_returns(self):
        assert calculate_sharpe_ratio([]) == 0.0

    def test_single_return(self):
        assert calculate_sharpe_ratio([0.01]) == 0.0

    def test_zero_std(self):
        returns = [0.0001, 0.0001, 0.0001]
        assert calculate_sharpe_ratio(returns) == 0.0


class TestMaxDrawdown:
    def test_no_loss(self):
        values = [100.0, 101.0, 102.0, 103.0]
        assert calculate_max_drawdown(values) == pytest.approx(0.0, abs=1e-6)

    def test_with_drop(self):
        values = [100.0, 110.0, 90.0, 95.0]
        dd = calculate_max_drawdown(values)
        assert dd == pytest.approx((110.0 - 90.0) / 110.0, abs=1e-4)

    def test_empty(self):
        assert calculate_max_drawdown([]) == 0.0

    def test_single_value(self):
        assert calculate_max_drawdown([100.0]) == 0.0


class TestAvgTradeReturn:
    def test_positive(self):
        returns = [0.01, 0.02, 0.03]
        assert calculate_avg_trade_return(returns) == pytest.approx(0.02)

    def test_empty(self):
        assert calculate_avg_trade_return([]) == 0.0


class TestEpisodeScore:
    def test_good_metrics(self):
        metrics = {"sharpe_ratio": 2.0, "total_return": 10.0, "max_drawdown": 0.05}
        score = calculate_episode_score(metrics)
        assert 0.7 < score <= 1.0

    def test_bad_metrics(self):
        metrics = {"sharpe_ratio": -2.0, "total_return": -20.0, "max_drawdown": 0.5}
        score = calculate_episode_score(metrics)
        assert 0.0 <= score < 0.3

    def test_neutral_metrics(self):
        metrics = {"sharpe_ratio": 0.0, "total_return": 0.0, "max_drawdown": 0.0}
        score = calculate_episode_score(metrics)
        assert 0.3 < score < 0.71

    def test_empty_dict(self):
        assert calculate_episode_score({}) == 0.0


class TestPerformanceTracker:
    def test_init(self):
        tracker = PerformanceTracker(window_size=50)
        assert tracker.window_size == 50
        assert tracker.portfolio_values == []

    def test_add_initial_value(self):
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        assert tracker.portfolio_values == [1000.0]
        tracker.add_initial_value(2000.0)
        assert len(tracker.portfolio_values) == 1

    def test_update_and_get_metrics(self):
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        tracker.update(1010.0, action=1, reward=0.01, transaction_cost=0.5,
                       position=0.5, balance=500.0, price=1010.0)
        tracker.update(1020.0, action=0, reward=0.005, transaction_cost=0.0,
                       position=0.5, balance=500.0, price=1020.0)
        tracker.update(1015.0, action=5, reward=-0.003, transaction_cost=0.3,
                       position=0.0, balance=1015.0, price=1015.0)

        metrics = tracker.get_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "action_counts" in metrics
        assert metrics["total_return"] == pytest.approx(1.5, abs=0.1)
        assert metrics["transaction_costs"] == pytest.approx(0.8)

    def test_action_counts(self):
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        for action in [0, 0, 1, 2, 5, 5, 5]:
            tracker.update(1000.0, action=action, reward=0.0)
        counts = tracker.get_action_counts()
        assert counts[0] == 2
        assert counts[1] == 1
        assert counts[5] == 3

    def test_get_recent_metrics(self):
        tracker = PerformanceTracker(window_size=3)
        tracker.add_initial_value(1000.0)
        for i in range(5):
            tracker.update(1000.0 + i * 10, action=0, reward=0.01)
        recent = tracker.get_recent_metrics()
        assert "total_return" in recent

    def test_improvement_rate(self):
        tracker = PerformanceTracker(window_size=10)
        tracker.add_initial_value(1000.0)
        for i in range(10):
            tracker.update(1000.0 + i * 10, action=0, reward=0.01)
        rate = tracker.get_improvement_rate()
        assert rate > 0

    def test_improvement_rate_insufficient_data(self):
        tracker = PerformanceTracker()
        assert tracker.get_improvement_rate() == 0.0

    def test_stability(self):
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        for _ in range(10):
            tracker.update(1000.0, action=0, reward=0.0)
        stability = tracker.get_stability()
        assert 0.0 <= stability <= 1.0

    def test_stability_insufficient_data(self):
        tracker = PerformanceTracker()
        assert tracker.get_stability() == 0.0

    def test_empty_metrics(self):
        tracker = PerformanceTracker()
        assert tracker.get_metrics() == {}
