"""Tests for the metrics module."""

import warnings

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

    def test_no_overflow_with_float32_inputs(self):
        """np.float32 inputs must not trigger `RuntimeWarning: overflow encountered in exp`.

        Regression test for the issue surfaced during the post-training test
        sweep where `PerformanceTracker` produced `np.float32` Sharpe values
        that propagated into `calculate_episode_score`. The legacy clip
        ceiling of 700 only protected `float64`; `np.exp(np.float32(700))`
        saturates above ~exp(88.7) and emits an overflow warning.
        """
        metrics = {
            "sharpe_ratio": np.float32(-50.0),
            "total_return": np.float32(-200.0),
            "max_drawdown": np.float32(0.9),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            score = calculate_episode_score(metrics)
        assert 0.0 <= score <= 1.0

    def test_extreme_negative_inputs_saturate_to_zero(self):
        """Extreme negative Sharpe / return should give ~zero score, not raise.

        Confirms the saturation-at-±80 ceiling behaves like the asymptote.
        """
        metrics = {"sharpe_ratio": -1e6, "total_return": -1e6, "max_drawdown": 0.99}
        score = calculate_episode_score(metrics)
        assert 0.0 <= score < 0.05

    def test_extreme_positive_inputs_saturate_to_one(self):
        """Extreme positive Sharpe / return should give ~one score, not raise."""
        metrics = {"sharpe_ratio": 1e6, "total_return": 1e6, "max_drawdown": 0.0}
        score = calculate_episode_score(metrics)
        assert 0.95 < score <= 1.0


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
        tracker.update(1010.0, action=1, reward=0.01, transaction_cost=0.5, position=0.5, balance=500.0, price=1010.0)
        tracker.update(1020.0, action=0, reward=0.005, transaction_cost=0.0, position=0.5, balance=500.0, price=1020.0)
        tracker.update(1015.0, action=5, reward=-0.003, transaction_cost=0.3, position=0.0, balance=1015.0, price=1015.0)

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

    def test_action_provenance_counts_split_greedy_eps(self):
        """Tier 2c: per-action greedy/eps split tracked alongside totals."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        # action 0 (flat) — 1 greedy, 1 eps
        tracker.update(1000.0, action=0, reward=0.0, was_greedy=True)
        tracker.update(1000.0, action=0, reward=0.0, was_greedy=False)
        # action 5 (max long) — 2 greedy, 3 eps
        for _ in range(2):
            tracker.update(1000.0, action=5, reward=0.0, was_greedy=True)
        for _ in range(3):
            tracker.update(1000.0, action=5, reward=0.0, was_greedy=False)
        # action 3 — provenance unknown (legacy caller)
        tracker.update(1000.0, action=3, reward=0.0)

        prov = tracker.get_action_provenance_counts()
        assert prov["greedy"][0] == 1
        assert prov["eps"][0] == 1
        assert prov["greedy"][5] == 2
        assert prov["eps"][5] == 3
        assert prov["unknown"][3] == 1
        # Totals must reconcile with get_action_counts.
        counts = tracker.get_action_counts()
        for a in range(6):
            assert counts[a] == prov["greedy"][a] + prov["eps"][a] + prov["unknown"][a]

    def test_epsilon_forced_trade_fraction_excludes_action_zero(self):
        """Tier 2c: only non-flat (action != 0) eps actions count toward the fraction."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        # 5 eps action-0 calls — must NOT influence the fraction.
        for _ in range(5):
            tracker.update(1000.0, action=0, reward=0.0, was_greedy=False)
        # 1 greedy non-flat + 3 eps non-flat → 3/4.
        tracker.update(1000.0, action=2, reward=0.0, was_greedy=True)
        for _ in range(3):
            tracker.update(1000.0, action=4, reward=0.0, was_greedy=False)
        assert tracker.get_epsilon_forced_trade_fraction() == pytest.approx(3.0 / 4.0)

    def test_epsilon_forced_trade_fraction_no_trades_returns_zero(self):
        """Tier 2c: with no non-flat actions the fraction is 0.0 (not NaN)."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        for _ in range(3):
            tracker.update(1000.0, action=0, reward=0.0, was_greedy=True)
        assert tracker.get_epsilon_forced_trade_fraction() == 0.0

    def test_get_metrics_includes_provenance_payload(self):
        """Tier 2c: get_metrics surfaces provenance counts and eps-trade fraction."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        tracker.update(1000.0, action=1, reward=0.0, was_greedy=True)
        tracker.update(1000.0, action=2, reward=0.0, was_greedy=False)
        m = tracker.get_metrics()
        assert "action_provenance_counts" in m
        assert "epsilon_forced_trade_fraction" in m
        assert m["action_provenance_counts"]["greedy"][1] == 1
        assert m["action_provenance_counts"]["eps"][2] == 1
        assert m["epsilon_forced_trade_fraction"] == pytest.approx(1.0 / 2.0)

    def test_get_reward_outlier_stats_flags_when_exceeds_5x_clip(self):
        """Tier 4b: outlier flag fires when any reward exceeds 5*clip."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        for r in [0.1, -0.05, 0.2, 6.0]:  # 6.0 > 5 * 1.0
            tracker.update(1000.0, action=0, reward=r)
        stats = tracker.get_reward_outlier_stats(reward_clip=1.0)
        assert stats["reward_max"] == pytest.approx(6.0)
        assert stats["reward_min"] == pytest.approx(-0.05)
        assert stats["reward_outlier_flag"] == 1.0
        assert stats["reward_p99_abs"] >= 0.0

    def test_get_reward_outlier_stats_no_flag_when_within_clip(self):
        """Tier 4b: with all rewards within 5*clip, the outlier flag stays at 0."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        for r in [0.5, -0.3, 1.5, -2.0]:  # All within 5 * 1.0 = 5.0
            tracker.update(1000.0, action=0, reward=r)
        stats = tracker.get_reward_outlier_stats(reward_clip=1.0)
        assert stats["reward_outlier_flag"] == 0.0

    def test_get_reward_outlier_stats_empty_returns_zeros(self):
        """Tier 4b: empty rewards collapse to all-zero stats so caller can log unconditionally."""
        tracker = PerformanceTracker()
        stats = tracker.get_reward_outlier_stats(reward_clip=1.0)
        assert stats == {
            "reward_min": 0.0,
            "reward_max": 0.0,
            "reward_p99_abs": 0.0,
            "reward_outlier_flag": 0.0,
        }

    def test_get_reward_by_action_stats_partitions_correctly(self):
        """Tier 4c: per-action reward mean/std splits the rewards by action."""
        tracker = PerformanceTracker()
        tracker.add_initial_value(1000.0)
        plan = [(0, 0.0), (0, 0.0), (1, 1.0), (1, 3.0), (5, -2.0)]
        for action, reward in plan:
            tracker.update(1000.0, action=action, reward=reward)
        stats = tracker.get_reward_by_action_stats()
        assert stats[0]["mean"] == pytest.approx(0.0)
        assert stats[0]["std"] == pytest.approx(0.0)
        assert stats[0]["count"] == pytest.approx(2.0)
        assert stats[1]["mean"] == pytest.approx(2.0)
        assert stats[1]["std"] == pytest.approx(1.0)  # std of [1, 3] = 1.0 ddof=0
        assert stats[1]["count"] == pytest.approx(2.0)
        assert stats[5]["mean"] == pytest.approx(-2.0)
        assert stats[5]["count"] == pytest.approx(1.0)

    def test_get_reward_by_action_stats_empty_returns_empty(self):
        """Tier 4c: empty tracker returns an empty mapping (callers default missing actions)."""
        tracker = PerformanceTracker()
        assert tracker.get_reward_by_action_stats() == {}

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
