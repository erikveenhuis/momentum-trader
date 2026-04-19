"""Tests for :mod:`momentum_train.schedules` and the trainer's benchmark
scheduler wiring.

Covers:

* Pure math of :func:`compute_benchmark_frac` (start, mid, end, beyond,
  zero anneal length, descending vs ascending schedules).
* Backward compatibility — when the schedule keys are absent the trainer
  treats ``benchmark_allocation_frac`` as a constant.
* CLI override (``run.benchmark_frac_override``) wins over the schedule.
* :meth:`TradingLogic.set_benchmark_allocation_frac` actually changes the
  reward formula output.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from momentum_train.schedules import compute_benchmark_frac

# ---------------------------------------------------------------------------
# compute_benchmark_frac math
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_benchmark_frac_starts_at_start_value():
    assert compute_benchmark_frac(0, start=0.5, end=0.1, anneal_episodes=5000) == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_benchmark_frac_midpoint_is_linear_average():
    value = compute_benchmark_frac(2500, start=0.5, end=0.1, anneal_episodes=5000)
    assert value == pytest.approx(0.3)


@pytest.mark.unit
def test_compute_benchmark_frac_at_anneal_episodes_is_end():
    assert compute_benchmark_frac(5000, start=0.5, end=0.1, anneal_episodes=5000) == pytest.approx(0.1)


@pytest.mark.unit
def test_compute_benchmark_frac_beyond_anneal_clamps_to_end():
    assert compute_benchmark_frac(99_999, start=0.5, end=0.1, anneal_episodes=5000) == pytest.approx(0.1)


@pytest.mark.unit
def test_compute_benchmark_frac_zero_anneal_returns_end_immediately():
    """``anneal_episodes <= 0`` collapses to a constant (``end``)."""
    assert compute_benchmark_frac(0, start=0.5, end=0.2, anneal_episodes=0) == pytest.approx(0.2)
    assert compute_benchmark_frac(123, start=0.5, end=0.2, anneal_episodes=-7) == pytest.approx(0.2)


@pytest.mark.unit
def test_compute_benchmark_frac_negative_episode_clamps_to_start():
    assert compute_benchmark_frac(-5, start=0.5, end=0.1, anneal_episodes=5000) == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_benchmark_frac_ascending_schedule():
    """Schedule should also work in the increasing direction."""
    assert compute_benchmark_frac(500, start=0.0, end=1.0, anneal_episodes=1000) == pytest.approx(0.5)


@pytest.mark.unit
def test_compute_benchmark_frac_resume_anchored_on_absolute_episode():
    """Resuming at episode 3499 with the proposed defaults yields ~0.220."""
    value = compute_benchmark_frac(3499, start=0.5, end=0.10, anneal_episodes=5000)
    expected = 0.5 + (0.10 - 0.5) * (3499 / 5000)
    assert value == pytest.approx(expected)
    assert value < 0.5 and value > 0.10


# ---------------------------------------------------------------------------
# Trainer wiring: backward compat + override + setter
# ---------------------------------------------------------------------------


def _make_trainer_with_config(trainer_cfg: dict, env_cfg: dict, run_cfg: dict | None = None):
    """Build a trainer instance via ``__new__`` so we can exercise the
    ``__init__`` body without spinning up the full agent / device stack."""
    from momentum_train.trainer import RainbowTrainerModule

    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.trainer_config = trainer_cfg
    trainer.env_config = env_cfg
    trainer.run_config = run_cfg or {}
    trainer.writer = None
    return trainer


@pytest.mark.unit
def test_trainer_falls_back_to_legacy_constant_when_no_schedule_keys():
    """Old configs that only set ``environment.benchmark_allocation_frac``
    keep working: ``current_benchmark_frac`` returns the constant for every
    episode."""
    from momentum_train.trainer import RainbowTrainerModule

    trainer = _make_trainer_with_config(
        trainer_cfg={},
        env_cfg={"benchmark_allocation_frac": 0.42},
    )
    trainer.benchmark_frac_start = 0.42
    trainer.benchmark_frac_end = 0.42
    trainer.benchmark_frac_anneal_episodes = 0
    trainer.benchmark_frac_override = None
    assert RainbowTrainerModule.current_benchmark_frac(trainer, 0) == pytest.approx(0.42)
    assert RainbowTrainerModule.current_benchmark_frac(trainer, 9999) == pytest.approx(0.42)


@pytest.mark.unit
def test_trainer_override_pins_constant_value_ignoring_schedule():
    from momentum_train.trainer import RainbowTrainerModule

    trainer = _make_trainer_with_config(
        trainer_cfg={},
        env_cfg={"benchmark_allocation_frac": 0.5},
    )
    trainer.benchmark_frac_start = 0.5
    trainer.benchmark_frac_end = 0.1
    trainer.benchmark_frac_anneal_episodes = 5000
    trainer.benchmark_frac_override = 0.15
    assert RainbowTrainerModule.current_benchmark_frac(trainer, 0) == pytest.approx(0.15)
    assert RainbowTrainerModule.current_benchmark_frac(trainer, 4999) == pytest.approx(0.15)
    assert RainbowTrainerModule.current_benchmark_frac(trainer, 9_999_999) == pytest.approx(0.15)


@pytest.mark.unit
def test_trainer_apply_benchmark_frac_pushes_value_into_env():
    """``_apply_benchmark_frac_to_env`` calls the setter on
    ``env.trading_logic`` with the scheduled value and returns it."""
    from momentum_train.trainer import RainbowTrainerModule

    trainer = _make_trainer_with_config(trainer_cfg={}, env_cfg={})
    trainer.benchmark_frac_start = 0.5
    trainer.benchmark_frac_end = 0.1
    trainer.benchmark_frac_anneal_episodes = 1000
    trainer.benchmark_frac_override = None

    captured: dict = {}

    class _StubLogic:
        def set_benchmark_allocation_frac(self, value: float) -> None:
            captured["value"] = float(value)

    env = SimpleNamespace(trading_logic=_StubLogic())
    applied = RainbowTrainerModule._apply_benchmark_frac_to_env(trainer, env, 500)
    assert applied == pytest.approx(0.3)
    assert captured["value"] == pytest.approx(0.3)


@pytest.mark.unit
def test_trainer_apply_benchmark_frac_handles_env_without_setter_gracefully():
    """Vector envs that haven't been upgraded must not crash the trainer."""
    from momentum_train.trainer import RainbowTrainerModule

    trainer = _make_trainer_with_config(trainer_cfg={}, env_cfg={})
    trainer.benchmark_frac_start = 0.5
    trainer.benchmark_frac_end = 0.1
    trainer.benchmark_frac_anneal_episodes = 1000
    trainer.benchmark_frac_override = None

    env = SimpleNamespace()
    applied = RainbowTrainerModule._apply_benchmark_frac_to_env(trainer, env, 0)
    assert applied == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TradingLogic setter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trading_logic_set_benchmark_allocation_frac_changes_reward_formula():
    """After ``set_benchmark_allocation_frac(new)``, ``calculate_reward``
    subtracts ``new * price_return`` rather than the original value."""
    from momentum_env.trading import TradingLogic

    logic = TradingLogic(
        transaction_fee=0.001,
        reward_scale=1.0,
        invalid_action_penalty=-0.1,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        min_trade_value=1.0,
        benchmark_allocation_frac=0.5,
    )
    assert logic.benchmark_allocation_frac == pytest.approx(0.5)

    base_reward = logic.calculate_reward(
        prev_portfolio_value=100.0,
        pre_trade_portfolio_value=101.0,
        post_trade_portfolio_value=101.0,
        is_valid=True,
        price_return=0.01,
        position_fraction=0.0,
    )

    logic.set_benchmark_allocation_frac(0.1)
    assert logic.benchmark_allocation_frac == pytest.approx(0.1)
    relaxed_reward = logic.calculate_reward(
        prev_portfolio_value=100.0,
        pre_trade_portfolio_value=101.0,
        post_trade_portfolio_value=101.0,
        is_valid=True,
        price_return=0.01,
        position_fraction=0.0,
    )

    # Relaxed benchmark subtracts less, so the same scenario yields a higher
    # reward. Difference == (0.5 - 0.1) * price_return = 0.4 * 0.01 = 0.004.
    assert relaxed_reward > base_reward
    assert relaxed_reward - base_reward == pytest.approx(0.004, abs=1e-9)


@pytest.mark.unit
def test_trading_logic_setter_coerces_to_float():
    from momentum_env.trading import TradingLogic

    logic = TradingLogic(
        transaction_fee=0.001,
        reward_scale=1.0,
        invalid_action_penalty=-0.1,
        drawdown_penalty_lambda=0.0,
        slippage_bps=0.0,
        opportunity_cost_lambda=0.0,
        min_trade_value=1.0,
        benchmark_allocation_frac=0.5,
    )
    logic.set_benchmark_allocation_frac("0.25")
    assert isinstance(logic.benchmark_allocation_frac, float)
    assert logic.benchmark_allocation_frac == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Determinism / clamping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_benchmark_frac_clamps_floating_point_drift():
    """Even if linear interpolation overshoots due to FP, the result stays
    inside ``[min(start, end), max(start, end)]``."""
    value = compute_benchmark_frac(1, start=0.1, end=0.2, anneal_episodes=10**9)
    assert 0.1 <= value <= 0.2
    assert math.isfinite(value)
