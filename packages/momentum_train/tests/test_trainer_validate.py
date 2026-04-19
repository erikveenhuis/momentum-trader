from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from momentum_train.metrics import PerformanceTracker
from momentum_train.trainer import RainbowTrainerModule


def _create_minimal_trainer(tmp_path) -> RainbowTrainerModule:
    trainer = RainbowTrainerModule.__new__(RainbowTrainerModule)
    trainer.validation_freq = 1
    trainer.checkpoint_save_freq = 5
    trainer.best_validation_metric = -np.inf
    trainer.early_stopping_counter = 0
    trainer.min_validation_threshold = 0.0
    trainer.model_dir = tmp_path
    trainer.latest_trainer_checkpoint_path = str(tmp_path / "checkpoint_latest.pt")
    trainer.best_trainer_checkpoint_base_path = str(tmp_path / "best_checkpoint")
    trainer.writer = None
    trainer.reward_window = 10
    trainer.min_episodes_before_early_stopping = 0
    trainer.agent = SimpleNamespace(lr_scheduler_enabled=False, num_actions=6)
    # Tier 4a default for tests: disabled unless explicitly opted in. Avoids
    # AttributeError in _maybe_log_per_stats -> _maybe_audit_per_buffer_distribution.
    trainer.per_buffer_audit_interval = 0
    return trainer


class _CapturingWriter:
    """Mock SummaryWriter capturing add_scalar/add_histogram calls for assertions."""

    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []
        self.histograms: list[tuple[str, object, int]] = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((str(tag), float(value), int(step)))

    def add_histogram(self, tag, values, step):
        self.histograms.append((str(tag), values, int(step)))

    def tags(self) -> list[str]:
        return [t for t, _v, _s in self.scalars]

    def value_for(self, tag: str) -> float:
        matches = [v for t, v, _s in self.scalars if t == tag]
        if not matches:
            raise KeyError(f"No scalar emitted with tag {tag!r}")
        return matches[-1]


@pytest.mark.unit
def test_validate_aggregates_metrics_and_writes_results(tmp_path, monkeypatch):
    trainer = _create_minimal_trainer(tmp_path)

    metrics_sequence = [
        {
            "avg_reward": 1.0,
            "portfolio_value": 1005.0,
            "total_return": 5.0,
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.02,
            "transaction_costs": 1.5,
        },
        {
            "avg_reward": 2.0,
            "portfolio_value": 1010.0,
            "total_return": 6.0,
            "sharpe_ratio": 0.2,
            "max_drawdown": 0.01,
            "transaction_costs": 2.0,
        },
    ]

    detailed_results = [
        {"file": "file1.csv", "reward": 1.0},
        {"file": "file2.csv", "reward": 2.0},
    ]

    episode_scores = [0.4, 0.6]
    iterator = iter(
        {
            "file_metrics": metrics_sequence[i],
            "detailed_result": detailed_results[i],
            "episode_score": episode_scores[i],
        }
        for i in range(len(metrics_sequence))
    )

    trainer._validate_single_file = lambda path, **kwargs: next(iterator)

    saved_records = []
    original_save = RainbowTrainerModule._save_validation_results

    def capture_save(self, validation_score, avg_metrics, results):
        saved_records.append((validation_score, avg_metrics, results))
        original_save(self, validation_score, avg_metrics, results)

    trainer._save_validation_results = capture_save.__get__(trainer, RainbowTrainerModule)

    def record_early_stopping(score, episode=0):
        trainer.best_validation_metric = score
        trainer.early_stopping_counter = 0
        return False

    trainer._check_early_stopping = record_early_stopping

    should_stop, validation_score, avg_metrics = trainer.validate([Path("file1.csv"), Path("file2.csv")], episode=0)

    assert not should_stop
    assert validation_score == pytest.approx(np.mean(episode_scores))
    assert trainer.best_validation_metric == pytest.approx(validation_score)
    assert saved_records, "Validation results were not persisted"

    saved_score, saved_avg, saved_details = saved_records[0]
    assert saved_score == validation_score
    assert saved_avg == avg_metrics
    assert len(saved_details) == len(detailed_results)

    output_files = list(tmp_path.glob("validation_results_*.json"))
    assert output_files, "Validation summary JSON was not generated"


@pytest.mark.unit
def test_calculate_average_validation_metrics_includes_action_rates(tmp_path):
    """Tier 1b: averaged per-action rates are returned alongside portfolio metrics."""
    trainer = _create_minimal_trainer(tmp_path)

    file_metrics = [
        {
            "avg_reward": 1.0,
            "portfolio_value": 1010.0,
            "total_return": 1.0,
            "sharpe_ratio": 0.1,
            "max_drawdown": 0.0,
            "transaction_costs": 1.0,
            "total_steps": 100,
            "action_counts": {0: 80, 1: 5, 2: 5, 3: 4, 4: 3, 5: 3},
        },
        {
            "avg_reward": 2.0,
            "portfolio_value": 990.0,
            "total_return": -1.0,
            "sharpe_ratio": -0.1,
            "max_drawdown": 0.02,
            "transaction_costs": 0.5,
            "total_steps": 200,
            "action_counts": {0: 200, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        },
    ]

    avg = trainer._calculate_average_validation_metrics(file_metrics)

    assert "action_rates" in avg
    rates = avg["action_rates"]
    assert set(rates.keys()) == {0, 1, 2, 3, 4, 5}
    assert rates[0] == pytest.approx((80 / 100 + 200 / 200) / 2)
    assert rates[1] == pytest.approx((5 / 100 + 0 / 200) / 2)
    assert rates[5] == pytest.approx((3 / 100 + 0 / 200) / 2)

    assert avg["final_portfolio_value"] == pytest.approx(np.mean([1010.0, 990.0]))
    assert avg["transaction_costs"] == pytest.approx(np.mean([1.0, 0.5]))


@pytest.mark.unit
@pytest.mark.tb_logging
def test_handle_validation_emits_action_rate_and_portfolio_scalars(tmp_path):
    """Tier 1b: validation TB block emits Validation/Action Rate/{0..5}, Final Portfolio Value, Transaction Costs."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()

    avg_val_metrics = {
        "avg_reward": 1.5,
        "portfolio_value": 1005.0,
        "final_portfolio_value": 1005.0,
        "total_return": 0.5,
        "sharpe_ratio": 0.05,
        "max_drawdown": 0.01,
        "transaction_costs": 1.25,
        "action_rates": {0: 0.7, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1},
    }

    def fake_validate(self, val_files, episode=0):
        return False, 0.42, avg_val_metrics

    def fake_check_early_stopping(score, episode=0):
        return False

    trainer.validate = fake_validate.__get__(trainer, RainbowTrainerModule)
    trainer._check_early_stopping = fake_check_early_stopping
    trainer._save_checkpoint = lambda *a, **kw: None
    trainer.should_validate = lambda episode, recent_metrics: True

    tracker = SimpleNamespace(get_recent_metrics=lambda: {})
    trainer._handle_validation_and_checkpointing(
        episode=0,
        total_train_steps=10,
        val_files=[Path("file1.csv")],
        tracker=tracker,
    )

    writer = trainer.writer
    tags = set(writer.tags())
    assert "Validation/Final Portfolio Value" in tags
    assert "Validation/Transaction Costs" in tags
    for action_idx in range(6):
        assert f"Validation/Action Rate/{action_idx}" in tags

    assert writer.value_for("Validation/Final Portfolio Value") == pytest.approx(1005.0)
    assert writer.value_for("Validation/Transaction Costs") == pytest.approx(1.25)
    assert writer.value_for("Validation/Action Rate/0") == pytest.approx(0.7)
    assert writer.value_for("Validation/Action Rate/5") == pytest.approx(0.1)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_maybe_log_per_stats_mirrors_to_tensorboard(tmp_path):
    """Tier 1d: PER stats are mirrored to Train/PER/* scalars."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.per_stats_log_freq = 1
    trainer.reward_clip_value = None  # Disable audit branch.

    stats = {
        "size": 1234,
        "capacity": 10000,
        "fill_ratio": 0.1234,
        "alpha": 0.6,
        "beta": 0.55,
        "beta_progress": 0.42,
        "total_priority": 9876.0,
        "avg_priority": 8.0,
        "max_priority": 100.0,
        "total_steps": 5000,
    }
    trainer.agent = SimpleNamespace(get_per_stats=lambda: stats, num_actions=6, buffer=None)

    trainer._maybe_log_per_stats(total_train_steps=5000)

    writer = trainer.writer
    tags = set(writer.tags())
    for expected in (
        "Train/PER/AvgPriority",
        "Train/PER/MaxPriority",
        "Train/PER/TotalPriority",
        "Train/PER/Beta",
        "Train/PER/Alpha",
        "Train/PER/Fill",
        "Train/PER/Size",
    ):
        assert expected in tags, f"Missing PER scalar: {expected}"
    assert writer.value_for("Train/PER/AvgPriority") == pytest.approx(8.0)
    assert writer.value_for("Train/PER/MaxPriority") == pytest.approx(100.0)
    assert writer.value_for("Train/PER/Fill") == pytest.approx(0.1234)
    assert writer.value_for("Train/PER/Beta") == pytest.approx(0.55)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_per_buffer_clip_bypass_audit_flags_outliers(tmp_path):
    """Tier 1d: clip-bypass audit flags stored rewards exceeding the n-step clip bound."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.per_stats_log_freq = 1
    trainer.reward_clip_value = 1.0
    trainer.agent_config = {"gamma": 0.0, "n_steps": 1}  # bound = 1.0 * 1 = 1.0.

    # Stub buffer holding both legal and out-of-bounds n-step rewards.
    rewards = [0.5, -0.5, 0.9, 5.0, -3.0]
    fake_experiences = [SimpleNamespace(reward=r) for r in rewards]
    fake_buffer = SimpleNamespace(buffer=fake_experiences)
    stats = {
        "size": len(fake_experiences),
        "capacity": 100,
        "fill_ratio": 0.05,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_progress": 0.0,
        "total_priority": 1.0,
        "avg_priority": 0.2,
        "max_priority": 1.0,
        "total_steps": 100,
    }
    trainer.agent = SimpleNamespace(get_per_stats=lambda: stats, buffer=fake_buffer, num_actions=6)

    trainer._maybe_log_per_stats(total_train_steps=100)

    writer = trainer.writer
    tags = set(writer.tags())
    assert "Train/PER/ClipBypassEventCount" in tags
    assert "Train/PER/ClipBypassFraction" in tags
    assert "Train/PER/StoredRewardClipBound" in tags
    assert writer.value_for("Train/PER/ClipBypassEventCount") == pytest.approx(2.0)
    assert writer.value_for("Train/PER/ClipBypassFraction") == pytest.approx(2.0 / 5.0)
    assert writer.value_for("Train/PER/StoredRewardClipBound") == pytest.approx(1.0, rel=1e-3)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_per_buffer_clip_bypass_audit_skipped_when_clip_disabled(tmp_path):
    """Tier 1d: when reward clipping is disabled, the audit is skipped (no scalars emitted)."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.per_stats_log_freq = 1
    trainer.reward_clip_value = None  # No clipping.
    trainer.agent_config = {"gamma": 0.99, "n_steps": 3}

    fake_experiences = [SimpleNamespace(reward=r) for r in [0.0, 100.0, -100.0]]
    fake_buffer = SimpleNamespace(buffer=fake_experiences)
    stats = {
        "size": 3,
        "capacity": 100,
        "fill_ratio": 0.03,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_progress": 0.0,
        "total_priority": 1.0,
        "avg_priority": 0.2,
        "max_priority": 1.0,
        "total_steps": 100,
    }
    trainer.agent = SimpleNamespace(get_per_stats=lambda: stats, buffer=fake_buffer, num_actions=6)

    trainer._maybe_log_per_stats(total_train_steps=100)

    writer = trainer.writer
    tags = set(writer.tags())
    assert "Train/PER/ClipBypassEventCount" not in tags
    assert "Train/PER/ClipBypassFraction" not in tags


def _make_per_buffer_for_audit(rewards, actions, priorities):
    """Build a (stored_buffer, sum_tree) pair shaped like PrioritizedReplayBuffer.

    Lockstep writes: buffer index k <-> SumTree leaf at index ``k + capacity-1``.
    See PrioritizedReplayBuffer.store -- the SumTree's internal ``data_indices``
    table is identity-mapped because ``buffer_write_idx`` and ``tree.write``
    cycle in lockstep starting from zero.
    """
    n = len(rewards)
    assert len(actions) == n and len(priorities) == n
    capacity = max(n, 4)
    tree = np.zeros(2 * capacity - 1, dtype=np.float64)
    for i, p in enumerate(priorities):
        tree[i + capacity - 1] = float(p)
    stored = [SimpleNamespace(reward=float(r), action=int(a)) for r, a in zip(rewards, actions)]
    sum_tree = SimpleNamespace(tree=tree, capacity=capacity)
    fake_buffer = SimpleNamespace(buffer=stored, tree=sum_tree)
    return fake_buffer


@pytest.mark.unit
@pytest.mark.tb_logging
def test_per_buffer_distribution_audit_emits_histogram_and_per_action_priorities(tmp_path):
    """Tier 4a: distribution audit emits reward histogram + per-action priority means."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.per_stats_log_freq = 1
    trainer.per_buffer_audit_interval = 1
    trainer.reward_clip_value = 1.0
    trainer.agent_config = {"gamma": 0.99, "n_steps": 1}

    rewards = [0.1, -0.2, 0.3, -0.4, 0.05, 10.0]
    actions = [0, 1, 2, 0, 1, 5]
    priorities = [0.5, 0.6, 0.7, 0.8, 0.9, 50.0]
    fake_buffer = _make_per_buffer_for_audit(rewards, actions, priorities)
    stats = {
        "size": len(rewards),
        "capacity": 100,
        "fill_ratio": 0.06,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_progress": 0.0,
        "total_priority": 1.0,
        "avg_priority": 0.2,
        "max_priority": 1.0,
        "total_steps": 100,
    }
    trainer.agent = SimpleNamespace(get_per_stats=lambda: stats, buffer=fake_buffer, num_actions=6)

    trainer._maybe_log_per_stats(total_train_steps=100)

    writer = trainer.writer
    tags = set(writer.tags())
    hist_tags = {h[0] for h in writer.histograms}
    assert "Train/PER/Reward/Histogram" in hist_tags
    assert "Train/PER/Reward/OutlierFrac" in tags
    # Per-action priorities for at least the actions present.
    for k in (0, 1, 2, 5):
        assert f"Train/PER/PriorityByAction/{k}" in tags
    # Top-1pct shares for every action 0..5.
    for k in range(6):
        assert f"Train/PER/Top1PctActionShare/{k}" in tags

    # Outlier fraction: |10.0| > 5*1.0 -> 1/6 of samples flagged.
    assert writer.value_for("Train/PER/Reward/OutlierFrac") == pytest.approx(1.0 / 6.0)

    # Top 1% (with 6 samples) collapses to top_n=max(1, 6//100)=1; the highest
    # priority belongs to action 5, so its share should be 1.0 and others 0.0.
    assert writer.value_for("Train/PER/Top1PctActionShare/5") == pytest.approx(1.0)
    for k in (0, 1, 2, 3, 4):
        assert writer.value_for(f"Train/PER/Top1PctActionShare/{k}") == pytest.approx(0.0)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_per_buffer_distribution_audit_skipped_when_interval_zero(tmp_path):
    """Tier 4a: setting per_buffer_audit_interval=0 disables Tier 4a scalars/histograms."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.per_stats_log_freq = 1
    trainer.per_buffer_audit_interval = 0  # Disabled.
    trainer.reward_clip_value = 1.0
    trainer.agent_config = {"gamma": 0.99, "n_steps": 1}

    fake_buffer = _make_per_buffer_for_audit([0.0, 1.0], [0, 1], [0.1, 0.2])
    stats = {
        "size": 2,
        "capacity": 100,
        "fill_ratio": 0.02,
        "alpha": 0.6,
        "beta": 0.4,
        "beta_progress": 0.0,
        "total_priority": 1.0,
        "avg_priority": 0.2,
        "max_priority": 1.0,
        "total_steps": 100,
    }
    trainer.agent = SimpleNamespace(get_per_stats=lambda: stats, buffer=fake_buffer, num_actions=6)

    trainer._maybe_log_per_stats(total_train_steps=100)

    tags = set(trainer.writer.tags())
    hist_tags = {h[0] for h in trainer.writer.histograms}
    assert "Train/PER/Reward/Histogram" not in hist_tags
    assert "Train/PER/Reward/OutlierFrac" not in tags
    for k in range(6):
        assert f"Train/PER/PriorityByAction/{k}" not in tags
        assert f"Train/PER/Top1PctActionShare/{k}" not in tags


@pytest.mark.unit
def test_handle_validation_and_checkpointing_triggers_best_checkpoint(tmp_path):
    trainer = _create_minimal_trainer(tmp_path)
    trainer.best_validation_metric = 0.2

    save_calls = []

    def capture_save(self, episode, total_steps, is_best, validation_score=None):
        save_calls.append(
            {
                "episode": episode,
                "total_steps": total_steps,
                "is_best": is_best,
                "validation_score": validation_score,
            }
        )

    trainer._save_checkpoint = capture_save.__get__(trainer, RainbowTrainerModule)

    def fake_validate(self, val_files, episode=0):
        # Update best_validation_metric to simulate the real validate() behavior
        trainer.best_validation_metric = 0.5
        return False, 0.5, {"avg_reward": 1.0}

    def record_early_stopping(score, episode=0):
        trainer.best_validation_metric = score
        trainer.early_stopping_counter = 0
        return False

    trainer.validate = fake_validate.__get__(trainer, RainbowTrainerModule)
    trainer._check_early_stopping = record_early_stopping

    tracker = SimpleNamespace(get_recent_metrics=lambda: {})

    should_stop = trainer._handle_validation_and_checkpointing(
        episode=0,
        total_train_steps=42,
        val_files=[Path("file1.csv")],
        tracker=tracker,
    )

    assert should_stop is False
    assert trainer.best_validation_metric == pytest.approx(0.5)
    assert len(save_calls) == 1

    call = save_calls[0]
    assert call["episode"] == 1
    assert call["total_steps"] == 42
    assert call["is_best"] is True
    assert call["validation_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tier 2c: action provenance + per-episode Train/Trade/* TB emission
# ---------------------------------------------------------------------------


def _make_trainer_for_episode_summary(tmp_path) -> RainbowTrainerModule:
    """Build a trainer instance suitable for exercising _log_episode_summary."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.update_freq = 1
    trainer.invalid_action_window = 100
    trainer.invalid_action_rate_window = deque(maxlen=trainer.invalid_action_window)
    trainer.per_stats_log_freq = 0  # Never trigger PER stats branch.
    trainer.reward_clip_value = None
    trainer.agent = SimpleNamespace(
        lr_scheduler_enabled=False,
        num_actions=6,
        current_epsilon=0.05,
        optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]),
    )
    return trainer


@pytest.mark.unit
@pytest.mark.tb_logging
def test_log_episode_summary_emits_greedy_eps_split_and_eps_trade_fraction(tmp_path):
    """Tier 2c: per-action greedy/eps rates and EpsilonForcedTradeFraction emitted."""
    trainer = _make_trainer_for_episode_summary(tmp_path)
    trainer.writer = _CapturingWriter()

    tracker = PerformanceTracker()
    tracker.add_initial_value(1000.0)
    # 10 steps total: 5 greedy action 0, 2 greedy action 1, 3 eps action 5.
    pv = 1000.0
    for _ in range(5):
        tracker.update(pv, action=0, reward=0.0, was_greedy=True)
    for _ in range(2):
        tracker.update(pv, action=1, reward=0.0, was_greedy=True)
    for _ in range(3):
        tracker.update(pv, action=5, reward=0.0, was_greedy=False)

    final_info = {"portfolio_value": pv, "position": 0.0}
    trainer._log_episode_summary(
        episode=0,
        episode_reward=0.0,
        total_rewards=[0.0],
        episode_loss=0.0,
        steps_in_episode=10,
        tracker=tracker,
        final_info=final_info,
        invalid_action_count=0,
        total_train_steps=10,
    )

    writer = trainer.writer
    tags = set(writer.tags())
    # Per-action greedy/eps split must be emitted for all 6 actions.
    for a in range(6):
        assert f"Train/Action Rate/Greedy/{a}" in tags
        assert f"Train/Action Rate/Eps/{a}" in tags
    assert "Train/EpsilonForcedTradeFraction" in tags

    # 5 greedy a=0 / 10 steps = 0.5; 2 greedy a=1 / 10 = 0.2.
    assert writer.value_for("Train/Action Rate/Greedy/0") == pytest.approx(0.5)
    assert writer.value_for("Train/Action Rate/Greedy/1") == pytest.approx(0.2)
    # 3 eps a=5 / 10 = 0.3.
    assert writer.value_for("Train/Action Rate/Eps/5") == pytest.approx(0.3)
    # Greedy non-flat = 2 (a=1), eps non-flat = 3 (a=5) → 3/(3+2) = 0.6.
    assert writer.value_for("Train/EpsilonForcedTradeFraction") == pytest.approx(0.6)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_log_episode_summary_emits_reward_outlier_scalars(tmp_path):
    """Tier 4b: per-episode RewardMin/Max/P99Abs + outlier flag are emitted."""
    trainer = _make_trainer_for_episode_summary(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.reward_clip_value = 1.0

    tracker = PerformanceTracker()
    tracker.add_initial_value(1000.0)
    rewards = [0.1, -0.2, 6.0]  # 6.0 > 5*1.0 → flag should fire.
    for r in rewards:
        tracker.update(1000.0, action=0, reward=r, was_greedy=True)

    trainer._log_episode_summary(
        episode=7,
        episode_reward=sum(rewards),
        total_rewards=[sum(rewards)],
        episode_loss=0.0,
        steps_in_episode=len(rewards),
        tracker=tracker,
        final_info={"portfolio_value": 1000.0, "position": 0.0},
        invalid_action_count=0,
        total_train_steps=len(rewards),
    )

    writer = trainer.writer
    tags = set(writer.tags())
    for expected in (
        "Train/Episode/RewardMin",
        "Train/Episode/RewardMax",
        "Train/Episode/RewardP99Abs",
        "Train/Episode/RewardOutlierFlag",
    ):
        assert expected in tags, f"Missing Tier 4b scalar: {expected}"
    assert writer.value_for("Train/Episode/RewardMax") == pytest.approx(6.0)
    assert writer.value_for("Train/Episode/RewardMin") == pytest.approx(-0.2)
    assert writer.value_for("Train/Episode/RewardOutlierFlag") == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_log_episode_summary_outlier_flag_zero_when_clip_disabled(tmp_path):
    """Tier 4b: outlier flag is 0 when reward clipping is disabled (no threshold)."""
    trainer = _make_trainer_for_episode_summary(tmp_path)
    trainer.writer = _CapturingWriter()
    trainer.reward_clip_value = None  # No clip → no flag (min/max/p99 still emitted).

    tracker = PerformanceTracker()
    tracker.add_initial_value(1000.0)
    for r in [50.0, -10.0, 0.0]:
        tracker.update(1000.0, action=0, reward=r, was_greedy=True)

    trainer._log_episode_summary(
        episode=3,
        episode_reward=40.0,
        total_rewards=[40.0],
        episode_loss=0.0,
        steps_in_episode=3,
        tracker=tracker,
        final_info={"portfolio_value": 1000.0, "position": 0.0},
        invalid_action_count=0,
        total_train_steps=3,
    )

    writer = trainer.writer
    assert writer.value_for("Train/Episode/RewardOutlierFlag") == pytest.approx(0.0)
    assert writer.value_for("Train/Episode/RewardMax") == pytest.approx(50.0)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_log_episode_summary_emits_reward_by_action_scalars(tmp_path):
    """Tier 4c: per-action reward mean/std emitted for every action in [0, num_actions)."""
    trainer = _make_trainer_for_episode_summary(tmp_path)
    trainer.writer = _CapturingWriter()

    tracker = PerformanceTracker()
    tracker.add_initial_value(1000.0)
    # Action 0 hold (mean 0), action 5 trades (mean +2, std=1 ddof=0).
    tracker.update(1000.0, action=0, reward=0.0, was_greedy=True)
    tracker.update(1000.0, action=0, reward=0.0, was_greedy=True)
    tracker.update(1000.0, action=5, reward=1.0, was_greedy=True)
    tracker.update(1000.0, action=5, reward=3.0, was_greedy=True)

    trainer._log_episode_summary(
        episode=11,
        episode_reward=4.0,
        total_rewards=[4.0],
        episode_loss=0.0,
        steps_in_episode=4,
        tracker=tracker,
        final_info={"portfolio_value": 1000.0, "position": 0.0},
        invalid_action_count=0,
        total_train_steps=4,
    )

    writer = trainer.writer
    tags = set(writer.tags())
    for k in range(6):
        assert f"Train/Reward/MeanByAction/{k}" in tags, f"missing MeanByAction for action {k}"
        assert f"Train/Reward/StdByAction/{k}" in tags, f"missing StdByAction for action {k}"
    assert writer.value_for("Train/Reward/MeanByAction/0") == pytest.approx(0.0)
    assert writer.value_for("Train/Reward/MeanByAction/5") == pytest.approx(2.0)
    assert writer.value_for("Train/Reward/StdByAction/5") == pytest.approx(1.0)
    # Actions never taken get zero (caller defaults via dict.get fallback).
    assert writer.value_for("Train/Reward/MeanByAction/3") == pytest.approx(0.0)


@pytest.mark.unit
def test_trade_metrics_from_tracker_recovers_pct_greedy(tmp_path):
    """Tier 2c: _trade_metrics_from_tracker reconstructs trades + greedy attribution."""
    trainer = _make_trainer_for_episode_summary(tmp_path)
    tracker = PerformanceTracker()
    tracker.add_initial_value(1000.0)

    # Open at price=100, hold one bar at 110, close at 120 → +20% gross.
    tracker.update(portfolio_value=1100.0, action=5, reward=0.0, position=1.0, balance=0.0, price=100.0, was_greedy=True)
    tracker.update(portfolio_value=1200.0, action=5, reward=0.0, position=1.0, balance=0.0, price=110.0, was_greedy=True)
    tracker.update(portfolio_value=1300.0, action=0, reward=0.0, position=0.0, balance=1300.0, price=120.0, was_greedy=False)

    trade_metrics = trainer._trade_metrics_from_tracker(tracker)

    assert trade_metrics, "Expected non-empty trade metrics"
    assert trade_metrics.get("trade_count") == pytest.approx(1.0)
    # The trade has 2 greedy in-trade entries (entry + hold) tracked by the
    # segmenter (closing flat step is not part of the in-trade span).
    assert trade_metrics.get("pct_greedy_actions") == pytest.approx(1.0)
    assert trade_metrics.get("hit_rate") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tier 2d: agent.greedy() context + EvalGap emission
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.tb_logging
def test_emit_eval_gap_scalars_emits_signed_diffs(tmp_path):
    """Tier 2d: Train/EvalGap/* equals (validation - recent train) for each metric."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()

    avg_val = {
        "total_return": 5.0,
        "sharpe_ratio": 0.4,
        "max_drawdown": 0.02,
        "transaction_costs": 1.5,
        "avg_reward": 0.3,
        "portfolio_value": 1050.0,
    }
    train_recent = {
        "total_return": 2.0,
        "sharpe_ratio": 0.1,
        "max_drawdown": 0.05,
        "transaction_costs": 1.0,
        "avg_reward": 0.1,
        "portfolio_value": 1020.0,
    }

    trainer._emit_eval_gap_scalars(avg_val, train_recent, episode=7)

    writer = trainer.writer
    assert writer.value_for("Train/EvalGap/TotalReturnPct") == pytest.approx(3.0)
    assert writer.value_for("Train/EvalGap/SharpeRatio") == pytest.approx(0.3)
    assert writer.value_for("Train/EvalGap/MaxDrawdown") == pytest.approx(-0.03)
    assert writer.value_for("Train/EvalGap/TransactionCosts") == pytest.approx(0.5)
    assert writer.value_for("Train/EvalGap/AvgReward") == pytest.approx(0.2)
    assert writer.value_for("Train/EvalGap/PortfolioValue") == pytest.approx(30.0)


@pytest.mark.unit
@pytest.mark.tb_logging
def test_emit_eval_gap_scalars_skips_missing_or_nonfinite(tmp_path):
    """Tier 2d: EvalGap silently skips a tag when either side is missing/NaN/Inf."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.writer = _CapturingWriter()

    avg_val = {"total_return": 5.0, "sharpe_ratio": float("nan"), "portfolio_value": 1050.0}
    train_recent = {"total_return": 2.0, "sharpe_ratio": 0.1}  # portfolio_value missing

    trainer._emit_eval_gap_scalars(avg_val, train_recent, episode=0)

    tags = set(trainer.writer.tags())
    assert "Train/EvalGap/TotalReturnPct" in tags
    assert "Train/EvalGap/SharpeRatio" not in tags  # NaN on val side
    assert "Train/EvalGap/PortfolioValue" not in tags  # missing on train side


@pytest.mark.unit
def test_eval_stochastic_flag_keeps_agent_in_training_mode_during_eval(tmp_path, monkeypatch):
    """Tier 2d: with eval_stochastic=True, _run_single_evaluation_episode does NOT flip eval mode."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.eval_stochastic = True
    trainer._render_on_reset_if_enabled = lambda *a, **kw: None

    states: list[bool] = []

    class _StubAgent:
        def __init__(self):
            self.training_mode = True

        def set_training_mode(self, training: bool) -> None:
            self.training_mode = bool(training)
            states.append(self.training_mode)

        def select_action(self, obs):
            return 0

        def select_action_with_provenance(self, obs):
            return 0, True

    trainer.agent = _StubAgent()

    class _StubEnv:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return (
                {"market_data": np.zeros(1), "account_state": np.zeros(1)},
                {"portfolio_value": 1000.0},
            )

        def step(self, action):
            self.steps += 1
            done = self.steps >= 1
            return (
                {"market_data": np.zeros(1), "account_state": np.zeros(1)},
                0.0,
                done,
                False,
                {"portfolio_value": 1000.0, "step_transaction_cost": 0.0, "position": 0.0, "price": 1.0},
            )

        def close(self):
            return None

    env = _StubEnv()
    trainer._run_single_evaluation_episode(env, context="validation")

    # No set_training_mode(False) call because eval_stochastic=True.
    assert all(s is True for s in states), f"Unexpected training-mode flips: {states}"


@pytest.mark.unit
def test_eval_default_runs_in_greedy_mode(tmp_path):
    """Tier 2d: default eval flips into eval mode on entry and restores on exit."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.eval_stochastic = False
    trainer._render_on_reset_if_enabled = lambda *a, **kw: None

    states: list[bool] = []

    class _StubAgent:
        def __init__(self):
            self.training_mode = True

        def set_training_mode(self, training: bool) -> None:
            self.training_mode = bool(training)
            states.append(self.training_mode)

        def select_action(self, obs):
            return 0

        def select_action_with_provenance(self, obs):
            return 0, True

    trainer.agent = _StubAgent()

    class _StubEnv:
        def __init__(self):
            self.steps = 0

        def reset(self):
            return (
                {"market_data": np.zeros(1), "account_state": np.zeros(1)},
                {"portfolio_value": 1000.0},
            )

        def step(self, action):
            self.steps += 1
            done = self.steps >= 1
            return (
                {"market_data": np.zeros(1), "account_state": np.zeros(1)},
                0.0,
                done,
                False,
                {"portfolio_value": 1000.0, "step_transaction_cost": 0.0, "position": 0.0, "price": 1.0},
            )

        def close(self):
            return None

    trainer._run_single_evaluation_episode(_StubEnv(), context="validation")

    # The agent flipped to eval mode (False) on entry and back to True on exit.
    assert states[0] is False
    assert states[-1] is True


# ---------------------------------------------------------------------------
# Validation cadence guard (_validate_validation_cadence_config)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_cadence_raises_when_freq_exceeds_episodes(tmp_path):
    """If validation_freq > num_episodes and there are val files, validation
    would never run and no `best` checkpoint would ever be saved — hard fail."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.validation_freq = 500
    with pytest.raises(RuntimeError, match="validation_freq"):
        trainer._validate_validation_cadence_config(num_episodes=100, has_val_files=True)


@pytest.mark.unit
def test_validate_cadence_no_val_files_skips_hard_fail(tmp_path):
    """Even with a wildly too-large validation_freq, no val files means the
    user has explicitly opted out of validation and we must not raise."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.validation_freq = 5000
    trainer._validate_validation_cadence_config(num_episodes=100, has_val_files=False)


@pytest.mark.unit
def test_validate_cadence_warns_when_fewer_than_5_runs(tmp_path, caplog):
    """Less than 5 validation runs across the whole training run is a loud
    WARNING (not a hard fail) about a sparse signal."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.validation_freq = 600
    with caplog.at_level("WARNING"):
        trainer._validate_validation_cadence_config(num_episodes=1000, has_val_files=True)
    assert any("less than 5" in rec.message.lower() or "validation_freq" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_validate_cadence_warns_when_min_episodes_exceeds_run_length(tmp_path, caplog):
    """``min_episodes_before_early_stopping > num_episodes`` silently means no
    `best` checkpoint will ever land on disk; warn loudly."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.validation_freq = 100
    trainer.min_episodes_before_early_stopping = 5000
    with caplog.at_level("WARNING"):
        trainer._validate_validation_cadence_config(num_episodes=1000, has_val_files=True)
    assert any("min_episodes_before_early_stopping" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_validate_cadence_silent_for_healthy_config(tmp_path, caplog):
    """A normal config (frequent validation, sensible early-stop threshold)
    must not warn or raise."""
    trainer = _create_minimal_trainer(tmp_path)
    trainer.validation_freq = 50
    trainer.min_episodes_before_early_stopping = 100
    with caplog.at_level("WARNING"):
        trainer._validate_validation_cadence_config(num_episodes=1000, has_val_files=True)
    bad_records = [rec for rec in caplog.records if "validation_freq" in rec.message or "min_episodes_before_early_stopping" in rec.message]
    assert bad_records == []


# ---------------------------------------------------------------------------
# Vectorized validation gate fix: simulate "completed_episodes jumps over a
# multiple of validation_freq" and assert the new "crossed-a-boundary" check
# triggers exactly one validation, while the legacy ``% == 0`` would have
# silently skipped it.
# ---------------------------------------------------------------------------


def _crossed(prev: int, current: int, freq: int) -> bool:
    """Replicates the new gate: True iff ``current`` and ``prev`` straddle a
    multiple of ``freq`` (or land on one)."""
    f = max(1, int(freq))
    return (current // f) > (prev // f)


@pytest.mark.unit
def test_vectorized_gate_fires_when_jump_lands_on_multiple():
    """``9 -> 10`` with freq 10 fires under both the legacy and new rules."""
    assert _crossed(prev=9, current=10, freq=10) is True


@pytest.mark.unit
def test_vectorized_gate_fires_when_jump_skips_over_multiple():
    """``9 -> 11`` with freq 10 should still fire under the new rule even
    though ``11 % 10 != 0`` (the legacy check would skip)."""
    assert _crossed(prev=9, current=11, freq=10) is True
    assert (11 % 10) != 0  # legacy gate would have been False


@pytest.mark.unit
def test_vectorized_gate_does_not_double_fire_within_window():
    """Two consecutive jumps inside the same ``freq`` window only fire once."""
    assert _crossed(prev=11, current=12, freq=10) is False
    assert _crossed(prev=15, current=18, freq=10) is False


@pytest.mark.unit
def test_vectorized_gate_handles_large_jumps():
    """An 8-env burst that lands at episode 207 with freq 200 still fires
    exactly once (not twice, even though it crosses 200)."""
    assert _crossed(prev=199, current=207, freq=200) is True
    assert _crossed(prev=207, current=208, freq=200) is False
