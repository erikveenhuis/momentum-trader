from types import SimpleNamespace

import momentum_train.run_training as run_training_module
import pytest
from momentum_train.data import DataManager
from momentum_train.run_training import ResumeFailedError


def _base_config(model_dir: str) -> dict:
    """Shared minimal config for resume-related tests.

    Kept module-level so resume-gate tests don't each re-declare the whole
    hyperparameter stanza — only the legacy resume tests below still inline
    their config (unchanged) for historical diff-stability.
    """
    return {
        "agent": {
            "gamma": 0.99,
            "lr": 0.001,
            "batch_size": 32,
            "replay_buffer_size": 1000,
            "target_update_freq": 100,
            "window_size": 1,
            "n_features": 1,
            "hidden_dim": 1,
            "num_actions": 2,
            "n_steps": 1,
            "num_atoms": 2,
            "v_min": -1.0,
            "v_max": 1.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 1000,
            "grad_clip_norm": 1.0,
            "epsilon_start": 0.3,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 1000,
            "entropy_coeff": 0.03,
        },
        "environment": {
            "window_size": 1,
            "initial_balance": 1000.0,
            "transaction_fee": 0.0,
            "reward_scale": 1.0,
            "invalid_action_penalty": -1.0,
        },
        "trainer": {
            "seed": 42,
            "warmup_steps": 10,
            "update_freq": 1,
            "log_freq": 10,
            "validation_freq": 100,
            "checkpoint_save_freq": 100,
            "reward_window": 10,
            "early_stopping_patience": 5,
            "min_validation_threshold": 0.0,
        },
        "run": {
            "episodes": 1,
            "model_dir": model_dir,
            "specific_file": None,
            "skip_evaluation": True,
        },
    }


class DummyParam:
    def __init__(self, n=1):
        self._n = n

    def numel(self):
        return self._n


class DummyNetwork:
    def parameters(self):
        return [DummyParam()]

    def state_dict(self):
        return {}

    def to(self, *_args, **_kwargs):
        return self


class StubBuffer:
    """Minimal buffer stub exposing both resume APIs (side-car + legacy).

    run_training.py always tries the side-car first and falls back to the
    legacy ``load_state_dict`` path for old checkpoints. These tests don't
    actually care how the buffer is restored -- they only care that
    *something* gets restored, because after the April 2026 OOM-kill
    incident we made the resume path fail loudly when it can't restore a
    buffer (to prevent the "hours of warmup + overwritten _final.pt"
    failure mode).
    """

    def __init__(self):
        self.loaded_from_path = None
        self.loaded_from_state = None

    def load_from_path(self, path):
        self.loaded_from_path = path

    def load_state_dict(self, state):
        self.loaded_from_state = state

    def __len__(self):
        return 0


class StubAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.total_steps = 0
        self.training_mode = True
        self.network = DummyNetwork()
        self.target_network = DummyNetwork()
        self.optimizer = SimpleNamespace(state_dict=lambda: {})
        self.scheduler = None
        self.lr_scheduler_enabled = False
        self.buffer = StubBuffer()

    def load_state(self, checkpoint):
        self.total_steps = checkpoint.get("agent_total_steps", 0)
        return True

    def save_model(self, *_args, **_kwargs):
        return None

    def set_training_mode(self, training=True):
        self.training_mode = training


class StubTrainer:
    def __init__(self, agent, device, data_manager, config, writer=None):
        self.agent = agent
        self.device = device
        self.data_manager = data_manager
        self.config = config
        self.writer = writer
        self.captured_start_total_steps = None

    def train(
        self,
        num_episodes,
        start_episode,
        start_total_steps,
        initial_best_score,
        initial_early_stopping_counter,
        specific_file=None,
    ):
        self.captured_start_total_steps = start_total_steps


class StubTradingEnv:
    def __init__(self, config):
        self.config = config

    def reset(self):
        obs = {
            "market_data": run_training_module.np.zeros((1, 12), dtype=float),
            "account_state": run_training_module.np.zeros(5, dtype=float),
        }
        info = {"portfolio_value": 1000.0}
        return obs, info

    def close(self):
        return None


def make_stub_data_manager(tmp_path):
    processed_dir = tmp_path / "processed"
    for split in ("train", "validation", "test"):
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "sample.csv").write_text("ticker,price\n")

    class StubDataManager(DataManager):
        def __init__(self):
            DataManager.__init__(self, base_dir=str(tmp_path))
            self.organize_data()

    return StubDataManager()


@pytest.mark.unit
def test_run_training_resume_preserves_separate_step_axes(monkeypatch, tmp_path):
    """Resume must keep the trainer env-step counter and the agent learn-step
    counter on independent axes.

    The trainer resumes from ``total_train_steps`` (env steps) so the env-step
    axis (episode rewards, rollout throughput) stays monotonic. The agent
    keeps whatever ``agent.load_state`` restored from ``agent_total_steps``
    (learn steps) so the learner-indexed TensorBoard tags
    (``Train/Loss``, ``Train/CategoricalTarget/Mean``, NoisyNet sigma, etc.)
    stay monotonic without forward jumps. Overwriting one with the other was
    a real bug that produced ~10x step-axis jumps on every resume.
    """
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("dummy")

    def fake_find_latest(model_dir, prefix):
        return str(checkpoint_path)

    def fake_load_checkpoint(path):
        assert str(path) == str(checkpoint_path)
        return {
            "episode": 5,
            "total_train_steps": 1234,
            "best_validation_metric": 0.5,
            "early_stopping_counter": 2,
            "agent_config": config["agent"],
            "agent_total_steps": 400,
            "network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
            # Legacy buffer_state is included so the strict "must restore a
            # buffer on --resume" gate in run_training.py is satisfied.
            # This test is about step-axis separation, not buffer I/O.
            "buffer_state": {"legacy-stub": True},
        }

    def dummy_trading_env_config(**kwargs):
        return SimpleNamespace(**kwargs)

    def dummy_summary_writer(*_args, **_kwargs):
        return SimpleNamespace(add_scalar=lambda *a, **k: None, close=lambda: None)

    config = {
        "agent": {
            "gamma": 0.99,
            "lr": 0.001,
            "batch_size": 32,
            "replay_buffer_size": 1000,
            "target_update_freq": 100,
            "window_size": 1,
            "n_features": 1,
            "hidden_dim": 1,
            "num_actions": 2,
            "n_steps": 1,
            "num_atoms": 2,
            "v_min": -1.0,
            "v_max": 1.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 1000,
            "grad_clip_norm": 1.0,
            "epsilon_start": 0.3,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 1000,
            "entropy_coeff": 0.03,
        },
        "environment": {
            "window_size": 1,
            "initial_balance": 1000.0,
            "transaction_fee": 0.0,
            "reward_scale": 1.0,
            "invalid_action_penalty": -1.0,
        },
        "trainer": {
            "seed": 42,
            "warmup_steps": 10,
            "update_freq": 1,
            "log_freq": 10,
            "validation_freq": 100,
            "checkpoint_save_freq": 100,
            "reward_window": 10,
            "early_stopping_patience": 5,
            "min_validation_threshold": 0.0,
        },
        "run": {
            "episodes": 1,
            "model_dir": str(tmp_path),
            "specific_file": None,
            "skip_evaluation": True,
        },
    }

    monkeypatch.setattr(run_training_module, "RainbowDQNAgent", StubAgent)
    monkeypatch.setattr(run_training_module, "RainbowTrainerModule", StubTrainer)
    monkeypatch.setattr(run_training_module, "find_latest_checkpoint", fake_find_latest)
    monkeypatch.setattr(run_training_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(run_training_module, "TradingEnv", StubTradingEnv)
    monkeypatch.setattr(run_training_module, "TradingEnvConfig", dummy_trading_env_config)
    monkeypatch.setattr(run_training_module, "SummaryWriter", dummy_summary_writer)

    data_manager = make_stub_data_manager(tmp_path)

    agent, trainer = run_training_module.run_training(
        config,
        data_manager,
        resume_training_flag=True,
        reset_lr_on_resume=False,
    )

    assert isinstance(trainer, StubTrainer)
    # Trainer resumes from the env-step counter in the checkpoint.
    assert trainer.captured_start_total_steps == 1234
    # Agent keeps its own learn-step counter (400, from agent_total_steps)
    # and is NOT force-synced to the 1234 env-step value. Collapsing these
    # two counters is the exact bug that caused the TensorBoard X-axis
    # discontinuity on agent-indexed tags after every resume.
    assert agent.total_steps == 400


@pytest.mark.unit
def test_resume_tensorboard_log_dir_reuses_existing_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    model_dir_rel = "models"
    model_dir_path = tmp_path / model_dir_rel
    model_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir_path / "checkpoint_trainer_latest.pt"
    checkpoint_path.write_text("dummy")

    run_id = "20251111-140948"
    existing_log_dir = model_dir_path / "runs" / run_id
    existing_log_dir.mkdir(parents=True, exist_ok=True)

    def fake_find_latest(model_dir, prefix):
        assert model_dir == model_dir_rel
        assert prefix == "checkpoint_trainer"
        return str(checkpoint_path)

    def fake_load_checkpoint(path):
        return {
            "episode": 2,
            "total_train_steps": 200,
            "best_validation_metric": 0.25,
            "early_stopping_counter": 1,
            "agent_config": config["agent"],
            "agent_total_steps": 200,
            "network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
            "tensorboard_log_dir": f"{model_dir_rel}/runs/{run_id}",
            # Satisfies the strict "--resume must restore a buffer" gate;
            # this test exercises TB log-dir reuse, not buffer I/O.
            "buffer_state": {"legacy-stub": True},
        }

    def dummy_trading_env_config(**kwargs):
        return SimpleNamespace(**kwargs)

    writer_calls = []

    def capturing_summary_writer(*_args, **kwargs):
        writer_calls.append({"args": _args, "kwargs": kwargs})
        log_dir_value = kwargs.get("log_dir") if kwargs else None
        return SimpleNamespace(add_scalar=lambda *a, **k: None, close=lambda: None, log_dir=log_dir_value)

    config = {
        "agent": {
            "gamma": 0.99,
            "lr": 0.001,
            "batch_size": 32,
            "replay_buffer_size": 1000,
            "target_update_freq": 100,
            "window_size": 1,
            "n_features": 1,
            "hidden_dim": 1,
            "num_actions": 2,
            "n_steps": 1,
            "num_atoms": 2,
            "v_min": -1.0,
            "v_max": 1.0,
            "alpha": 0.6,
            "beta_start": 0.4,
            "beta_frames": 1000,
            "grad_clip_norm": 1.0,
            "epsilon_start": 0.3,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 1000,
            "entropy_coeff": 0.03,
        },
        "environment": {
            "window_size": 1,
            "initial_balance": 1000.0,
            "transaction_fee": 0.0,
            "reward_scale": 1.0,
            "invalid_action_penalty": -1.0,
        },
        "trainer": {
            "seed": 42,
            "warmup_steps": 10,
            "update_freq": 1,
            "log_freq": 10,
            "validation_freq": 100,
            "checkpoint_save_freq": 100,
            "reward_window": 10,
            "early_stopping_patience": 5,
            "min_validation_threshold": 0.0,
        },
        "run": {
            "episodes": 1,
            "model_dir": model_dir_rel,
            "specific_file": None,
            "skip_evaluation": True,
        },
    }

    monkeypatch.setattr(run_training_module, "RainbowDQNAgent", StubAgent)
    monkeypatch.setattr(run_training_module, "RainbowTrainerModule", StubTrainer)
    monkeypatch.setattr(run_training_module, "find_latest_checkpoint", fake_find_latest)
    monkeypatch.setattr(run_training_module, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(run_training_module, "TradingEnv", StubTradingEnv)
    monkeypatch.setattr(run_training_module, "TradingEnvConfig", dummy_trading_env_config)
    monkeypatch.setattr(run_training_module, "SummaryWriter", capturing_summary_writer)

    data_manager = make_stub_data_manager(tmp_path)

    run_training_module.run_training(
        config,
        data_manager,
        resume_training_flag=True,
        reset_lr_on_resume=False,
    )

    assert writer_calls, "Expected SummaryWriter to be invoked during resume."
    writer_kwargs = writer_calls[0]["kwargs"]
    assert str(existing_log_dir.resolve()) == writer_kwargs.get("log_dir")
    assert writer_kwargs.get("purge_step") == 200

    nested_dir = existing_log_dir / model_dir_rel
    assert not nested_dir.exists(), f"Unexpected nested log directory created at {nested_dir}"


# ---------------------------------------------------------------------------
# Resume-failure gates. These regression-guard the April 2026 incident where a
# zero-byte checkpoint (OOM mid-save) caused ``--resume`` to silently fall
# back to a fresh run, discarding 13 hours of training and clobbering
# ``rainbow_transformer_final_agent_state.pt`` on the next finalize.
# ---------------------------------------------------------------------------


def _apply_common_resume_monkeypatches(monkeypatch, tmp_path, *, find_latest, load_checkpoint, agent_class=StubAgent):
    """Wire up the common stubs every resume-gate test needs."""

    def dummy_trading_env_config(**kwargs):
        return SimpleNamespace(**kwargs)

    def dummy_summary_writer(*_args, **_kwargs):
        return SimpleNamespace(add_scalar=lambda *a, **k: None, close=lambda: None)

    monkeypatch.setattr(run_training_module, "RainbowDQNAgent", agent_class)
    monkeypatch.setattr(run_training_module, "RainbowTrainerModule", StubTrainer)
    monkeypatch.setattr(run_training_module, "find_latest_checkpoint", find_latest)
    monkeypatch.setattr(run_training_module, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(run_training_module, "TradingEnv", StubTradingEnv)
    monkeypatch.setattr(run_training_module, "TradingEnvConfig", dummy_trading_env_config)
    monkeypatch.setattr(run_training_module, "SummaryWriter", dummy_summary_writer)


@pytest.mark.unit
def test_resume_raises_when_no_checkpoint_found(monkeypatch, tmp_path):
    """``--resume`` with no checkpoint must hard-fail, not silently start fresh."""
    _apply_common_resume_monkeypatches(
        monkeypatch,
        tmp_path,
        find_latest=lambda model_dir, prefix: None,
        load_checkpoint=lambda path: pytest.fail("load_checkpoint should not be called when no path is found"),
    )

    config = _base_config(str(tmp_path))
    data_manager = make_stub_data_manager(tmp_path)

    with pytest.raises(ResumeFailedError, match="no usable checkpoint"):
        run_training_module.run_training(
            config,
            data_manager,
            resume_training_flag=True,
            reset_lr_on_resume=False,
        )


@pytest.mark.unit
def test_resume_raises_when_load_checkpoint_returns_none(monkeypatch, tmp_path):
    """The exact April-22 failure mode: path found but file is zero-byte / corrupt."""
    stub_path = tmp_path / "checkpoint_trainer_latest_20260422_ep3301_reward-inf.pt"
    stub_path.write_bytes(b"")

    _apply_common_resume_monkeypatches(
        monkeypatch,
        tmp_path,
        find_latest=lambda model_dir, prefix: str(stub_path),
        load_checkpoint=lambda path: None,
    )

    config = _base_config(str(tmp_path))
    data_manager = make_stub_data_manager(tmp_path)

    with pytest.raises(ResumeFailedError, match="could not be loaded"):
        run_training_module.run_training(
            config,
            data_manager,
            resume_training_flag=True,
            reset_lr_on_resume=False,
        )


@pytest.mark.unit
def test_resume_raises_when_agent_load_state_returns_false(monkeypatch, tmp_path):
    """If agent.load_state rejects the checkpoint, do not silently use fresh weights."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"valid-bytes")
    config = _base_config(str(tmp_path))

    def fake_load_checkpoint(path):
        return {
            "episode": 5,
            "total_train_steps": 1234,
            "best_validation_metric": 0.5,
            "early_stopping_counter": 0,
            "agent_config": config["agent"],
            "agent_total_steps": 400,
            "network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
        }

    class RejectingAgent(StubAgent):
        def load_state(self, checkpoint):
            return False

    _apply_common_resume_monkeypatches(
        monkeypatch,
        tmp_path,
        find_latest=lambda model_dir, prefix: str(checkpoint_path),
        load_checkpoint=fake_load_checkpoint,
        agent_class=RejectingAgent,
    )

    data_manager = make_stub_data_manager(tmp_path)

    with pytest.raises(ResumeFailedError, match="load_state returned False"):
        run_training_module.run_training(
            config,
            data_manager,
            resume_training_flag=True,
            reset_lr_on_resume=False,
        )


@pytest.mark.unit
def test_resume_raises_when_agent_instantiation_raises(monkeypatch, tmp_path):
    """An exception during agent construction must surface, not swallow into fresh init."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"valid-bytes")
    config = _base_config(str(tmp_path))

    def fake_load_checkpoint(path):
        return {
            "episode": 5,
            "total_train_steps": 1234,
            "best_validation_metric": 0.5,
            "early_stopping_counter": 0,
            "agent_config": config["agent"],
            "agent_total_steps": 400,
            "network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
        }

    class BrokenAgent:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("shape mismatch in network constructor")

    _apply_common_resume_monkeypatches(
        monkeypatch,
        tmp_path,
        find_latest=lambda model_dir, prefix: str(checkpoint_path),
        load_checkpoint=fake_load_checkpoint,
        agent_class=BrokenAgent,
    )

    data_manager = make_stub_data_manager(tmp_path)

    with pytest.raises(ResumeFailedError, match="instantiating the agent") as excinfo:
        run_training_module.run_training(
            config,
            data_manager,
            resume_training_flag=True,
            reset_lr_on_resume=False,
        )
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "shape mismatch" in str(excinfo.value.__cause__)


@pytest.mark.unit
def test_resume_raises_when_no_buffer_state_in_checkpoint(monkeypatch, tmp_path):
    """``--resume`` must fail loudly when the checkpoint has no buffer to restore.

    After the April 2026 memmap-sidecar migration, every ``.pt`` written by
    the trainer points at a sibling ``<name>.buffer/`` directory via
    ``buffer_sidecar_relpath``. If neither that key nor the legacy inline
    ``buffer_state`` dict is present, there is literally no replay buffer
    to resume from -- silently continuing would re-run warmup (hours of
    lost training) and on the next ``_finalize_training`` overwrite
    ``rainbow_transformer_final_agent_state.pt`` with what is effectively
    a fresh network. This test is the regression guard for exactly that
    fall-through path.
    """
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"valid-bytes")
    config = _base_config(str(tmp_path))

    def fake_load_checkpoint(path):
        # Note the deliberate absence of both ``buffer_state`` and
        # ``buffer_sidecar_relpath``.
        return {
            "episode": 5,
            "total_train_steps": 1234,
            "best_validation_metric": 0.5,
            "early_stopping_counter": 0,
            "agent_config": config["agent"],
            "agent_total_steps": 400,
            "network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
        }

    _apply_common_resume_monkeypatches(
        monkeypatch,
        tmp_path,
        find_latest=lambda model_dir, prefix: str(checkpoint_path),
        load_checkpoint=fake_load_checkpoint,
    )

    data_manager = make_stub_data_manager(tmp_path)

    with pytest.raises(ResumeFailedError, match="no replay buffer to restore"):
        run_training_module.run_training(
            config,
            data_manager,
            resume_training_flag=True,
            reset_lr_on_resume=False,
        )
