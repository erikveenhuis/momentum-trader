from types import SimpleNamespace

import momentum_train.run_training as run_training_module
import pytest
from momentum_train.data import DataManager


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


class StubAgent:
    def __init__(self, config, device, scaler):
        self.config = config
        self.device = device
        self.scaler = scaler
        self.total_steps = 0
        self.training_mode = True
        self.network = DummyNetwork()
        self.target_network = DummyNetwork()
        self.optimizer = SimpleNamespace(state_dict=lambda: {})
        self.scheduler = None
        self.lr_scheduler_enabled = False

    def load_state(self, checkpoint):
        self.total_steps = checkpoint.get("agent_total_steps", 0)
        return True

    def save_model(self, *_args, **_kwargs):
        return None

    def set_training_mode(self, training=True):
        self.training_mode = training


class StubTrainer:
    def __init__(self, agent, device, data_manager, config, scaler=None, writer=None):
        self.agent = agent
        self.device = device
        self.data_manager = data_manager
        self.config = config
        self.scaler = scaler
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
            "market_data": run_training_module.np.zeros((1, 1), dtype=float),
            "account_state": run_training_module.np.zeros(2, dtype=float),
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
def test_run_training_resume_uses_trainer_steps(monkeypatch, tmp_path):
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
    assert trainer.captured_start_total_steps == 1234
    assert agent.total_steps == 1234


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
