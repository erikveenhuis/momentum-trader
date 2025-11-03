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


@pytest.mark.unittest
def test_run_training_resume_uses_trainer_steps(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("dummy")

    data_file = tmp_path / "sample.csv"
    data_file.write_text("ticker,price\n")

    # Create necessary directories and files for DataManager
    processed_dir = tmp_path / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Move sample file to train directory and create corresponding validation/test samples
    train_file = train_dir / "sample.csv"
    train_file.write_text("ticker,price\n")

    val_file = val_dir / "sample.csv"
    val_file.write_text("ticker,price\n")

    test_file = test_dir / "sample.csv"
    test_file.write_text("ticker,price\n")

    class StubDataManager(DataManager):
        def __init__(self):
            DataManager.__init__(self, base_dir=str(tmp_path))
            self.organize_data()

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

    data_manager = StubDataManager()

    agent, trainer = run_training_module.run_training(config, data_manager, resume_training_flag=True)

    assert isinstance(trainer, StubTrainer)
    assert trainer.captured_start_total_steps == 1234
    assert agent.total_steps == 1234
