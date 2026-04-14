"""Integration tests for vectorized (multi-env) training."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from momentum_agent import RainbowDQNAgent
from momentum_train.data import DataManager
from momentum_train.trainer import RainbowTrainerModule

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _copy_random_files(src_dir: Path, dst_dir: Path, count: int, rng: random.Random) -> list[Path]:
    if not src_dir.is_dir():
        pytest.skip(f"Source directory not found: {src_dir}")
    source_files = sorted(src_dir.glob("*.npz"))
    if not source_files:
        source_files = sorted(src_dir.glob("*.csv"))
    if len(source_files) < count:
        pytest.skip(f"Not enough data files in {src_dir} (needed {count}, found {len(source_files)})")
    dst_dir.mkdir(parents=True, exist_ok=True)
    selected = rng.sample(source_files, count)
    for f in selected:
        shutil.copy2(f, dst_dir / f.name)
    return selected


@pytest.fixture
def vec_training_setup(tmp_path):
    """Set up a small data slice and config for vectorized training tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")

    rng = random.Random(9999)
    prod = PROJECT_ROOT / "data" / "processed"
    processed = tmp_path / "processed"
    _copy_random_files(prod / "train", processed / "train", 10, rng)
    _copy_random_files(prod / "validation", processed / "validation", 3, rng)
    _copy_random_files(prod / "test", processed / "test", 2, rng)

    dm = DataManager(base_dir=str(tmp_path))
    dm.organize_data()

    config = yaml.safe_load((PROJECT_ROOT / "config" / "training_config.yaml").read_text())
    config["run"]["model_dir"] = str(tmp_path / "models")
    config["run"]["episodes"] = 10
    config["run"]["skip_evaluation"] = True
    config["agent"]["seed"] = config["trainer"]["seed"]
    config["trainer"]["validation_freq"] = 10
    config["trainer"]["checkpoint_save_freq"] = 10

    return config, dm


@pytest.mark.integration
def test_vectorized_training_completes(vec_training_setup):
    """Vectorized training with num_vector_envs=2 completes without error."""
    config, dm = vec_training_setup
    config["trainer"]["num_vector_envs"] = 2

    agent = RainbowDQNAgent(config=config["agent"], device="cuda")
    trainer = RainbowTrainerModule(
        agent=agent,
        device=torch.device("cuda"),
        data_manager=dm,
        config=config,
        writer=None,
    )

    trainer.train(
        num_episodes=config["run"]["episodes"],
        start_episode=0,
        start_total_steps=0,
        initial_best_score=float("-inf"),
        initial_early_stopping_counter=0,
    )

    assert trainer.total_train_steps > 0, "No training steps recorded"


@pytest.mark.integration
def test_vectorized_buffer_fills(vec_training_setup):
    """Replay buffer receives transitions from all parallel envs."""
    config, dm = vec_training_setup
    config["trainer"]["num_vector_envs"] = 2

    agent = RainbowDQNAgent(config=config["agent"], device="cuda")
    trainer = RainbowTrainerModule(
        agent=agent,
        device=torch.device("cuda"),
        data_manager=dm,
        config=config,
        writer=None,
    )

    trainer.train(
        num_episodes=config["run"]["episodes"],
        start_episode=0,
        start_total_steps=0,
        initial_best_score=float("-inf"),
        initial_early_stopping_counter=0,
    )

    assert len(agent.buffer) > 0, "Replay buffer is empty after vectorized training"


@pytest.mark.integration
def test_vectorized_matches_single_env_buffer_growth(vec_training_setup):
    """With 2 envs, buffer should grow at roughly 2x the rate of 1 env (same num_episodes)."""
    config, dm = vec_training_setup

    # Single-env run
    config_1 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    config_1["trainer"] = dict(config["trainer"])
    config_1["trainer"]["num_vector_envs"] = 1

    agent_1 = RainbowDQNAgent(config=config_1["agent"], device="cuda")
    trainer_1 = RainbowTrainerModule(
        agent=agent_1,
        device=torch.device("cuda"),
        data_manager=dm,
        config=config_1,
        writer=None,
    )
    trainer_1.train(
        num_episodes=config_1["run"]["episodes"],
        start_episode=0,
        start_total_steps=0,
        initial_best_score=float("-inf"),
        initial_early_stopping_counter=0,
    )
    buf_size_1 = len(agent_1.buffer)

    # 2-env run
    config_2 = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
    config_2["trainer"] = dict(config["trainer"])
    config_2["trainer"]["num_vector_envs"] = 2

    agent_2 = RainbowDQNAgent(config=config_2["agent"], device="cuda")
    trainer_2 = RainbowTrainerModule(
        agent=agent_2,
        device=torch.device("cuda"),
        data_manager=dm,
        config=config_2,
        writer=None,
    )
    trainer_2.train(
        num_episodes=config_2["run"]["episodes"],
        start_episode=0,
        start_total_steps=0,
        initial_best_score=float("-inf"),
        initial_early_stopping_counter=0,
    )
    buf_size_2 = len(agent_2.buffer)

    assert buf_size_1 > 0
    assert buf_size_2 > 0


@pytest.mark.benchmark
def test_vectorized_speedup_benchmark(vec_training_setup, capsys):
    """Compare wall-clock time, throughput, and GPU utilization for N=1,4,8.

    Run with: pytest -k test_vectorized_speedup_benchmark -s
    Uses low warmup so the benchmark finishes quickly.
    """
    import threading
    import time

    config, dm = vec_training_setup
    episodes = 8
    results = {}

    has_pynvml = False
    try:
        import pynvml

        pynvml.nvmlInit()
        has_pynvml = True
    except Exception:
        pass

    for n_envs in [1, 4, 8]:
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
        cfg["trainer"] = dict(config["trainer"])
        cfg["trainer"]["num_vector_envs"] = n_envs
        cfg["trainer"]["warmup_steps"] = 500
        cfg["trainer"]["validation_freq"] = episodes + 1
        cfg["trainer"]["checkpoint_save_freq"] = episodes + 1
        cfg["run"] = dict(config["run"])
        cfg["run"]["episodes"] = episodes

        agent = RainbowDQNAgent(config=cfg["agent"], device="cuda")
        trainer = RainbowTrainerModule(
            agent=agent,
            device=torch.device("cuda"),
            data_manager=dm,
            config=cfg,
            writer=None,
        )

        gpu_samples: list[tuple[int, int]] = []
        stop_event = threading.Event()

        def _sample_gpu():
            if not has_pynvml:
                return
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not stop_event.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_samples.append((util.gpu, int(mem_info.used * 100 / mem_info.total)))
                stop_event.wait(0.1)

        sampler = threading.Thread(target=_sample_gpu, daemon=True)
        sampler.start()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        trainer.train(
            num_episodes=episodes,
            start_episode=0,
            start_total_steps=0,
            initial_best_score=float("-inf"),
            initial_early_stopping_counter=0,
        )
        elapsed = time.perf_counter() - t0
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        stop_event.set()
        sampler.join(timeout=1)

        gpu_util_avg = int(np.mean([s[0] for s in gpu_samples])) if gpu_samples else -1
        gpu_util_max = int(np.max([s[0] for s in gpu_samples])) if gpu_samples else -1
        mem_util_avg = int(np.mean([s[1] for s in gpu_samples])) if gpu_samples else -1

        results[n_envs] = {
            "wall_clock_s": elapsed,
            "total_steps": trainer.total_train_steps,
            "episodes": episodes,
            "peak_mem_mb": peak_mem_mb,
            "gpu_util_avg": gpu_util_avg,
            "gpu_util_max": gpu_util_max,
            "mem_util_avg": mem_util_avg,
            "samples": len(gpu_samples),
        }

    with capsys.disabled():
        print("\n" + "=" * 78)
        print(f"VECTORIZED TRAINING BENCHMARK  ({episodes} episodes each)")
        print("=" * 78)
        header = f"  {'N':>2s}  {'time':>7s}  {'eps/sec':>7s}  {'env_steps':>9s}  {'GPU%avg':>7s}  {'GPU%max':>7s}  {'VRAM%':>5s}  {'peakMB':>7s}"
        print(header)
        print("  " + "-" * 74)
        for n_envs, r in sorted(results.items()):
            eps_per_sec = r["episodes"] / r["wall_clock_s"]
            gpu_avg = f"{r['gpu_util_avg']}%" if r["gpu_util_avg"] >= 0 else "n/a"
            gpu_max = f"{r['gpu_util_max']}%" if r["gpu_util_max"] >= 0 else "n/a"
            mem_avg = f"{r['mem_util_avg']}%" if r["mem_util_avg"] >= 0 else "n/a"
            print(
                f"  {n_envs:>2d}  {r['wall_clock_s']:>6.2f}s  {eps_per_sec:>7.1f}  "
                f"{r['total_steps']:>9,d}  {gpu_avg:>7s}  {gpu_max:>7s}  "
                f"{mem_avg:>5s}  {r['peak_mem_mb']:>6.0f}M"
            )

        r1 = results[1]
        print()
        for n_envs in [4, 8]:
            if n_envs in results:
                speedup = r1["wall_clock_s"] / results[n_envs]["wall_clock_s"]
                print(f"  Wall-clock speedup N={n_envs} vs N=1: {speedup:.2f}x")
        print("=" * 78)


@pytest.mark.unit
def test_agent_per_env_nstep_isolation():
    """N-step buffers are independent across env_ids -- done in one env doesn't affect another."""
    config = {
        "seed": 42,
        "gamma": 0.99,
        "lr": 1e-4,
        "batch_size": 32,
        "target_update_freq": 100,
        "n_steps": 3,
        "num_atoms": 51,
        "v_min": -1,
        "v_max": 1,
        "num_actions": 6,
        "window_size": 5,
        "n_features": 12,
        "hidden_dim": 64,
        "replay_buffer_size": 1000,
        "alpha": 0.6,
        "beta_start": 0.4,
        "beta_frames": 1000,
        "grad_clip_norm": 1.0,
        "epsilon_start": 0.0,
        "epsilon_end": 0.0,
        "epsilon_decay_steps": 1,
        "entropy_coeff": 0.0,
        "nhead": 2,
        "num_encoder_layers": 1,
        "dim_feedforward": 64,
        "transformer_dropout": 0.0,
        "store_partial_n_step": True,
    }

    agent = RainbowDQNAgent(config=config, device="cpu", inference_only=True)
    agent.set_num_envs(2)

    def _obs():
        return {
            "market_data": np.random.randn(5, 12).astype(np.float32),
            "account_state": np.random.randn(5).astype(np.float32).clip(-1, 1),
        }

    # Feed 2 transitions to env 0, 1 to env 1
    agent.store_transition(_obs(), 0, 0.1, _obs(), False, env_id=0)
    agent.store_transition(_obs(), 1, 0.2, _obs(), False, env_id=0)
    agent.store_transition(_obs(), 2, 0.3, _obs(), False, env_id=1)

    assert len(agent._n_step_buffers[0]) == 2
    assert len(agent._n_step_buffers[1]) == 1

    # Done on env 0 should not clear env 1's buffer
    agent.store_transition(_obs(), 0, -0.1, _obs(), True, env_id=0)
    assert agent._n_step_needs_reset_flags[0] is True
    assert agent._n_step_needs_reset_flags[1] is False
    assert len(agent._n_step_buffers[1]) == 1

    # Next transition on env 0 clears its buffer (lazy reset)
    agent.store_transition(_obs(), 0, 0.5, _obs(), False, env_id=0)
    assert len(agent._n_step_buffers[0]) == 1
    assert agent._n_step_needs_reset_flags[0] is False
    assert len(agent._n_step_buffers[1]) == 1
