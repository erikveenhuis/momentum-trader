#!/usr/bin/env python3
# Main training script for transformer trader (Rainbow DQN version)

import argparse  # Added for command-line arguments
import json
import logging
import os
import sys  # Added sys module
import time  # Added for timestamping log directories
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml  # Added for config loading
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger, setup_package_logging
from momentum_env import TradingEnv, TradingEnvConfig

# AMP uses bfloat16 autocast (no GradScaler needed)
# --- Add TensorBoard import ---
from torch.utils.tensorboard import SummaryWriter

from .data import DataManager
from .trainer import RainbowTrainerModule
from .utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint
from .utils.utils import get_random_data_file, set_seeds

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Assume environment, agent (DDPG version), trainer (DDPG version), utils are correct
# print("Imported TradingEnv") # <-- Removed print

# Use the new unified logging setup function
# from hyperparameters import parse_args # Import argument parser

# Get logger instance
logger = get_logger("momentum_train.Main")


DEFAULT_LOG_LEVEL_OVERRIDES: dict[str, Any] = {
    "momentum_train.Main": logging.INFO,
    "Trainer": logging.INFO,
    "Agent": logging.INFO,
    "DataManager": logging.INFO,
    "TransformerModel": logging.INFO,
    "Buffer": logging.INFO,
    "Metrics": logging.INFO,
    "Evaluation": logging.INFO,
}


def configure_logging(cli_log_level: str | None = None, config: dict[str, Any] | None = None) -> None:
    """Configure logging for the momentum_train package."""

    config = config or {}

    log_filename = config.get("log_filename", "training.log")
    logs_dir = config.get("logs_dir")

    # Determine base levels, allowing CLI to take precedence when provided.
    root_level = cli_log_level or config.get("root_level", logging.INFO)
    console_level = cli_log_level or config.get("console_level", root_level)
    file_level = cli_log_level or config.get("file_level", root_level)

    # Merge default overrides with config-specified overrides (config wins)
    level_overrides = DEFAULT_LOG_LEVEL_OVERRIDES.copy()
    config_overrides = config.get("level_overrides") or {}
    if isinstance(config_overrides, dict):
        level_overrides.update(config_overrides)
    else:
        logger.warning("logging.level_overrides must be a dictionary; ignoring provided value.")

    setup_package_logging(
        "momentum_train",
        log_filename=log_filename,
        root_level=root_level,
        console_level=console_level,
        file_level=file_level,
        logs_dir=logs_dir,
        level_overrides=level_overrides,
    )


def _safe_add_scalar(writer: SummaryWriter, tag: str, value: Any, step: int) -> None:
    """Write a scalar only when finite — TB rejects NaN/Inf and litters warnings otherwise."""
    if value is None:
        return
    try:
        scalar_value = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(scalar_value):
        return
    writer.add_scalar(tag, scalar_value, step)


def _emit_action_rates(
    writer: SummaryWriter,
    tag_prefix: str,
    detailed_results: list[dict],
    step: int,
) -> None:
    """Average per-action rates across files and emit one scalar per action index."""
    if not detailed_results:
        return
    aggregate: dict[int, list[float]] = {}
    for item in detailed_results:
        action_rates = item.get("action_rates") if isinstance(item, dict) else None
        if not action_rates:
            counts = item.get("action_counts") if isinstance(item, dict) else None
            steps_count = item.get("steps") if isinstance(item, dict) else None
            if not counts or not steps_count:
                continue
            try:
                steps_total = int(steps_count)
            except (TypeError, ValueError):
                continue
            if steps_total <= 0:
                continue
            action_rates = {k: int(v) / steps_total for k, v in counts.items()}
        for k, v in action_rates.items():
            try:
                k_int = int(k)
                v_float = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v_float):
                continue
            aggregate.setdefault(k_int, []).append(v_float)
    for k_int, values in sorted(aggregate.items()):
        if not values:
            continue
        _safe_add_scalar(writer, f"{tag_prefix}/{k_int}", float(np.mean(values)), step)


def _emit_trade_metrics(
    writer: SummaryWriter,
    tag_prefix: str,
    detailed_results: list[dict],
    step: int,
) -> None:
    """Aggregate per-file trade KPIs into a single scalar per metric (mean across files)."""
    if not detailed_results:
        return
    trade_payloads = [
        item.get("trade_metrics") for item in detailed_results if isinstance(item, dict) and isinstance(item.get("trade_metrics"), dict)
    ]
    if not trade_payloads:
        return
    keys = set()
    for payload in trade_payloads:
        keys.update(payload.keys())
    for key in sorted(keys):
        values = [
            float(payload[key]) for payload in trade_payloads if key in payload and payload[key] is not None and np.isfinite(payload[key])
        ]
        if not values:
            continue
        _safe_add_scalar(writer, f"{tag_prefix}/{key}", float(np.mean(values)), step)

    # Tier 2b: aggregate per-trade PnL across all files into a histogram so the
    # full sniper-PnL distribution (not just its mean) is visible in TB.
    trade_pnls: list[float] = []
    for item in detailed_results:
        if not isinstance(item, dict):
            continue
        for trade in item.get("trades", []) or []:
            pnl = trade.get("pnl_pct") if isinstance(trade, dict) else None
            try:
                pnl_f = float(pnl) if pnl is not None else None
            except (TypeError, ValueError):
                continue
            if pnl_f is None or not np.isfinite(pnl_f):
                continue
            trade_pnls.append(pnl_f)
    if trade_pnls:
        try:
            writer.add_histogram(
                f"{tag_prefix}/PnLDistribution",
                np.asarray(trade_pnls, dtype=np.float32),
                step,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to emit %s/PnLDistribution histogram: %s", tag_prefix, exc)


def evaluate_on_test_data(
    agent: RainbowDQNAgent,
    trainer: RainbowTrainerModule,
    config: dict,
    *,
    writer: SummaryWriter | None = None,
    tb_step: int | None = None,
) -> None:
    """Run evaluation across the test split and log aggregate results.

    If ``writer`` is provided (or ``trainer.writer`` is set), per-test scalars are also
    mirrored to TensorBoard under ``Test/*`` and ``Test/Trade/*`` so the test pass is
    visible alongside training/validation curves. ``tb_step`` controls the x-axis: it
    defaults to ``trainer.agent.total_steps`` so the test scalar lines up with the last
    training step in the same run.
    """
    if not hasattr(trainer, "data_manager"):
        logger.error("Trainer does not expose a data_manager; cannot evaluate on test data.")
        return

    if writer is None:
        writer = getattr(trainer, "writer", None)
    if tb_step is None:
        agent_for_step = getattr(trainer, "agent", None) or agent
        tb_step_raw = getattr(agent_for_step, "total_steps", 0) if agent_for_step is not None else 0
        try:
            tb_step = int(tb_step_raw or 0)
        except (TypeError, ValueError):
            tb_step = 0

    try:
        test_files = trainer.data_manager.get_test_files()
    except Exception as exc:
        logger.error(f"Unable to retrieve test files for evaluation: {exc}")
        trainer.close_cached_environments()
        return

    if not test_files:
        logger.warning("Test evaluation skipped: no test files available.")
        trainer.close_cached_environments()
        return

    logger.info("============================================")
    logger.info(f"RUNNING TEST EVALUATION ON {len(test_files)} FILES")
    logger.info("============================================")

    all_file_metrics = []
    detailed_results = []
    episode_scores = []

    try:
        for i, test_file in enumerate(test_files):
            try:
                result = trainer._validate_single_file(
                    test_file, validation_episode=i, total_validation_episodes=len(test_files), context="test"
                )
            except Exception as exc:  # Defensive: _validate_single_file already catches most errors
                logger.error(f"Unexpected error while evaluating {test_file.name}: {exc}")
                continue

            if not result:
                continue

            all_file_metrics.append(result.get("file_metrics", {}))
            detailed_results.append(result.get("detailed_result", {}))
            episode_scores.append(result.get("episode_score", -np.inf))

        if not all_file_metrics:
            logger.warning("Test evaluation produced no valid metrics.")
            return

        avg_metrics = trainer._calculate_average_validation_metrics(all_file_metrics)

        finite_scores = [score for score in episode_scores if np.isfinite(score)]
        average_score = float(np.mean(finite_scores)) if finite_scores else -np.inf

        logger.info("\n=== TEST EVALUATION SUMMARY ===")
        logger.info(f"Average Episode Score: {average_score:.4f}")
        logger.info(f"Average Reward: {avg_metrics['avg_reward']:.2f}")
        logger.info(f"Average Portfolio: ${avg_metrics['portfolio_value']:.2f}")
        logger.info(f"Average Return: {avg_metrics['total_return']:.2f}%")
        logger.info(f"Average Sharpe: {avg_metrics['sharpe_ratio']:.4f}")
        logger.info(f"Average Max Drawdown: {avg_metrics['max_drawdown'] * 100:.2f}%")
        logger.info(f"Average Transaction Costs: ${avg_metrics['transaction_costs']:.2f}")
        logger.info("============================================")

        # Mirror aggregate test results to TensorBoard (Tier 1a).
        # Portfolio-level scalars are namespaced under Test/Portfolio/* per the KPI
        # hierarchy in the comprehensive logging plan: they're informational, not the
        # sniper-edge optimisation target.
        if writer is not None:
            try:
                _safe_add_scalar(writer, "Test/Score", average_score, tb_step)
                _safe_add_scalar(writer, "Test/Avg Reward", avg_metrics.get("avg_reward"), tb_step)
                _safe_add_scalar(writer, "Test/Avg Portfolio", avg_metrics.get("portfolio_value"), tb_step)
                _safe_add_scalar(writer, "Test/Transaction Costs", avg_metrics.get("transaction_costs"), tb_step)
                _safe_add_scalar(writer, "Test/Avg Position", avg_metrics.get("avg_position"), tb_step)
                _safe_add_scalar(writer, "Test/Avg Abs Position", avg_metrics.get("avg_abs_position"), tb_step)
                _safe_add_scalar(writer, "Test/Avg Exposure Pct", avg_metrics.get("avg_exposure_pct"), tb_step)
                _safe_add_scalar(writer, "Test/Max Exposure Pct", avg_metrics.get("max_exposure_pct"), tb_step)

                _safe_add_scalar(writer, "Test/Portfolio/Total Return Pct", avg_metrics.get("total_return"), tb_step)
                _safe_add_scalar(writer, "Test/Portfolio/Sharpe Ratio", avg_metrics.get("sharpe_ratio"), tb_step)
                max_dd = avg_metrics.get("max_drawdown")
                if max_dd is not None and np.isfinite(max_dd):
                    _safe_add_scalar(writer, "Test/Portfolio/Max Drawdown Pct", float(max_dd) * 100.0, tb_step)

                # Per-action rates aggregated across test files.
                _emit_action_rates(writer, "Test/Action Rate", detailed_results, tb_step)

                # Per-trade aggregates (Tier 2b) when the evaluation populated trade KPIs.
                _emit_trade_metrics(writer, "Test/Trade", detailed_results, tb_step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"Failed to mirror test results to TensorBoard: {exc}")

        # Persist detailed test results alongside validation outputs
        model_dir = Path(config.get("run", {}).get("model_dir", "models"))
        model_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = model_dir / f"test_results_{timestamp}.json"

        try:
            with results_file.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timestamp": timestamp,
                        "average_episode_score": average_score,
                        "average_metrics": avg_metrics,
                        "detailed_results": detailed_results,
                    },
                    f,
                    indent=4,
                )
            logger.info(f"Test evaluation results saved to {results_file}")
        except Exception as exc:
            logger.error(f"Failed to save test evaluation results: {exc}")
    finally:
        trainer.close_cached_environments()


def run_training(
    config: dict,
    data_manager: DataManager,
    resume_training_flag: bool,
    reset_lr_on_resume: bool = False,
    reset_noisy_on_resume: bool = False,
    noisy_sigma_init: float | None = None,
    benchmark_frac_override: float | None = None,
):
    """Runs the training loop for the Rainbow DQN agent."""
    # Extract relevant config sections directly (will raise KeyError if missing)
    agent_config = config["agent"]
    env_config = config["environment"]
    trainer_config = config["trainer"]
    run_config = config["run"]

    # Get run parameters, using .get() only for genuinely optional/defaultable values
    model_dir = run_config.get("model_dir", "models")  # Allow default
    # resume_training = run_config.get('resume', False) # Resume status now comes from flag
    num_episodes = run_config.get("episodes", 1000)  # Allow default
    specific_file = run_config.get("specific_file", None)  # Allow default (None)

    if benchmark_frac_override is not None:
        config.setdefault("run", {})["benchmark_frac_override"] = float(benchmark_frac_override)
        logger.info(
            "--benchmark-frac-override active: pinning benchmark_allocation_frac=%.4f for the entire run.",
            float(benchmark_frac_override),
        )

    set_seeds(trainer_config["seed"])
    # Update config dict to reflect actual resume status from flag for logging
    config["run"]["resume"] = resume_training_flag
    logger.info(f"Running training with config: {config}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        error_msg = "GPU required: neither CUDA nor MPS devices detected. Aborting to prevent running training on CPU."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.info(f"Using {device} device")

    scaler = None  # GradScaler removed; bfloat16 autocast handles mixed precision

    # Agent class is fixed for this script
    AgentClass = RainbowDQNAgent
    # --- Add seed to agent_config --- # Added
    if "seed" in trainer_config:
        agent_config["seed"] = trainer_config["seed"]
    else:
        logger.warning("Seed not found in trainer config, agent may not be fully reproducible.")
        # Optionally set a default seed for the agent if missing entirely
        # agent_config['seed'] = agent_config.get('seed', 42)
    # ------------------------------- #
    # Agent config validation happens within AgentClass.__init__ if needed
    logger.info(f"Configuring for {AgentClass.__name__} Agent.")

    # --- Initialize variables for potential checkpoint loading ---
    # checkpoint = None  # Not used - agent handles loading
    start_episode = 0
    start_total_steps = 0
    initial_best_score = -np.inf
    initial_early_stopping_counter = 0
    checkpoint_data: dict[str, Any] | None = None
    # optimizer_state = None <-- Removed unused variable
    # Buffer state loading is typically not done, but agent load_model now handles optimizer/steps
    # --- End Initialization ---

    # --- Load from Checkpoint if resuming ---
    agent_loaded = False  # Flag to track if agent state was successfully loaded
    if resume_training_flag:
        # --- MODIFIED: Use find_latest_checkpoint utility ---
        trainer_checkpoint_path = find_latest_checkpoint(model_dir, "checkpoint_trainer")
        if not trainer_checkpoint_path:
            logger.warning(f"No suitable checkpoint found in {model_dir}. Starting training from scratch.")
            agent_loaded = False
        else:
            logger.info(f"Resume flag is set. Attempting to load unified checkpoint from: {trainer_checkpoint_path}")

            checkpoint_data = load_checkpoint(trainer_checkpoint_path)

            if checkpoint_data:
                if reset_lr_on_resume:
                    logger.info("Reset LR on resume requested. Removing optimizer/scheduler/scaler states from checkpoint.")
                    removed_optimizer = checkpoint_data.pop("optimizer_state_dict", None)
                    removed_scheduler = checkpoint_data.pop("scheduler_state_dict", None)
                    removed_scaler = checkpoint_data.pop("scaler_state_dict", None)
                    if removed_optimizer is not None:
                        logger.info("  - Optimizer state removed. New optimizer will use config lr=%s.", agent_config.get("lr"))
                    else:
                        logger.warning("  - No optimizer state found to remove from checkpoint.")
                    if removed_scheduler is not None:
                        logger.info("  - LR scheduler state removed. Scheduler will restart fresh.")
                    if removed_scaler is not None:
                        logger.info("  - GradScaler state removed to avoid stale statistics.")
                logger.info("Unified checkpoint loaded successfully.")
                # Extract trainer state
                start_episode = checkpoint_data.get("episode", 0)
                initial_best_score = checkpoint_data.get("best_validation_metric", -np.inf)
                initial_early_stopping_counter = checkpoint_data.get("early_stopping_counter", 0)
                if reset_lr_on_resume and initial_early_stopping_counter:
                    logger.info(
                        "Reset LR on resume: resetting early stopping counter from %d to 0.",
                        initial_early_stopping_counter,
                    )
                    initial_early_stopping_counter = 0
                # Temporary store trainer steps for comparison, agent steps are definitive
                trainer_steps_from_checkpoint = checkpoint_data.get("total_train_steps", 0)
                logger.info(
                    f"Extracted trainer state: Ep={start_episode}, BestScore={initial_best_score:.4f}, EarlyStopCounter={initial_early_stopping_counter}, TrainerSteps={trainer_steps_from_checkpoint}"
                )

                # Instantiate the agent *before* loading its state
                try:
                    # Validate loaded config if necessary (agent init might do this)
                    loaded_agent_config = checkpoint_data.get("agent_config")
                    if loaded_agent_config != agent_config:
                        logger.warning("Agent config in checkpoint differs from current config file. Using current config.")
                        # Decide if this should be an error or just a warning
                        # agent_config = loaded_agent_config # Optionally force use of loaded config

                    # Pass scaler to Agent constructor
                    agent = AgentClass(config=agent_config, device=device, scaler=scaler)
                    logger.info("Agent instantiated. Attempting to load agent state from checkpoint...")
                    agent_loaded = agent.load_state(checkpoint_data)  # Pass the whole dict

                    if agent_loaded:
                        buf_state = checkpoint_data.get("buffer_state")
                        if buf_state and hasattr(agent.buffer, "load_state_dict"):
                            try:
                                agent.buffer.load_state_dict(buf_state)
                                logger.info(f"Replay buffer restored ({len(agent.buffer)} transitions).")
                            except Exception as e:
                                logger.warning(f"Could not restore replay buffer: {e}")

                        start_total_steps = trainer_steps_from_checkpoint
                        if agent.total_steps != trainer_steps_from_checkpoint:
                            logger.warning(
                                "Agent total_steps (%s) differ from trainer checkpoint steps (%s). Synchronizing to trainer steps.",
                                agent.total_steps,
                                trainer_steps_from_checkpoint,
                            )
                            agent.total_steps = trainer_steps_from_checkpoint
                        logger.info(f"Agent state loaded successfully. Resuming from Trainer Step: {start_total_steps}")

                        if reset_noisy_on_resume:
                            try:
                                reset_count = agent.reset_noisy_sigma(std_init=noisy_sigma_init)
                                logger.info(
                                    "Reset NoisyNet sigma on resume: %d online + %d target layer(s) refilled (std_init=%s).",
                                    reset_count,
                                    reset_count,
                                    "per-layer" if noisy_sigma_init is None else f"{float(noisy_sigma_init):.4f}",
                                )
                            except Exception as exc:
                                logger.error("Failed to reset NoisyNet sigma on resume: %s", exc, exc_info=True)
                    else:
                        # Agent state loading failed, reset trainer progress
                        logger.error(
                            "Failed to load agent state from the checkpoint dictionary, even though checkpoint file was loaded. Starting training from scratch."
                        )
                        start_episode = 0
                        start_total_steps = 0
                        initial_best_score = -np.inf
                        initial_early_stopping_counter = 0
                        # Agent instance exists but is fresh
                        checkpoint_data = None
                except Exception as e:
                    logger.error(
                        f"Error occurred while instantiating agent or loading state from checkpoint: {e}. Starting training from scratch.",
                        exc_info=True,
                    )
                    start_episode = 0
                    start_total_steps = 0
                    initial_best_score = -np.inf
                    initial_early_stopping_counter = 0
                    agent_loaded = False  # Ensure agent is re-instantiated below
                    checkpoint_data = None

            else:
                # Checkpoint file not found or failed basic loading/validation
                logger.warning(f"Failed to load or validate checkpoint file at {trainer_checkpoint_path}. Starting training from scratch.")
                agent_loaded = False
        # --- END MODIFIED ---

    # --- Ensure agent is instantiated if not loaded during resume attempt ---
    if not agent_loaded:
        logger.info("Instantiating fresh agent.")
        # Pass scaler to Agent constructor
        agent = AgentClass(config=agent_config, device=device, scaler=scaler)

    assert isinstance(agent, RainbowDQNAgent), "Agent not instantiated correctly"
    logger.info(f"Agent instantiated with {sum(p.numel() for p in agent.network.parameters()):,} parameters.")

    # --- Initialize TensorBoard Writer ---
    log_dir_base = (Path(model_dir) / "runs").resolve()
    log_dir_base.mkdir(parents=True, exist_ok=True)

    resume_log_dir_str = checkpoint_data.get("tensorboard_log_dir") if checkpoint_data else None
    log_dir_path: Path

    # Aggressive disk flush to survive hard freezes / power loss / driver hangs.
    # Default flush_secs=120 can lose up to ~2 min of scalars on an unclean
    # shutdown. flush_secs=1 caps that background-flush loss window at ~1 s.
    # We leave max_queue at the default so the vectorized training loop (which
    # emits many scalars per step across parallel envs) doesn't stall on the
    # writer queue; the explicit _flush_writer() calls at episode and
    # checkpoint boundaries provide synchronous guarantees at the points that
    # actually matter for resume correctness.
    tb_flush_kwargs: dict[str, Any] = {"flush_secs": 1}

    if resume_training_flag and resume_log_dir_str:
        resumed = Path(resume_log_dir_str)
        log_dir_path = resumed if resumed.is_absolute() else (Path.cwd() / resumed).resolve()
        log_dir_path.mkdir(parents=True, exist_ok=True)
        writer_kwargs: dict[str, Any] = {"log_dir": str(log_dir_path), **tb_flush_kwargs}
        if start_total_steps > 0:
            writer_kwargs["purge_step"] = start_total_steps
        writer = SummaryWriter(**writer_kwargs)
        logger.info(f"Resuming TensorBoard logging in: {log_dir_path}")
    else:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        log_dir_path = log_dir_base / current_time
        log_dir_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir_path), **tb_flush_kwargs)
        logger.info(f"TensorBoard logs: {log_dir_path}")

    # Store log dir for downstream components if needed
    config.setdefault("run", {})["tensorboard_log_dir"] = str(log_dir_path)

    # --- Instantiate Trainer ---
    trainer = RainbowTrainerModule(
        agent=agent,
        device=device,
        data_manager=data_manager,
        config=config,  # Pass the full config to the trainer
        scaler=scaler,  # Pass scaler to Trainer constructor
        writer=writer,  # Pass the TensorBoard writer
        # Remove handler passing, as root logger handles it now
        # train_log_handler=train_log_handler,
        # validation_log_handler=validation_log_handler
    )
    assert isinstance(trainer, RainbowTrainerModule), "Failed to instantiate RainbowTrainerModule"
    logger.info("RAINBOW Trainer initialized.")

    # --- Initial Env Setup ---
    try:
        logger.info(f"DataManager type: {type(data_manager)}")
        logger.info(f"DataManager has organize_data: {hasattr(data_manager, 'organize_data')}")
        logger.info(f"DataManager _data_organized: {getattr(data_manager, '_data_organized', 'N/A')}")
        initial_file = get_random_data_file(data_manager)
        assert isinstance(initial_file, Path), "Failed to get a valid initial data file path"
        logger.info(f"Using initial file for env setup check: {initial_file.name}")
        # Use env_config for environment parameters
        # Add data_path to the env_config dictionary
        env_config["data_path"] = str(initial_file)
        # Create config object first, now including data_path
        env_config_obj = TradingEnvConfig(**env_config)
        initial_env = TradingEnv(
            # data_path=str(initial_file), # Remove data_path, now in config
            config=env_config_obj  # Pass the config object
        )
        assert isinstance(initial_env, TradingEnv), "Failed to create initial TradingEnv instance"
    except Exception as e:
        logger.error(f"Failed to create initial environment: {e}")
        raise  # Stop if initial env setup fails

    logger.info("=============================================")
    logger.info(f"STARTING RAINBOW TRAINING{' (Resuming via flag)' if resume_training_flag else ''}")
    logger.info("=============================================")

    # --- Run Training ---
    logger.debug(f"Agent config: {agent_config}")
    logger.debug(f"Environment config: {env_config}")
    try:
        trainer.train(
            # env=initial_env, # Removed argument
            num_episodes=num_episodes,
            start_episode=start_episode,
            start_total_steps=start_total_steps,
            initial_best_score=initial_best_score,
            initial_early_stopping_counter=initial_early_stopping_counter,
            specific_file=specific_file,
            # Other params like validation_freq, gamma, batch_size etc. are now taken from config inside trainer
        )
    finally:
        # Force a flush to disk even if training crashed (FloatingPointError,
        # KeyboardInterrupt, etc.). This is our last-chance sync before the
        # process exits; a hard OS freeze still can't be helped here, but every
        # normal exit path will have TB events and buffers on disk.
        try:
            trainer.writer.flush() if getattr(trainer, "writer", None) is not None else None
        except Exception:
            logger.debug("Failed to flush TensorBoard writer in run_training finally", exc_info=True)

    # Close the initial environment (might be redundant if trainer closes final env)
    try:
        initial_env.close()
    except Exception:
        logger.debug("Error closing initial environment (may already be closed)", exc_info=True)

    # NOTE: the TensorBoard writer is intentionally NOT closed here. It is owned by
    # the trainer (``trainer.writer``) and remains open so post-training evaluation
    # (e.g. ``evaluate_on_test_data``) can mirror Test/* scalars under the same
    # event file. ``main()`` is responsible for closing the writer once evaluation
    # has completed.

    return agent, trainer


def main():  # Remove default config_path
    """Main function to load config and run Rainbow DQN training/evaluation."""

    # --- Argument Parsing --- # Added
    parser = argparse.ArgumentParser(description="Run Rainbow DQN Training or Evaluation")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/training_config.yaml",
        help="Path to the configuration YAML file.",
    )
    # ADD definition for --resume flag
    parser.add_argument(
        "--resume",
        action="store_true",  # Makes it a flag, True if present, False otherwise
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument(
        "--reset-lr-on-resume",
        action="store_true",
        help="When resuming, discard optimizer/scheduler states so learning rate resets to config value.",
    )
    parser.add_argument(
        "--reset-noisy-on-resume",
        action="store_true",
        help=(
            "When resuming, refill NoisyLinear sigma parameters to re-energise exploration. "
            "Mu (the deterministic part) is left untouched so the policy keeps what it learned."
        ),
    )
    parser.add_argument(
        "--noisy-sigma-init",
        type=float,
        default=None,
        help=("Override the std_init scalar used by --reset-noisy-on-resume. Defaults to each layer's constructor value (typically 0.5)."),
    )
    parser.add_argument(
        "--benchmark-frac-override",
        type=float,
        default=None,
        help=("Pin benchmark_allocation_frac to this constant for the entire run, ignoring any scheduled anneal in trainer config."),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (e.g. DEBUG, INFO, WARNING). Overrides MOMENTUM_LOG_LEVEL* environment variables.",
    )
    # Tier 2d: by default validation/test runs in deterministic ``agent.greedy()``
    # mode. ``--eval-stochastic`` opts back into the pre-Tier-2d behaviour where
    # epsilon-greedy and NoisyNet noise are still active during evaluation; this
    # is useful to measure the gap between the policy actually being trained and
    # its greedy projection (Train/EvalGap/* in TensorBoard).
    parser.add_argument(
        "--eval-stochastic",
        action="store_true",
        help="Run validation/test in stochastic (training) mode instead of agent.greedy().",
    )
    args = parser.parse_args()
    config_path = args.config_path
    # Use the command-line flag directly for resuming
    resume_training_flag = args.resume
    reset_lr_on_resume = args.reset_lr_on_resume
    reset_noisy_on_resume = args.reset_noisy_on_resume
    noisy_sigma_init = args.noisy_sigma_init
    benchmark_frac_override = args.benchmark_frac_override
    eval_stochastic_flag = args.eval_stochastic
    configure_logging(args.log_level)

    logger.info("Starting Rainbow DQN training script...")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    # ----------------------- #

    # --- Load Configuration --- # Use parsed config_path
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}. Exiting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}. Exiting.")
        return

    # Reconfigure logging if the config supplies overrides (CLI takes precedence)
    logging_config = config.get("logging") if isinstance(config, dict) else None
    configure_logging(args.log_level, logging_config)
    logger.info(f"Configuration loaded successfully from {config_path}")

    # --- Extract sections and parameters ---
    # Expect these sections to exist
    agent_config = config["agent"]
    # trainer_config = config["trainer"]  # Not used in main section
    # env_config = config["environment"]  # Not used in main section
    run_config = config["run"]  # Expect 'run' section

    # Get run parameters, allowing defaults only where sensible
    mode = run_config.get("mode", "train")  # Default to train is reasonable
    model_dir = run_config.get("model_dir", "models")  # Default model dir is reasonable
    # REMOVE reliance on config for resume, use flag instead
    # resume_training = run_config.get('resume', False)
    eval_model_prefix = run_config.get("eval_model_prefix", f"{model_dir}/rainbow_transformer_best")  # Default prefix is reasonable
    skip_evaluation = run_config.get("skip_evaluation", False)  # Default to False is reasonable
    data_base_dir = run_config.get("data_base_dir", "data")  # Default base dir is reasonable

    # Tier 2d: surface --eval-stochastic into the config so the trainer reads it
    # off ``run.eval_stochastic`` like every other run-level toggle.
    config.setdefault("run", {})["eval_stochastic"] = eval_stochastic_flag
    if eval_stochastic_flag:
        logger.info("--eval-stochastic enabled: validation/test will run with epsilon-greedy + NoisyNet noise active.")

    # --- Initialize DataManager ---
    # Pass base_dir from config. Processed dir name defaults to 'processed' unless specified.
    data_manager = DataManager(base_dir=data_base_dir)
    data_manager.organize_data()  # Load file lists from directories
    assert isinstance(data_manager, DataManager), "Failed to initialize DataManager"

    os.makedirs(model_dir, exist_ok=True)

    if mode == "train":
        # Pass the resume_training_flag to run_training
        trained_agent, trained_trainer = run_training(
            config,
            data_manager,
            resume_training_flag,
            reset_lr_on_resume=reset_lr_on_resume,
            reset_noisy_on_resume=reset_noisy_on_resume,
            noisy_sigma_init=noisy_sigma_init,
            benchmark_frac_override=benchmark_frac_override,
        )
        assert isinstance(trained_agent, RainbowDQNAgent), "run_training did not return a valid agent"
        assert isinstance(trained_trainer, RainbowTrainerModule), "run_training did not return a valid trainer"

        if not skip_evaluation:  # Check the flag before running evaluation
            logger.info("--- Starting Evaluation on Test Data after Training (Rainbow) ---")
            # Pass necessary config parts to evaluation function. The trainer's writer
            # is reused so Test/* TB scalars line up with Train/* / Validation/* views.
            evaluate_on_test_data(
                agent=trained_agent,
                trainer=trained_trainer,
                config=config,
            )
        else:
            logger.info("--- Skipping Evaluation on Test Data as per configuration (skip_evaluation=True) ---")

        train_writer = getattr(trained_trainer, "writer", None)
        if train_writer is not None:
            try:
                train_writer.close()
                logger.info("TensorBoard writer closed.")
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to close TensorBoard writer", exc_info=True)

    elif mode == "eval":
        logger.info("--- Starting Evaluation Mode (Rainbow) --- ")
        assert isinstance(eval_model_prefix, str) and len(eval_model_prefix) > 0, "Invalid eval_model_prefix in config"
        logger.info(f"Loading model from prefix: {eval_model_prefix}")

        # Determine device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"Using device: {device}")

        # Instantiate Rainbow agent using loaded config for evaluation
        # Ensure agent config has the seed for reproducibility during eval if needed
        if "seed" not in agent_config and "trainer" in config and "seed" in config["trainer"]:
            agent_config["seed"] = config["trainer"]["seed"]
            logger.info(f"Added seed {agent_config['seed']} to agent config for evaluation.")

        # Pass scaler=None during evaluation as AMP is typically for training
        eval_agent = RainbowDQNAgent(config=agent_config, device=device, scaler=None)
        assert isinstance(eval_agent, RainbowDQNAgent), "Failed to instantiate agent for evaluation"

        # Load model weights
        # Note: load_model now doesn't need architecture args, they come from agent's config
        eval_agent.load_model(
            eval_model_prefix,
        )
        assert eval_agent.network is not None, f"Model loading failed for prefix {eval_model_prefix}, network is None"
        logger.info("Model loaded successfully for evaluation.")
        eval_agent.set_training_mode(False)

        # Pass full config to trainer for evaluation setup (if needed)
        # Pass scaler=None to trainer during evaluation
        eval_trainer = RainbowTrainerModule(agent=eval_agent, device=device, data_manager=data_manager, config=config, scaler=None)

        # Run evaluation - internal asserts will check inputs
        evaluate_on_test_data(
            agent=eval_agent,
            trainer=eval_trainer,  # Trainer might hold metrics or env creation logic
            config=config,  # Pass full config for evaluation needs
        )

    else:
        logger.error(f"Invalid mode specified in config run section: {mode}. Use 'train' or 'eval'.")  # Or raise ValueError

    logger.info(f"Script finished ({mode} mode, agent: rainbow).")


if __name__ == "__main__":
    main()  # Call main without arguments
