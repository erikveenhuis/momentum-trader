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

import numpy as np
import torch
import yaml  # Added for config loading
from momentum_agent import RainbowDQNAgent
from momentum_core.logging import get_logger, setup_package_logging
from momentum_env import TradingEnv, TradingEnvConfig

# --- Add AMP imports ---
from torch.amp import GradScaler

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


def configure_logging(log_level: str | None = None) -> None:
    """Configure logging for the momentum_train package."""

    setup_package_logging(
        "momentum_train",
        log_filename="training.log",
        root_level=log_level if log_level is not None else logging.INFO,
        console_level=log_level if log_level is not None else logging.INFO,
        level_overrides={
            "momentum_train.Main": logging.INFO,
            "Trainer": logging.INFO,
            "Agent": logging.INFO,
            "DataManager": logging.INFO,
            "TransformerModel": logging.INFO,
            "Buffer": logging.INFO,
            "Metrics": logging.INFO,
            "Evaluation": logging.INFO,
        },
    )


def evaluate_on_test_data(agent: RainbowDQNAgent, trainer: RainbowTrainerModule, config: dict) -> None:
    """Run evaluation across the test split and log aggregate results."""
    if not hasattr(trainer, "data_manager"):
        logger.error("Trainer does not expose a data_manager; cannot evaluate on test data.")
        return

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
        for test_file in test_files:
            try:
                result = trainer._validate_single_file(test_file)
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
        logger.info(f"Average Max Drawdown: {avg_metrics['max_drawdown']*100:.2f}%")
        logger.info(f"Average Transaction Costs: ${avg_metrics['transaction_costs']:.2f}")
        logger.info("============================================")

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


def run_training(config: dict, data_manager: DataManager, resume_training_flag: bool):
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
        error_msg = "GPU required: neither CUDA nor MPS devices detected. " "Aborting to prevent running training on CPU."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.info(f"Using {device} device")

    # --- Initialize GradScaler for AMP if using CUDA ---
    scaler = None
    if device.type == "cuda":
        scaler = GradScaler("cuda")
        logger.info("Initialized GradScaler for Automatic Mixed Precision (AMP).")
    # --------------------------------------------------

    # --- Initialize TensorBoard Writer ---
    log_dir_base = Path(model_dir) / "runs"
    # Create a unique directory name using a timestamp
    current_time = time.strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir_base / current_time
    writer = SummaryWriter(log_dir=str(log_dir))
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    # ----------------------------------

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

            loaded_checkpoint = load_checkpoint(trainer_checkpoint_path)

            if loaded_checkpoint:
                logger.info("Unified checkpoint loaded successfully.")
                # Extract trainer state
                start_episode = loaded_checkpoint.get("episode", 0)
                initial_best_score = loaded_checkpoint.get("best_validation_metric", -np.inf)
                initial_early_stopping_counter = loaded_checkpoint.get("early_stopping_counter", 0)
                # Temporary store trainer steps for comparison, agent steps are definitive
                trainer_steps_from_checkpoint = loaded_checkpoint.get("total_train_steps", 0)
                logger.info(
                    f"Extracted trainer state: Ep={start_episode}, BestScore={initial_best_score:.4f}, EarlyStopCounter={initial_early_stopping_counter}, TrainerSteps={trainer_steps_from_checkpoint}"
                )

                # Instantiate the agent *before* loading its state
                try:
                    # Validate loaded config if necessary (agent init might do this)
                    loaded_agent_config = loaded_checkpoint.get("agent_config")
                    if loaded_agent_config != agent_config:
                        logger.warning("Agent config in checkpoint differs from current config file. Using current config.")
                        # Decide if this should be an error or just a warning
                        # agent_config = loaded_agent_config # Optionally force use of loaded config

                    # Pass scaler to Agent constructor
                    agent = AgentClass(config=agent_config, device=device, scaler=scaler)
                    logger.info("Agent instantiated. Attempting to load agent state from checkpoint...")
                    agent_loaded = agent.load_state(loaded_checkpoint)  # Pass the whole dict

                    if agent_loaded:
                        # Prefer the trainer's recorded step count for resume consistency
                        start_total_steps = trainer_steps_from_checkpoint
                        if agent.total_steps != trainer_steps_from_checkpoint:
                            logger.warning(
                                "Agent total_steps (%s) differ from trainer checkpoint steps (%s). Synchronizing to trainer steps.",
                                agent.total_steps,
                                trainer_steps_from_checkpoint,
                            )
                            agent.total_steps = trainer_steps_from_checkpoint
                        logger.info(f"Agent state loaded successfully. Resuming from Trainer Step: {start_total_steps}")
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

    # Close the initial environment (might be redundant if trainer closes final env)
    try:
        initial_env.close()
    except Exception:
        pass  # Ignore errors closing env that might already be closed

    # --- Close TensorBoard Writer ---
    writer.close()
    logger.info("TensorBoard writer closed.")
    # -----------------------------

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
        "--log-level",
        default=None,
        help="Logging level (e.g. DEBUG, INFO, WARNING). Overrides MOMENTUM_LOG_LEVEL* environment variables.",
    )
    args = parser.parse_args()
    config_path = args.config_path
    # Use the command-line flag directly for resuming
    resume_training_flag = args.resume
    configure_logging(args.log_level)

    logger.info("Starting Rainbow DQN training script...")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    # ----------------------- #

    # --- Load Configuration --- # Use parsed config_path
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Exiting.")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}. Exiting.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred loading config: {e}. Exiting.")
        return

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

    # --- Initialize DataManager ---
    # Pass base_dir from config. Processed dir name defaults to 'processed' unless specified.
    data_manager = DataManager(base_dir=data_base_dir)
    data_manager.organize_data()  # Load file lists from directories
    assert isinstance(data_manager, DataManager), "Failed to initialize DataManager"

    os.makedirs(model_dir, exist_ok=True)

    if mode == "train":
        # Pass the resume_training_flag to run_training
        trained_agent, trained_trainer = run_training(config, data_manager, resume_training_flag)
        assert isinstance(trained_agent, RainbowDQNAgent), "run_training did not return a valid agent"
        assert isinstance(trained_trainer, RainbowTrainerModule), "run_training did not return a valid trainer"

        if not skip_evaluation:  # Check the flag before running evaluation
            logger.info("--- Starting Evaluation on Test Data after Training (Rainbow) ---")
            # Pass necessary config parts to evaluation function
            evaluate_on_test_data(
                agent=trained_agent,
                trainer=trained_trainer,  # Trainer might hold metrics or env creation logic
                config=config,  # Pass full config for evaluation needs
            )
        else:
            logger.info("--- Skipping Evaluation on Test Data as per configuration (skip_evaluation=True) ---")

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
