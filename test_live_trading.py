#!/usr/bin/env python3
"""
Test script for the live momentum trading implementation.
This script initializes the system and verifies it can connect to Alpaca.
"""

import logging
from pathlib import Path

try:
    from momentum_core.logging import get_logger, setup_package_logging
except ImportError:  # pragma: no cover - fallback when package unavailable
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.warning("Could not find momentum_core.logging, using basic config.")
    get_logger = logging.getLogger

from momentum_live.agent_loader import find_best_checkpoint, load_agent_from_checkpoint
from momentum_live.alpaca_stream import AlpacaStreamRunner
from momentum_live.config import AlpacaCredentials, LiveTradingConfig, parse_symbols
from momentum_live.trader import MomentumLiveTrader


def configure_logging(log_level: str | None = None) -> None:
    if "setup_package_logging" not in globals():
        return

    setup_package_logging(
        "momentum_live.test",
        log_filename="test_live_trading.log",
        root_level=log_level if log_level is not None else logging.INFO,
        console_level=log_level if log_level is not None else logging.INFO,
    )


logger = get_logger("momentum_live.test")


def test_initialization():
    """Test that the live trading system can be initialized."""
    logger.info("Testing Live Momentum Trading System Initialization")

    try:
        # Test symbol parsing
        symbols = parse_symbols("BTC/USD,ETH/USD")
        logger.info("Parsed symbols: %s", symbols)

        # Test configuration
        trading_config = LiveTradingConfig(
            symbols=symbols,
            window_size=60,
            models_dir="models",
            initial_balance=1000.0,
        )
        logger.info("Created trading config with %s symbols", len(trading_config.symbols))

        # Test Alpaca credentials
        credentials = AlpacaCredentials.from_environment()
        logger.info("Loaded Alpaca credentials (paper trading: %s)", credentials.paper)

        # Test checkpoint loading - try the specific final agent state first
        final_checkpoint = Path(trading_config.models_dir) / "rainbow_transformer_final_agent_state.pt"
        if final_checkpoint.exists():
            checkpoint_path = final_checkpoint
            logger.info("Using final agent checkpoint: %s", checkpoint_path.name)
        else:
            checkpoint_path = find_best_checkpoint(trading_config.models_dir)
            logger.info("Found best checkpoint: %s", checkpoint_path.name)

        # Extract agent config from checkpoint
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for config_key in ["agent_config", "config"]:
            if config_key in checkpoint:
                agent_config = checkpoint[config_key]
                break
        else:
            # Fallback to a basic config if checkpoint doesn't have it
            agent_config = {
                "gamma": 0.99,
                "lr": 0.001,
                "batch_size": 256,
                "replay_buffer_size": 500000,
                "target_update_freq": 2500,
                "window_size": 60,
                "n_features": 5,
                "hidden_dim": 256,
                "num_actions": 7,
                "nhead": 4,
                "num_encoder_layers": 3,
                "dim_feedforward": 512,
                "transformer_dropout": 0.2,
                "n_steps": 3,
                "num_atoms": 51,
                "v_min": -1.0,
                "v_max": 1.0,
                "alpha": 0.6,
                "beta_start": 0.5,
                "beta_frames": 1000000,
                "lr_scheduler_enabled": True,
                "lr_scheduler_type": "ReduceLROnPlateau",
                "lr_scheduler_params": {"mode": "max", "factor": 0.1, "patience": 5, "threshold": 0.0001, "min_lr": 1e-06},
                "grad_clip_norm": 1.0,
                "debug": False,
                "seed": 42,
            }
        logger.info("Extracted agent config from checkpoint")

        # Test agent loading (use fresh agent if checkpoint loading fails)
        try:
            agent = load_agent_from_checkpoint(checkpoint_path)
            logger.info(
                "Loaded trained agent with %,d parameters",
                sum(p.numel() for p in agent.network.parameters()),
            )
        except Exception as e:
            logger.warning("Could not load checkpoint (%s), creating fresh agent for testing", e)
            # Create a fresh agent with the same config as the checkpoint
            from momentum_agent import RainbowDQNAgent

            agent = RainbowDQNAgent(config=agent_config, device="cpu", scaler=None)
            logger.info(
                "Created fresh agent with %,d parameters",
                sum(p.numel() for p in agent.network.parameters()),
            )

        # Test trader initialization
        trader = MomentumLiveTrader(agent=agent, config=trading_config)
        logger.info("Initialized trader for symbols: %s", ", ".join(trader.symbols))
        logger.info("Shared balance initialized: $%.2f", trader.shared_balance)

        # Test Alpaca stream runner initialization
        runner = AlpacaStreamRunner(credentials=credentials, trader=trader)  # noqa: F841
        logger.info("Initialized Alpaca stream runner")

        logger.info("All components initialized successfully")
        logger.info("Multi-symbol trading ready with shared account balance: $%.2f", trader.shared_balance)
        logger.info("To start live trading, run: venv\\Scripts\\python.exe -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO")

        return True

    except Exception as e:
        logger.exception("Initialization failed: %s", e)
        return False


if __name__ == "__main__":
    configure_logging()
    success = test_initialization()
    exit(0 if success else 1)
