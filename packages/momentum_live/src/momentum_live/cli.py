"""Command line interface for running live momentum trading."""

from __future__ import annotations

import argparse
import logging
import sys

from momentum_core.logging import get_logger, setup_package_logging

from .agent_loader import find_best_checkpoint, load_agent_from_checkpoint
from .alpaca_stream import AlpacaStreamRunner
from .config import AlpacaCredentials, LiveTradingConfig, parse_symbols
from .trader import MomentumLiveTrader

logger = get_logger("momentum_live.cli")


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    setup_package_logging(
        "momentum_live",
        root_level=numeric_level,
        console_level=numeric_level,
        log_filename="live_trading.log",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the momentum agent against live Alpaca crypto data")
    parser.add_argument("--symbols", required=True, help="Comma or space separated list of crypto pairs (e.g. 'BTC/USD,ETH/USD')")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint file to load. Defaults to best score in models-dir")
    parser.add_argument("--window-size", type=int, default=60, help="Rolling window size used during training")
    parser.add_argument("--initial-balance", type=float, default=1000.0, help="Initial virtual balance per symbol")
    parser.add_argument("--transaction-fee", type=float, default=0.001, help="Transaction fee fraction (e.g. 0.001 = 0.1%)")
    parser.add_argument("--reward-scale", type=float, default=50.0, help="Reward scale used during training")
    parser.add_argument("--invalid-penalty", type=float, default=-0.05, help="Penalty applied when the agent proposes an invalid trade")
    parser.add_argument("--location", default=None, help="Alpaca crypto data feed location (us or us-1)")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (DEBUG, INFO, WARN, ...)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    try:
        symbols = parse_symbols(args.symbols)
        trading_config = LiveTradingConfig(
            symbols=symbols,
            window_size=args.window_size,
            models_dir=args.models_dir,
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            reward_scale=args.reward_scale,
            invalid_action_penalty=args.invalid_penalty,
        )

        credentials = AlpacaCredentials.from_environment()
        if args.location:
            credentials = AlpacaCredentials(
                api_key=credentials.api_key,
                secret_key=credentials.secret_key,
                location=args.location,
            )

        checkpoint_path = args.checkpoint
        if not checkpoint_path:
            checkpoint_path = find_best_checkpoint(trading_config.models_dir, trading_config.checkpoint_pattern)

        # Try to load agent from checkpoint, fallback to fresh agent if loading fails
        try:
            agent = load_agent_from_checkpoint(checkpoint_path)
            logger.info(f"Loaded trained agent with {sum(p.numel() for p in agent.network.parameters()):,} parameters")
        except Exception as e:
            logger.warning(f"Could not load checkpoint ({e}), creating fresh agent for live trading")
            # Create a fresh agent with the same config as the checkpoint
            import torch
            from momentum_agent import RainbowDQNAgent

            # Extract agent config from checkpoint for fresh agent
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

            agent = RainbowDQNAgent(config=agent_config, device="cuda" if torch.cuda.is_available() else "cpu", scaler=None)
            logger.info(f"Created fresh agent with {sum(p.numel() for p in agent.network.parameters()):,} parameters")

        trader = MomentumLiveTrader(agent=agent, config=trading_config)
        runner = AlpacaStreamRunner(credentials=credentials, trader=trader)
        runner.run_forever()
    except Exception as exc:  # pragma: no cover - CLI level protection
        logging.getLogger("momentum_live.cli").exception("Fatal error in live runner: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
