"""Command line interface for Broker-API live momentum trading."""

from __future__ import annotations

import argparse
import logging
import sys

from momentum_core.logging import get_logger, setup_package_logging

from .account_registry import BrokerAccountRegistry
from .agent_loader import find_best_checkpoint, load_agent_from_checkpoint
from .broker import BrokerAccountManager, BrokerCredentials
from .config import AlpacaCredentials, LiveTradingConfig, parse_symbols
from .multi_pair_runner import MultiPairRunner

LOGGER = get_logger("momentum_live.cli")


def _parse_adopt_spec(spec: str) -> tuple[str, str]:
    """Parse a ``PAIR:ACCOUNT_ID`` CLI argument into its components.

    The pair itself contains a ``/`` (e.g. ``BTC/USD``) and the account id is
    a UUID, so we split on the *last* ``:`` to be unambiguous.
    """
    if ":" not in spec:
        raise ValueError(f"--adopt value {spec!r} must be in the form PAIR:ACCOUNT_ID, e.g. BTC/USD:a7880383-9924-446b-8be7-3d8b3bcdf68f")
    pair, account_id = spec.rsplit(":", 1)
    pair = pair.strip()
    account_id = account_id.strip()
    if not pair or not account_id:
        raise ValueError(f"--adopt value {spec!r} has an empty pair or account id")
    return pair, account_id


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    setup_package_logging(
        "momentum_live",
        root_level=numeric_level,
        console_level=numeric_level,
        log_filename="live_trading.log",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the momentum agent against live Alpaca crypto data, with one Broker "
            "sub-account per symbol. Source scripts/env-paper.sh first."
        ),
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated crypto pairs (e.g. BTC/USD,ETH/USD)")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint file to load")
    parser.add_argument(
        "--checkpoint-pattern",
        default="checkpoint_trainer_best_*.pt",
        help="Glob pattern for checkpoint files",
    )
    parser.add_argument("--window-size", type=int, required=True, help="Rolling window size (must match training)")
    parser.add_argument(
        "--initial-balance",
        type=float,
        required=True,
        help="Per-sub-account starting balance in USD (also the soft-reset target)",
    )
    parser.add_argument("--transaction-fee", type=float, required=True, help="Transaction fee fraction (e.g. 0.001)")
    parser.add_argument("--reward-scale", type=float, required=True, help="Reward scale (must match training)")
    parser.add_argument("--invalid-penalty", type=float, required=True, help="Invalid action penalty")
    parser.add_argument("--drawdown-penalty-lambda", type=float, required=True, help="Drawdown penalty weight")
    parser.add_argument("--slippage-bps", type=float, required=True, help="Slippage in basis points")
    parser.add_argument("--opportunity-cost-lambda", type=float, required=True, help="Opportunity cost weight")
    parser.add_argument(
        "--benchmark-allocation-frac",
        type=float,
        required=True,
        help="Benchmark allocation for relative reward",
    )
    parser.add_argument("--min-rebalance-pct", type=float, required=True, help="Min allocation delta to trigger trade")
    parser.add_argument("--min-trade-value", type=float, required=True, help="Min trade notional in USD")
    parser.add_argument("--location", default=None, help="Alpaca crypto data feed location (default: ALPACA_CRYPTO_FEED or 'us')")
    parser.add_argument(
        "--reset-mode",
        choices=("none", "soft", "hard"),
        default="soft",
        help="Pre-run reset behaviour: 'none' leaves sub-accounts as is, 'soft' (default) "
        "cancels orders + closes positions + JNLCs balance back to --initial-balance, "
        "'hard' is reserved for a future revision",
    )
    parser.add_argument(
        "--subaccount-registry",
        default="models/broker_subaccounts.json",
        help="Path to the JSON file mapping pair -> Broker sub-account id",
    )
    parser.add_argument(
        "--adopt",
        action="append",
        default=[],
        metavar="PAIR:ACCOUNT_ID",
        help=(
            "Upsert a pre-existing Broker sub-account id into the registry "
            "before running (repeatable). Example: "
            "--adopt BTC/USD:a7880383-9924-446b-8be7-3d8b3bcdf68f. Useful for "
            "reusing accounts created via the Brokerdash tutorial when "
            "programmatic sub-account creation or firm-level funding is "
            "unavailable."
        ),
    )
    parser.add_argument(
        "--tb-log-dir",
        default=None,
        help="If set, mirror Live/Trade/*, Live/Action Rate/*, Live/Q/* scalars to TensorBoard.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
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
            initial_balance=args.initial_balance,
            transaction_fee=args.transaction_fee,
            reward_scale=args.reward_scale,
            invalid_action_penalty=args.invalid_penalty,
            drawdown_penalty_lambda=args.drawdown_penalty_lambda,
            slippage_bps=args.slippage_bps,
            opportunity_cost_lambda=args.opportunity_cost_lambda,
            benchmark_allocation_frac=args.benchmark_allocation_frac,
            min_rebalance_pct=args.min_rebalance_pct,
            min_trade_value=args.min_trade_value,
            models_dir=args.models_dir,
            checkpoint_pattern=args.checkpoint_pattern,
            tb_log_dir=args.tb_log_dir,
            reset_mode=args.reset_mode,
            subaccount_registry_path=args.subaccount_registry,
        )

        data_credentials = AlpacaCredentials.from_environment()
        if args.location:
            data_credentials = AlpacaCredentials(
                api_key=data_credentials.api_key,
                secret_key=data_credentials.secret_key,
                location=args.location,
            )

        broker_credentials = BrokerCredentials.from_environment()
        if trading_config.reset_mode != "none" and not broker_credentials.has_firm_account:
            raise RuntimeError(
                "ALPACA_BROKER_ACCOUNT_ID is not set but --reset-mode is "
                f"{trading_config.reset_mode!r}. Either set the firm account "
                "id in .env or re-run with --reset-mode none (agent-only "
                "validation, no JNLC)."
            )

        registry = BrokerAccountRegistry(trading_config.subaccount_registry_path)
        broker_manager = BrokerAccountManager(broker_credentials, registry)

        for spec in args.adopt or []:
            pair, account_id = _parse_adopt_spec(spec)
            broker_manager.adopt_subaccount(pair, account_id)

        checkpoint_path = args.checkpoint or find_best_checkpoint(
            trading_config.models_dir,
            trading_config.checkpoint_pattern,
        )
        agent = load_agent_from_checkpoint(checkpoint_path)
        n_params = sum(p.numel() for p in agent.network.parameters())
        LOGGER.info(
            "Loaded trained agent (%s parameters) from %s",
            f"{n_params:,}",
            checkpoint_path,
        )

        runner = MultiPairRunner(
            agent=agent,
            broker=broker_manager,
            data_credentials=data_credentials,
            config=trading_config,
        )

        if trading_config.reset_mode != "none":
            LOGGER.info("Pre-run reset (mode=%s)", trading_config.reset_mode)
            runner.reset_all(trading_config.reset_mode)

        runner.run_forever()
    except KeyboardInterrupt:
        LOGGER.info("Received interrupt signal, shutting down")
    except RuntimeError as exc:
        LOGGER.error("Live trading runtime error: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - CLI level guard
        LOGGER.exception("Fatal error in live runner: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
