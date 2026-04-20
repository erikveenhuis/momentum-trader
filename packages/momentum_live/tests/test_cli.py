"""Unit tests for ``momentum_live.cli`` (argparse + adopt-spec helpers)."""

from __future__ import annotations

import pytest
from momentum_live.cli import _parse_adopt_spec, build_arg_parser


def _minimal_argv(*extra: str) -> list[str]:
    return [
        "--symbols",
        "BTC/USD",
        "--window-size",
        "60",
        "--initial-balance",
        "10000",
        "--transaction-fee",
        "0.001",
        "--reward-scale",
        "1.0",
        "--invalid-penalty",
        "-0.1",
        "--drawdown-penalty-lambda",
        "0.5",
        "--slippage-bps",
        "5",
        "--opportunity-cost-lambda",
        "0.1",
        "--benchmark-allocation-frac",
        "0.5",
        "--min-rebalance-pct",
        "0.02",
        "--min-trade-value",
        "1.0",
        *extra,
    ]


def test_parse_adopt_spec_valid() -> None:
    pair, account_id = _parse_adopt_spec("BTC/USD:a7880383-9924-446b-8be7-3d8b3bcdf68f")
    assert pair == "BTC/USD"
    assert account_id == "a7880383-9924-446b-8be7-3d8b3bcdf68f"


def test_parse_adopt_spec_strips_whitespace() -> None:
    pair, account_id = _parse_adopt_spec("  ETH/USD : 11111111-2222-3333-4444-555555555555 ")
    assert pair == "ETH/USD"
    assert account_id == "11111111-2222-3333-4444-555555555555"


def test_parse_adopt_spec_missing_separator() -> None:
    with pytest.raises(ValueError, match="PAIR:ACCOUNT_ID"):
        _parse_adopt_spec("BTC/USD-no-colon")


def test_parse_adopt_spec_empty_half() -> None:
    with pytest.raises(ValueError, match="empty pair or account id"):
        _parse_adopt_spec(":abc")


def test_arg_parser_accepts_multiple_adopt_specs() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        _minimal_argv(
            "--adopt",
            "BTC/USD:a7880383-9924-446b-8be7-3d8b3bcdf68f",
            "--adopt",
            "ETH/USD:11111111-2222-3333-4444-555555555555",
        )
    )
    assert args.adopt == [
        "BTC/USD:a7880383-9924-446b-8be7-3d8b3bcdf68f",
        "ETH/USD:11111111-2222-3333-4444-555555555555",
    ]


def test_arg_parser_adopt_defaults_to_empty_list() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(_minimal_argv())
    assert args.adopt == []


def test_arg_parser_reset_mode_none() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(_minimal_argv("--reset-mode", "none"))
    assert args.reset_mode == "none"
