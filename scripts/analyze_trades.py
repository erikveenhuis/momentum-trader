#!/usr/bin/env python3
"""Tier 5c: offline trade-level analyzer.

Consumes the per-step + trade JSONL artifacts produced by ``eval_greedy.py``
(or any compatible producer — the trainer also writes ``*.trades.jsonl``
sidecars during validation/test), then emits:

* Per-trade metrics split by greedy / epsilon-forced provenance.
* Per-trade metrics rolled up by user-provided rolling windows (e.g.
  most-recent N=50 trades) so we can see whether the policy is improving.
* Equity-curve summary (per-step portfolio value -> mean/min/max/final).
* Trade-PnL CDF — list of percentiles at every 5% step.
* Bootstrap 95% confidence intervals around per-trade Sharpe and expectancy.

Outputs a single JSON report (``analysis.json``) in the input bundle (or
``--output-path``).

The script is intentionally tolerant about missing inputs: if only trades
are available, the equity-curve section is omitted. The pure helpers are
unit-tested in ``test_analyze_trades_script.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from momentum_train.run_training import configure_logging
from momentum_train.trade_metrics import TradeRecord, aggregate_trade_metrics


def _trade_dict_to_record(t: dict) -> TradeRecord:
    """Coerce a JSONL trade record back into a :class:`TradeRecord`.

    Tolerant about missing / extra fields: the canonical KPI aggregator only
    reads the fields listed in the dataclass, so we drop everything else
    (e.g. user-added annotations) and default-fill anything missing.
    """
    return TradeRecord(
        entry_index=int(t.get("entry_index", 0)),
        exit_index=int(t.get("exit_index", 0)),
        duration_steps=int(t.get("duration_steps", 0)),
        entry_price=float(t.get("entry_price", 0.0)),
        exit_price=float(t.get("exit_price", 0.0)),
        entry_portfolio_value=float(t.get("entry_portfolio_value", 0.0)),
        exit_portfolio_value=float(t.get("exit_portfolio_value", 0.0)),
        pnl_pct=float(t.get("pnl_pct", 0.0)),
        pnl_absolute=float(t.get("pnl_absolute", 0.0)),
        max_position=float(t.get("max_position", 0.0)),
        mae_pct=float(t.get("mae_pct", 0.0)),
        mfe_pct=float(t.get("mfe_pct", 0.0)),
        transaction_cost_total=float(t.get("transaction_cost_total", 0.0)),
        actions_taken=list(t.get("actions_taken", [])),
        pct_greedy_actions=float(t.get("pct_greedy_actions", float("nan"))),
    )


LOGGER_NAME = "momentum_train.AnalyzeTrades"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=str,
        help="Path to a *.trades.jsonl file OR a directory containing them.",
    )
    parser.add_argument(
        "--steps-glob",
        type=str,
        default="*.steps.parquet",
        help="Glob (relative to input dir) for per-step parquet files. Falls back to *.steps.csv / *.steps.jsonl when parquet is absent.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="Tail window size for rolling-window aggregates (0 disables).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for Sharpe / expectancy CIs.",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.05,
        help="Two-sided alpha for the bootstrap CI (default 0.05 -> 95%% CI).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the bootstrap (reproducibility).")
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to write analysis.json. Defaults to <input>/analysis.json (or "
        "<dirname(input)>/analysis_<timestamp>.json for single-file inputs).",
    )
    parser.add_argument("--log-level", type=str, default=None)
    return parser.parse_args()


def _load_trades(path: Path) -> list[dict]:
    trades: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a single bad line rather than abort; offline input may
                # have been truncated mid-write by an interrupted run.
                continue
    return trades


def _gather_trade_files(input_path: Path) -> list[Path]:
    """Return the list of *.trades.jsonl files implied by the input."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # Both shapes ``eval_greedy.py`` writes are matched: the per-split
        # subdirs (val/, test/) plus a flat layout for simple uses.
        return sorted({p for p in input_path.rglob("*.trades.jsonl") if p.is_file()})
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _percentile_table(values: Sequence[float]) -> dict[str, float]:
    """5%-step percentiles plus min/max — the trade-PnL CDF in JSON form."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    out: dict[str, float] = {"min": float(arr.min()), "max": float(arr.max())}
    for q in range(5, 100, 5):
        out[f"p{q}"] = float(np.percentile(arr, q))
    return out


def _bootstrap_ci(
    values: Sequence[float],
    *,
    iters: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Bootstrap (sharpe, expectancy) 95% CIs.

    ``sharpe = mean / std`` (population std, not annualized — the per-trade
    KPI defined in :mod:`momentum_train.trade_metrics`). Sample sizes < 2
    short-circuit to NaN since both metrics are undefined.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 2 or iters <= 0:
        nan_ci = {"low": float("nan"), "high": float("nan"), "median": float("nan")}
        return {"sharpe": dict(nan_ci), "expectancy_pct": dict(nan_ci)}

    sharpes = np.empty(iters, dtype=np.float64)
    means = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = arr[rng.integers(0, n, size=n)]
        m = float(sample.mean())
        s = float(sample.std(ddof=0))
        sharpes[i] = m / s if s > 1e-12 else 0.0
        means[i] = m
    lo, hi = alpha / 2.0 * 100.0, (1.0 - alpha / 2.0) * 100.0
    return {
        "sharpe": {
            "low": float(np.percentile(sharpes, lo)),
            "high": float(np.percentile(sharpes, hi)),
            "median": float(np.percentile(sharpes, 50)),
        },
        "expectancy_pct": {
            "low": float(np.percentile(means, lo)),
            "high": float(np.percentile(means, hi)),
            "median": float(np.percentile(means, 50)),
        },
    }


def _split_by_greedy(trades: list[dict]) -> dict[str, list[dict]]:
    """Bucket trades by their dominant provenance.

    A trade is "greedy" if ``pct_greedy_actions >= 0.5`` (i.e. majority of
    in-trade steps came from the greedy policy). NaN provenance lands in
    ``unknown`` so we never silently mis-attribute trades from legacy
    callers.
    """
    greedy: list[dict] = []
    eps: list[dict] = []
    unknown: list[dict] = []
    for t in trades:
        pct = t.get("pct_greedy_actions")
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            unknown.append(t)
        elif float(pct) >= 0.5:
            greedy.append(t)
        else:
            eps.append(t)
    return {"greedy": greedy, "eps": eps, "unknown": unknown}


def _summarize_trades(trades: list[dict]) -> dict:
    """Wrap aggregate_trade_metrics + percentile table into one payload.

    Note ``aggregate_trade_metrics`` expects :class:`TradeRecord` instances,
    so we coerce the dicts (loaded from JSONL) back to records first. The
    canonical key is ``trade_count`` (float) coming from
    :func:`aggregate_trade_metrics`; we expose ``num_trades`` as well for
    callers that prefer the int alias.
    """
    if not trades:
        return {"trade_count": 0.0, "num_trades": 0}
    records = [_trade_dict_to_record(t) for t in trades]
    metrics = dict(aggregate_trade_metrics(records))
    metrics["num_trades"] = int(metrics.get("trade_count", len(trades)))
    metrics["pnl_percentiles"] = _percentile_table([float(t.get("pnl_pct", 0.0)) for t in trades])
    return metrics


def _equity_curve_from_steps_dir(input_dir: Path, steps_glob: str) -> dict:
    """Aggregate per-step portfolio values across whatever per-step files exist.

    Best-effort: parquet preferred (matches eval_greedy output), CSV/JSONL as
    fallbacks. Skipped silently when neither pandas nor any per-step file is
    present.
    """
    pattern = steps_glob
    paths = sorted(input_dir.rglob(pattern))
    if not paths:
        # Fall back: look for csv/jsonl variants.
        paths = sorted(input_dir.rglob("*.steps.csv")) + sorted(input_dir.rglob("*.steps.jsonl"))
    if not paths:
        return {}

    all_values: list[float] = []
    per_file: list[dict] = []
    try:
        import pandas as pd  # type: ignore
    except ImportError:  # pragma: no cover - environment-specific
        pd = None

    for p in paths:
        values: list[float] = []
        try:
            if p.suffix == ".parquet" and pd is not None:
                df = pd.read_parquet(p)
                values = [float(v) for v in df["portfolio_value"].tolist()]
            elif p.suffix == ".csv" and pd is not None:
                df = pd.read_csv(p)
                values = [float(v) for v in df["portfolio_value"].tolist()]
            elif p.suffix == ".jsonl":
                with p.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if "portfolio_value" in rec:
                            values.append(float(rec["portfolio_value"]))
        except Exception:
            continue
        if not values:
            continue
        all_values.extend(values)
        per_file.append(
            {
                "file": p.name,
                "steps": len(values),
                "initial": float(values[0]),
                "final": float(values[-1]),
                "min": float(min(values)),
                "max": float(max(values)),
            }
        )

    if not all_values:
        return {}
    arr = np.asarray(all_values, dtype=np.float64)
    return {
        "files": len(per_file),
        "steps_total": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "per_file": per_file,
    }


def _build_report(
    trades: list[dict],
    *,
    rolling_window: int,
    bootstrap_iters: int,
    bootstrap_alpha: float,
    seed: int,
    equity_curve: dict | None,
) -> dict:
    rng = np.random.default_rng(seed)

    by_provenance = _split_by_greedy(trades)
    pnl_all = [float(t.get("pnl_pct", 0.0)) for t in trades]

    report: dict = {
        "trade_count_total": len(trades),
        "overall": _summarize_trades(trades),
        "by_provenance": {bucket: _summarize_trades(items) for bucket, items in by_provenance.items()},
        "bootstrap_ci": _bootstrap_ci(pnl_all, iters=bootstrap_iters, alpha=bootstrap_alpha, rng=rng),
    }
    if rolling_window > 0 and len(trades) >= rolling_window:
        recent = trades[-rolling_window:]
        report["rolling_window"] = {
            "window": rolling_window,
            "metrics": _summarize_trades(recent),
        }
    if equity_curve:
        report["equity_curve"] = equity_curve
    return report


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level, None)
    logger = logging.getLogger(LOGGER_NAME)

    input_path = Path(args.input).expanduser().resolve()
    trade_files = _gather_trade_files(input_path)
    if not trade_files:
        logger.error("No *.trades.jsonl files found at %s", input_path)
        return 2

    trades: list[dict] = []
    for f in trade_files:
        trades.extend(_load_trades(f))
    logger.info("Loaded %d trades from %d file(s).", len(trades), len(trade_files))

    equity = _equity_curve_from_steps_dir(input_path, args.steps_glob) if input_path.is_dir() else {}

    report = _build_report(
        trades,
        rolling_window=args.rolling_window,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_alpha=args.bootstrap_alpha,
        seed=args.seed,
        equity_curve=equity,
    )
    report["_meta"] = {
        "input": str(input_path),
        "trade_files": [str(p) for p in trade_files],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "rolling_window": args.rolling_window,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_alpha": args.bootstrap_alpha,
        "seed": args.seed,
    }

    if args.output_path:
        out_path = Path(args.output_path).expanduser().resolve()
    elif input_path.is_dir():
        out_path = input_path / "analysis.json"
    else:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = input_path.parent / f"analysis_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    logger.info("Trade analysis written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
