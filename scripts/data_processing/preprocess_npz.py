"""Convert processed CSV files to compact .npz archives with precomputed derived features.

Reads OHLCV+transactions CSVs from data/processed/{train,validation,test}/ and writes
.npz files alongside them. Each .npz contains:
  - close_prices: float32 [T] — used for trade execution
  - features:     float32 [T, N_FEATURES] — raw + derived features

Feature layout (N_FEATURES = 12):
  [0]  open
  [1]  high
  [2]  low
  [3]  close
  [4]  volume
  [5]  transactions
  --- derived (approximately stationary) ---
  [6]  log_return_1   — log(close[t] / close[t-1])
  [7]  log_return_5   — log(close[t] / close[t-5])
  [8]  log_return_10  — log(close[t] / close[t-10])
  [9]  realized_vol   — rolling std of log_return_1 (20-bar)
  [10] volume_ratio   — volume / rolling_mean(volume, 20)
  [11] hl_range_ratio — (high-low)/close / rolling_mean((high-low)/close, 20)

Raw features (0-5) are z-score normalized per observation window at training time.
Derived features (6-11) are already approximately stationary.
"""

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from momentum_core.logging import get_logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    get_logger = logging.getLogger

logger = get_logger(__name__)

from momentum_env.features import compute_derived_features  # noqa: E402

N_RAW_FEATURES = 6
N_DERIVED_FEATURES = 6
N_TOTAL_FEATURES = N_RAW_FEATURES + N_DERIVED_FEATURES

RAW_COLUMNS = ["open", "high", "low", "close", "volume", "transactions"]
FEATURE_NAMES = RAW_COLUMNS + [
    "log_return_1",
    "log_return_5",
    "log_return_10",
    "realized_vol",
    "volume_ratio",
    "hl_range_ratio",
]

# --- Price-contamination filter ---
# Some minute-bar sources (historically Polygon/Massive for ETC-USD, XTZ-USD, DOGE-USD
# and others in 2017-2021) interleave two venue streams, producing close-price series
# that alternate between two price levels every other bar. A naive RL agent can "learn"
# to round-trip against these artifacts and produce astronomical returns that do not
# generalize to live trading. These constants identify such files.
#
#   - MIN_ALTERNATING_BIG_PAIRS: consecutive bars where both |log return| > 5% AND the
#     two returns have opposite signs. Real markets almost never do this >~80 times in
#     a single day; contaminated files typically hit 100-600.
#   - MAX_ORACLE_LOG_GROWTH: the perfect-trader log-growth a file affords
#     (sum over bars of max(0, log(p[t+1]/p[t]))). Healthy crypto days rarely exceed
#     ~20; we reject >= 25 when the daily range is narrow.
#   - CONTAMINATION_MAX_RANGE_RATIO: price_max / price_min. Real extreme events (LUNA
#     collapse, May 2021 crash, Covid crash) pair huge oracle growth with huge range;
#     contamination pairs huge oracle growth with a narrow range.
MIN_ALTERNATING_BIG_PAIRS = 100
MAX_ORACLE_LOG_GROWTH = 25.0
CONTAMINATION_MAX_RANGE_RATIO = 5.0
BIG_MOVE_LOG_THRESHOLD = float(np.log(1.05))


def check_price_contamination(close_prices: np.ndarray) -> str | None:
    """Return a reason string if close_prices look dual-feed contaminated, else None.

    Two independent checks, either of which triggers rejection:
      1. Many alternating big (>5%) bar-to-bar moves (classic two-venue interleave).
      2. Oracle-like perfect-trader growth that is absurd relative to the daily range.
    """
    if close_prices.size < 2:
        return None

    cp = close_prices.astype(np.float64, copy=False)
    if not np.all(np.isfinite(cp)) or np.any(cp <= 0):
        return "non-finite or non-positive close prices"

    log_returns = np.log(cp[1:] / cp[:-1])
    abs_big = np.abs(log_returns) > BIG_MOVE_LOG_THRESHOLD

    if abs_big.size >= 2:
        # Alternating big-move pairs: adjacent bars both >5% magnitude with opposite signs.
        both_big = abs_big[:-1] & abs_big[1:]
        opposite_sign = log_returns[:-1] * log_returns[1:] < 0.0
        alternating_pairs = int(np.sum(both_big & opposite_sign))
        if alternating_pairs >= MIN_ALTERNATING_BIG_PAIRS:
            return f"alternating >5% bar pairs = {alternating_pairs} (>= {MIN_ALTERNATING_BIG_PAIRS})"

    oracle_log_growth = float(np.sum(np.log(np.maximum(cp[1:] / cp[:-1], 1.0))))
    range_ratio = float(cp.max() / cp.min())
    if oracle_log_growth >= MAX_ORACLE_LOG_GROWTH and range_ratio < CONTAMINATION_MAX_RANGE_RATIO:
        return (
            f"oracle log-growth {oracle_log_growth:.1f} >= {MAX_ORACLE_LOG_GROWTH:.0f} "
            f"with narrow daily range {range_ratio:.2f}x < {CONTAMINATION_MAX_RANGE_RATIO:.0f}x"
        )

    return None


def process_single_csv(csv_path: Path) -> str | None:
    """Convert one CSV to .npz. Returns error message on failure/rejection, None on success.

    Files that fail the price-contamination check are skipped and any pre-existing .npz
    at the same path is removed.
    """
    npz_path = csv_path.with_suffix(".npz")
    try:
        df = pd.read_csv(csv_path)

        required = ["open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            return f"{csv_path.name}: missing columns {missing}"

        df = df.dropna(subset=required)
        if len(df) < 2:
            return f"{csv_path.name}: fewer than 2 rows after dropna"

        if "transactions" not in df.columns:
            df["transactions"] = 0.0

        raw = df[RAW_COLUMNS].values.astype(np.float32)
        close_prices = df["close"].values.astype(np.float32)

        contamination = check_price_contamination(close_prices)
        if contamination is not None:
            if npz_path.exists():
                npz_path.unlink()
            return f"REJECTED {csv_path.name}: {contamination}"

        derived = compute_derived_features(df)

        features = np.concatenate([raw, derived], axis=1)
        assert features.shape == (len(df), N_TOTAL_FEATURES), f"Feature shape mismatch: {features.shape}"

        np.savez_compressed(npz_path, close_prices=close_prices, features=features)
        return None

    except Exception as e:
        return f"{csv_path.name}: {e}"


def run_preprocessing(
    processed_dir: str,
    max_workers: int = 8,
    subdirs: tuple[str, ...] = ("train", "validation", "test"),
    clean: bool = False,
):
    """Convert all CSVs in processed_dir/{subdirs}/ to .npz."""
    base = Path(processed_dir)
    if not base.is_dir():
        logger.error(f"Processed directory not found: {base}")
        return

    if clean:
        for sub in subdirs:
            sub_path = base / sub
            if sub_path.is_dir():
                old_npz = list(sub_path.glob("*.npz"))
                if old_npz:
                    for f in old_npz:
                        f.unlink()
                    logger.info(f"Cleaned {len(old_npz)} old .npz files from {sub_path}")

    all_csvs: list[Path] = []
    for sub in subdirs:
        sub_path = base / sub
        if sub_path.is_dir():
            csvs = sorted(sub_path.glob("*.csv"))
            logger.info(f"Found {len(csvs)} CSV files in {sub_path}")
            all_csvs.extend(csvs)
        else:
            logger.warning(f"Subdirectory not found: {sub_path}")

    if not all_csvs:
        logger.warning("No CSV files found to process.")
        return

    logger.info(f"Processing {len(all_csvs)} CSV files with {max_workers} workers...")

    success = 0
    errors = 0
    rejected = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_csv, p): p for p in all_csvs}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is None:
                success += 1
            elif result.startswith("REJECTED "):
                rejected += 1
                logger.warning(result)
            else:
                errors += 1
                logger.error(f"Error: {result}")
            if i % 5000 == 0:
                logger.info(f"Progress: {i}/{len(all_csvs)} (success={success}, rejected={rejected}, errors={errors})")

    logger.info(
        f"Preprocessing complete. Success: {success}, Rejected (contamination): {rejected}, Errors: {errors}, Total: {len(all_csvs)}"
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / "config" / "split_config.yaml"

    processed_dir = str(base_dir / "data" / "processed")

    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        rel = config.get("split_output_base_dir", "data/processed")
        processed_dir = str(base_dir / rel)

    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV data to .npz format")
    parser.add_argument("--input-dir", default=processed_dir, help="Processed data directory")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--clean", action="store_true", help="Remove existing .npz files before processing")
    args = parser.parse_args()

    run_preprocessing(args.input_dir, max_workers=args.max_workers, clean=args.clean)
