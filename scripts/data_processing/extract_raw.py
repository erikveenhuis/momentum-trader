import logging
import logging.handlers
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from momentum_core.logging import get_logger, setup_package_logging
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.warning("Could not find momentum_core.logging, using basic config.")
    get_logger = logging.getLogger

try:
    from .data_utils import clear_directory
except ImportError:
    logging.error("Could not import utility functions from data_utils. Ensure the file exists in the same directory.")
    sys.exit(1)

logger = get_logger("data_processing.extract_raw")


class DiskWriteError(RuntimeError):
    """Raised when extracted CSV data cannot be written to disk."""

    pass


def configure_logging(log_level: str | None = None) -> None:
    """Configure logging for the extract_raw script."""

    if "setup_package_logging" not in globals():
        return

    setup_package_logging(
        "data_processing.extract_raw",
        log_filename="extract_raw_data.log",
        root_level=log_level if log_level is not None else logging.INFO,
        console_level=log_level if log_level is not None else logging.INFO,
        level_overrides={
            "data_processing.extract_raw": logging.INFO,
        },
    )


def worker_log_setup(queue: Any, worker_logger_name: str):
    """Configure logging handler for a worker process."""
    q_handler = logging.handlers.QueueHandler(queue)
    worker_logger = logging.getLogger(worker_logger_name)
    for handler in worker_logger.handlers[:]:
        worker_logger.removeHandler(handler)
    worker_logger.addHandler(q_handler)
    worker_logger.setLevel(logging.DEBUG)


def extract_gz_and_split_by_ticker(
    log_queue: Any,
    input_gz_path: Path,
    output_dir: Path,
    filter_usd_only: bool,
    filter_complete_days: bool,
    exact_datapoints: int,
    exclude_tickers: list[str],
) -> tuple[int, int, int, int, int, int]:
    """Read a compressed CSV, clean tickers, apply filters, and save one raw CSV per
    (date, ticker). Logs are forwarded via ``log_queue`` to the main process.

    Price-contamination filtering is deliberately not done here -- it runs at the .npz
    stage in :func:`preprocess_npz.check_price_contamination`, which is stricter and
    catches dual-feed-interleave artifacts that simple z-score / OHLC-range checks miss.

    Returns:
        (saved, total_skipped_in_file, skipped_incomplete, skipped_non_usd,
         save_error_skips, skipped_excluded_ticker)
    """
    worker_logger_name = "ExtractRawData"
    worker_log_setup(log_queue, worker_logger_name)
    logger = logging.getLogger(worker_logger_name)

    saved_count = 0
    save_error_skips = 0
    skipped_incomplete_count = 0
    skipped_non_usd_count = 0
    skipped_excluded_ticker_count = 0

    try:
        df = pd.read_csv(input_gz_path, compression="gzip")

        if "ticker" not in df.columns or "window_start" not in df.columns:
            logger.warning(f"[{input_gz_path.name}] Skipping: Missing 'ticker' or 'window_start' column.")
            initial_unique_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
            return 0, initial_unique_tickers, 0, 0, 0, 0

        # --- Clean ticker names ---
        initial_tickers_set = set(df["ticker"].unique())
        df["ticker"] = df["ticker"].str.replace(r"^X:", "", regex=True)
        cleaned_tickers_set = set(df["ticker"].unique())
        changed_tickers = initial_tickers_set - cleaned_tickers_set
        if changed_tickers:
            logger.debug(f"[{input_gz_path.name}] Removed 'X:' prefix from {len(changed_tickers)} tickers.")
        if df["ticker"].isnull().any():
            logger.warning(f"[{input_gz_path.name}] Found null tickers after cleaning. Removing affected rows.")
            df = df.dropna(subset=["ticker"])
            if df.empty:
                logger.warning(f"[{input_gz_path.name}] Skipping: DataFrame empty after removing null tickers.")
                return 0, 0, 0, 0, 0, 0

        # --- Convert timestamp and extract date ---
        try:
            df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")
            df["date"] = df["window_start"].dt.date
        except Exception as e:
            logger.error(f"[{input_gz_path.name}] Error converting timestamp or extracting date: {e}. Skipping file.")
            return 0, 0, 0, 0, 0, 0

        all_day_ticker_indices = set(df.set_index(["date", "ticker"]).index)
        indices_to_keep = all_day_ticker_indices.copy()
        skipped_reasons: dict[tuple, str] = {}

        # --- Filter for complete days (optional) ---
        if filter_complete_days:
            daily_counts = df.groupby(["date", "ticker"]).size()
            incomplete_indices = set(daily_counts[daily_counts != exact_datapoints].index)
            for idx in incomplete_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    skipped_reasons[idx] = f"Incomplete (found {daily_counts.loc[idx]}, expected {exact_datapoints})"
                    skipped_incomplete_count += 1
            if incomplete_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(incomplete_indices)} combos as incomplete.")

        # --- Filter USD tickers (optional) ---
        if filter_usd_only:
            non_usd_indices = {idx for idx in indices_to_keep if not idx[1].endswith("-USD")}
            for idx in non_usd_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    if idx not in skipped_reasons:
                        skipped_reasons[idx] = "Non-USD"
                        skipped_non_usd_count += 1
            if non_usd_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(non_usd_indices)} combos as non-USD.")

        # --- Filter excluded tickers (optional) ---
        if exclude_tickers:
            valid_cleaned_tickers = cleaned_tickers_set - set(exclude_tickers)
            excluded_ticker_indices = {idx for idx in indices_to_keep if idx[1] not in valid_cleaned_tickers}
            for idx in excluded_ticker_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    if idx not in skipped_reasons:
                        skipped_reasons[idx] = "Excluded ticker"
                        skipped_excluded_ticker_count += 1
            if excluded_ticker_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(excluded_ticker_indices)} combos as explicitly excluded.")

        skipped_indices = all_day_ticker_indices - indices_to_keep
        if skipped_indices:
            logger.info(f"[{input_gz_path.name}] Individual skip details ({len(skipped_indices)} total):")
            for idx in sorted(skipped_indices):
                reason = skipped_reasons.get(idx, "Unknown reason")
                date_str = idx[0].strftime("%Y-%m-%d")
                ticker_str = idx[1]
                logger.info(f"    - Skipped: {date_str}_{ticker_str}.csv | Reason: {reason}")

        if not indices_to_keep:
            logger.info(f"[{input_gz_path.name}] No day-ticker combinations remaining after all filters.")
            return (
                0,
                len(skipped_indices),
                skipped_incomplete_count,
                skipped_non_usd_count,
                save_error_skips,
                skipped_excluded_ticker_count,
            )

        df_filtered = df[df.set_index(["date", "ticker"]).index.isin(indices_to_keep)]

        for (file_date, ticker_name), ticker_day_df in df_filtered.groupby(["date", "ticker"]):
            if ticker_day_df.empty:
                continue

            date_str = file_date.strftime("%Y-%m-%d")
            safe_ticker = re.sub(r'[\\\\/*?"<>|]', "_", ticker_name)
            output_file = output_dir / f"{date_str}_{safe_ticker}.csv"

            try:
                ticker_day_df.drop(columns=["date"]).sort_values("window_start").to_csv(output_file, index=False)
                saved_count += 1
            except OSError as e:
                if output_file.exists():
                    try:
                        output_file.unlink()
                    except Exception as cleanup_err:
                        logger.warning(
                            f"[{input_gz_path.name}] Failed to remove incomplete file {output_file.name} "
                            f"after disk write error: {cleanup_err}"
                        )
                logger.error(
                    f"[{input_gz_path.name}] Disk write error while saving {output_file.name}: {e}",
                    exc_info=True,
                )
                raise DiskWriteError(f"Disk write error for {output_file}") from e
            except Exception as e:
                logger.error(
                    f"[{input_gz_path.name}] Skipped (Save Error): Ticker {ticker_name} for date {date_str}. Reason: {e}",
                    exc_info=True,
                )
                save_error_skips += 1

        total_skipped_in_file = len(skipped_indices) + save_error_skips
        logger.debug(f"[{input_gz_path.name}] GZ summary - Saved: {saved_count}, Total Skipped: {total_skipped_in_file}")

        return (
            saved_count,
            total_skipped_in_file,
            skipped_incomplete_count,
            skipped_non_usd_count,
            save_error_skips,
            skipped_excluded_ticker_count,
        )

    except pd.errors.EmptyDataError:
        logger.warning(f"[{input_gz_path.name}] Skipping: Input GZ file is empty.")
        return 0, 0, 0, 0, 0, 0
    except FileNotFoundError:
        logger.error(f"[{input_gz_path.name}] Skipping: Input GZ file not found.")
        return 0, 0, 0, 0, 0, 0
    except DiskWriteError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing GZ file {input_gz_path.name}: {e}", exc_info=True)
        return 0, 0, 0, 0, 0, 0


def run_extraction(
    raw_dir: str,
    extracted_dir: str,
    max_workers: int,
    clear_output: bool,
    filter_usd_only: bool,
    filter_complete_days: bool,
    exact_datapoints: int,
    year_filter: str | None,
    exclude_tickers: list[str],
):
    """Find raw ``.csv.gz`` files and run :func:`extract_gz_and_split_by_ticker` in
    parallel. Logs from worker processes are forwarded via a ``Manager`` queue."""
    logger = logging.getLogger("ExtractRawData")

    raw_path = Path(raw_dir)
    extracted_path = Path(extracted_dir)

    if not raw_path.exists() or not raw_path.is_dir():
        logger.error(f"Raw data directory not found: {raw_path}")
        return

    if clear_output:
        clear_directory(extracted_path)
    else:
        extracted_path.mkdir(parents=True, exist_ok=True)

    files_to_process: list[Path] = []
    search_path = raw_path
    if year_filter:
        year_dir = raw_path / year_filter
        if year_dir.exists() and year_dir.is_dir():
            logger.info(f"Filtering for files from year: {year_filter}")
            search_path = year_dir
        else:
            logger.warning(f"Year directory '{year_filter}' not found in {raw_path}, searching base raw directory.")

    logger.info(f"Searching for *.csv.gz files in {search_path} and its subdirectories...")
    files_to_process = sorted(search_path.rglob("*.csv.gz"))

    if not files_to_process:
        logger.warning(f"No *.csv.gz files found in {search_path} or subdirectories matching the criteria.")
        return

    logger.info(f"Found {len(files_to_process)} raw files to process.")

    with Manager() as manager:
        log_queue = manager.Queue(-1)

        root_logger = logging.getLogger()
        handlers = root_logger.handlers[:]
        if not handlers:
            logger.warning("No handlers found on root logger. Logging from workers might be lost.")

        listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
        listener.start()
        logger.info("Log listener started for worker processes.")

        total_saved = 0
        total_skipped = 0
        total_skipped_incomplete = 0
        total_skipped_non_usd = 0
        total_save_errors = 0
        total_skipped_excluded = 0
        processed_files_count = 0

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        extract_gz_and_split_by_ticker,
                        log_queue,
                        file,
                        extracted_path,
                        filter_usd_only,
                        filter_complete_days,
                        exact_datapoints,
                        exclude_tickers,
                    ): file
                    for file in files_to_process
                }

                for future in futures:
                    gz_file = futures[future]
                    processed_files_count += 1
                    try:
                        saved, skipped, incomplete, non_usd, save_err, excluded_err = future.result()
                        total_saved += saved
                        total_skipped += skipped
                        total_skipped_incomplete += incomplete
                        total_skipped_non_usd += non_usd
                        total_save_errors += save_err
                        total_skipped_excluded += excluded_err

                        if processed_files_count % 100 == 0 or (saved > 0 or skipped > 0):
                            skip_summary = (
                                f"TOTALS - Saved: {total_saved}, Skipped: {total_skipped} | "
                                f"Incomplete: {total_skipped_incomplete}, NonUSD: {total_skipped_non_usd}, "
                                f"SaveErr: {total_save_errors}, Excluded Ticker: {total_skipped_excluded}"
                            )
                            logger.info(f"Processed GZ {processed_files_count}/{len(files_to_process)} ({gz_file.name}) -> {skip_summary}")

                    except DiskWriteError as exc:
                        logger.error(
                            f"Aborting extraction due to disk write error while processing {gz_file.name}: {exc}",
                            exc_info=True,
                        )
                        raise
                    except Exception as exc:
                        logger.error(f"Error getting result for GZ file {gz_file.name}: {exc}", exc_info=True)

        finally:
            listener.stop()
            logger.info("Log listener stopped.")

    logger.info("Extraction complete.")
    logger.info(f"  TOTAL Files Saved: {total_saved}")
    logger.info(f"  TOTAL Combinations Skipped: {total_skipped}")
    logger.info(f"    - Reason Incomplete: {total_skipped_incomplete}")
    logger.info(f"    - Reason Non-USD: {total_skipped_non_usd}")
    logger.info(f"    - Reason Save Error: {total_save_errors}")
    logger.info(f"    - Reason Excluded Ticker: {total_skipped_excluded}")
    logger.info(f"Extracted data saved to: {extracted_path}")


if __name__ == "__main__":
    configure_logging()

    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / "config" / "extract_raw_config.yaml"
    config: dict = {}
    try:
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/extract_raw_config.yaml'.")
            logger.info(
                "Example `extract_raw_config.yaml`:\n"
                "raw_dir: data/raw\n"
                "extracted_dir: data/extracted\n"
                "max_workers: 12\n"
                "clear_output: true\n"
                "filter_usd_only: true\n"
                "filter_complete_days: true\n"
                "exact_datapoints: 1440\n"
                "year_filter: null          # e.g. '2023' to process only that year\n"
                "exclude_tickers: []        # e.g. ['UST-USD', 'DAI-USD']"
            )
            sys.exit(1)
    except ImportError:
        logger.error("PyYAML library not found. Install with `pip install pyyaml`.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}.")
        sys.exit(1)

    try:
        raw_dir_rel = config["raw_dir"]
        extracted_dir_rel = config["extracted_dir"]
        max_workers = config["max_workers"]
        clear_output = config["clear_output"]
        filter_usd_only = config["filter_usd_only"]
        filter_complete_days = config["filter_complete_days"]
        exact_datapoints = config["exact_datapoints"]
        year_filter = config["year_filter"]
        exclude_tickers = config.get("exclude_tickers", [])
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)

    raw_dir = base_dir / raw_dir_rel
    extracted_dir = base_dir / extracted_dir_rel

    logger.info(f"Starting raw data extraction. Raw dir: {raw_dir}, Extracted dir: {extracted_dir}")
    logger.info(
        f"Settings: USD Only={filter_usd_only}, Complete Days={filter_complete_days} "
        f"({exact_datapoints} points), Year={year_filter or 'All'}"
    )
    if exclude_tickers:
        logger.info(
            f"Excluding {len(exclude_tickers)} tickers: {', '.join(exclude_tickers[:10])}{'...' if len(exclude_tickers) > 10 else ''}"
        )

    run_extraction(
        raw_dir=str(raw_dir),
        extracted_dir=str(extracted_dir),
        max_workers=max_workers,
        clear_output=clear_output,
        filter_usd_only=filter_usd_only,
        filter_complete_days=filter_complete_days,
        exact_datapoints=exact_datapoints,
        year_filter=year_filter,
        exclude_tickers=exclude_tickers,
    )

    logger.info("Extraction script finished.")
