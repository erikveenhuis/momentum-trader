import logging

# Add multiprocessing queue and logging handlers
import logging.handlers
import re
import sys
from concurrent.futures import ProcessPoolExecutor

# Use Manager Queue for better cross-platform compatibility
from multiprocessing import Manager
from pathlib import Path

# Import Any for type hinting queues
from typing import Any, List, Optional, Tuple

import pandas as pd
import yaml

# Import logging configuration
try:
    from momentum_train.logging_config import get_logger, setup_logging
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.warning("Could not find logging_config, using basic config.")
    get_logger = logging.getLogger

# Import shared functions/constants
try:
    # Import detect_anomalies and PRICE_COLS
    from .data_utils import (
        PRICE_COLS,
        clear_directory,
        detect_anomalies,
        detect_ohlc_range_anomalies,
    )
except ImportError:
    logging.error("Could not import utility functions from data_utils. Ensure the file exists in the same directory.")
    sys.exit(1)

# Get logger instance for the main process (used before multiprocessing starts)
logger = get_logger("ExtractRawData")


# Worker process logging setup function
def worker_log_setup(queue: Any, worker_logger_name: str):
    """Configure logging handler for worker process."""
    q_handler = logging.handlers.QueueHandler(queue)
    worker_logger = logging.getLogger(worker_logger_name)
    # Remove existing handlers if any leaked into the child process
    for handler in worker_logger.handlers[:]:
        worker_logger.removeHandler(handler)
    worker_logger.addHandler(q_handler)
    # Set level high enough to capture desired messages (INFO, DEBUG etc.)
    # The listener in the main process respects the main config levels
    worker_logger.setLevel(logging.DEBUG)  # Or logging.INFO based on needs


def extract_gz_and_split_by_ticker(
    # Parameter type hint remains Queue (or could be generalized)
    log_queue: Any,
    input_gz_path: Path,
    output_dir: Path,
    filter_usd_only: bool,
    filter_complete_days: bool,
    exact_datapoints: int,
    # Add anomaly filter parameters
    filter_anomalies: bool,
    anomaly_threshold: float,
    ohlc_range_anomaly_threshold: float,
    ohlc_range_anomaly_occurrence_threshold: int,
    # Add exclude_tickers list parameter
    exclude_tickers: List[str],
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Reads a compressed CSV, cleans tickers, optionally filters USD pairs,
    complete days, anomalies, and excluded tickers, and saves one raw CSV per ticker for that day.
    Sends logs back to the main process via a queue.

    Args:
        log_queue (Queue): Queue to send log records to the main process.
        input_gz_path (Path): Path to the input compressed CSV file (.csv.gz).
        output_dir (Path): Directory to save the extracted ticker CSV files.
        filter_usd_only (bool): If True, only process tickers ending in "-USD".
        filter_complete_days (bool): If True, only process days with at least exact_datapoints.
        exact_datapoints (int): Minimum number of data points required for a day to be considered complete.
        filter_anomalies (bool): If True, filter based on anomaly detection.
        anomaly_threshold (float): Threshold for Z-score anomaly detection.
        ohlc_range_anomaly_threshold (float): Threshold for OHLC range anomaly detection ((high-low)/high).
        ohlc_range_anomaly_occurrence_threshold (int): Min occurrences for OHLC range anomaly to flag file.
        exclude_tickers (List[str]): List of tickers to exclude.

    Returns:
        Tuple[int, int, int, int, int, int, int]:
            saved_count,
            total_skipped_in_file,
            skipped_incomplete_count,
            skipped_non_usd_count,
            skipped_anomalous_count,
            save_error_skips,
            skipped_excluded_ticker_count
    """
    # Configure logging for this worker process to use the queue
    worker_logger_name = "ExtractRawData"  # Use the same name as the main logger
    worker_log_setup(log_queue, worker_logger_name)
    # Get the logger configured by worker_log_setup
    logger = logging.getLogger(worker_logger_name)

    saved_count = 0
    save_error_skips = 0  # Counter for save errors specifically
    skipped_incomplete_count = 0
    skipped_non_usd_count = 0
    skipped_anomalous_count = 0
    skipped_excluded_ticker_count = 0  # Initialize new counter

    try:
        df = pd.read_csv(input_gz_path, compression="gzip")

        if "ticker" not in df.columns or "window_start" not in df.columns:
            logger.warning(f"[{input_gz_path.name}] Skipping: Missing 'ticker' or 'window_start' column.")
            # Estimate skipped count based on unique tickers if present
            initial_unique_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
            return 0, initial_unique_tickers, 0, 0, 0, 0, 0

        # --- Clean Ticker Names ---
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
                # All original tickers potentially skipped
                return 0, 0, 0, 0, 0, 0, 0

        # --- Convert Timestamp and Extract Date ---
        try:
            # Ensure correct unit based on data source (ns for Polygon)
            df["window_start"] = pd.to_datetime(df["window_start"], unit="ns")
            df["date"] = df["window_start"].dt.date  # Extract date for grouping/filtering
        except Exception as e:
            logger.error(f"[{input_gz_path.name}] Error converting timestamp or extracting date: {e}. Skipping file.")
            # Estimate skipped tickers
            return 0, 0, 0, 0, 0, 0, 0

        # Calculate initial number of unique day-ticker combinations
        # We'll use a set to keep track of combinations to keep
        all_day_ticker_indices = set(df.set_index(["date", "ticker"]).index)
        indices_to_keep = all_day_ticker_indices.copy()  # Start assuming we keep all
        skipped_reasons = {}  # Dictionary to store reason for skipping: (date, ticker) -> reason_str

        # --- Filter for Complete Days (Optional) ---
        if filter_complete_days:
            daily_counts = df.groupby(["date", "ticker"]).size()
            incomplete_indices = set(daily_counts[daily_counts != exact_datapoints].index)
            for idx in incomplete_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    skipped_reasons[idx] = f"Incomplete (found {daily_counts.loc[idx]}, expected {exact_datapoints})"
                    skipped_incomplete_count += 1  # Increment counter
            # Keep debug log for marking, main count logged at end
            if incomplete_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(incomplete_indices)} combos as incomplete.")

        # --- Filter USD Tickers (Optional) ---
        if filter_usd_only:
            non_usd_indices = set()
            # Iterate through combinations potentially kept so far
            for date_idx, ticker_idx in indices_to_keep:
                if not ticker_idx.endswith("-USD"):
                    non_usd_indices.add((date_idx, ticker_idx))

            for idx in non_usd_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    # Don't overwrite skip reason if already skipped for completeness
                    if idx not in skipped_reasons:
                        skipped_reasons[idx] = "Non-USD"
                        skipped_non_usd_count += 1  # Increment counter only if this is the primary reason
                    # If already skipped for another reason, still potentially non-usd, but don't double count primary skip reason

            # Keep debug log for marking
            if non_usd_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(non_usd_indices)} combos as non-USD.")

        # --- Filter Excluded Tickers (Optional) ---
        if exclude_tickers:
            excluded_ticker_indices = set()
            # Use the original cleaned tickers set for efficiency
            valid_cleaned_tickers = cleaned_tickers_set - set(exclude_tickers)
            # Iterate through combinations potentially kept so far
            for date_idx, ticker_idx in indices_to_keep:
                if ticker_idx not in valid_cleaned_tickers:
                    excluded_ticker_indices.add((date_idx, ticker_idx))

            for idx in excluded_ticker_indices:
                if idx in indices_to_keep:
                    indices_to_keep.remove(idx)
                    if idx not in skipped_reasons:
                        skipped_reasons[idx] = "Excluded ticker"
                        skipped_excluded_ticker_count += 1  # Increment new counter

            if excluded_ticker_indices:
                logger.debug(f"[{input_gz_path.name}] Marked {len(excluded_ticker_indices)} combos as explicitly excluded.")

        # --- Filter Anomalies (Optional, after other filters) ---
        if filter_anomalies:
            anomalous_indices = set()
            if not all(col in df.columns for col in PRICE_COLS):
                logger.warning(
                    f"[{input_gz_path.name}] Cannot perform anomaly check: Missing required columns ({PRICE_COLS}). Skipping anomaly filter for this file."
                )
            else:
                # Group by the original dataframe BUT only check groups we intend to keep
                grouped = df[df.set_index(["date", "ticker"]).index.isin(indices_to_keep)].groupby(["date", "ticker"])
                for index, group_df in grouped:
                    is_anomalous_zscore = detect_anomalies(group_df, anomaly_threshold)
                    is_anomalous_ohlc_range = detect_ohlc_range_anomalies(
                        group_df,
                        ohlc_range_anomaly_threshold,
                        ohlc_range_anomaly_occurrence_threshold,
                    )

                    if is_anomalous_zscore or is_anomalous_ohlc_range:
                        anomalous_indices.add(index)
                        # Log which filter caught it for better debugging
                        reason_parts = []
                        if is_anomalous_zscore:
                            reason_parts.append(f"Z-score (std_dev>{anomaly_threshold})")
                        if is_anomalous_ohlc_range:
                            reason_parts.append(
                                f"OHLC range (range>{ohlc_range_anomaly_threshold*100}% in >{ohlc_range_anomaly_occurrence_threshold} candles)"
                            )
                        skipped_reasons[index] = f"Anomaly ({'; '.join(reason_parts)})"
                        # The skipped_anomalous_count will be incremented later if this is the primary skip reason

                for idx in anomalous_indices:
                    if idx in indices_to_keep:
                        indices_to_keep.remove(idx)
                        if idx not in skipped_reasons:  # Should have been added above with details
                            # This case might not be strictly necessary if skipped_reasons is always populated above
                            skipped_reasons[idx] = "Anomaly (mixed criteria)"
                        skipped_anomalous_count += 1  # Increment counter only if this is primary reason
                # Keep debug log for marking
                if anomalous_indices:
                    logger.debug(f"[{input_gz_path.name}] Marked {len(anomalous_indices)} combos as anomalous.")

        # --- Log individual skipped files (DEBUG Level) ---
        skipped_indices = all_day_ticker_indices - indices_to_keep
        if skipped_indices:
            # Remove the INFO level summary logs per GZ file
            # logger.info(f"[{input_gz_path.name}] Skipped {len(skipped_indices)} day-ticker combinations (see details below):")
            # if skipped_incomplete_count > 0: logger.info(f"    - Reason Incomplete: {skipped_incomplete_count}")
            # if skipped_non_usd_count > 0: logger.info(f"    - Reason Non-USD: {skipped_non_usd_count}")
            # if skipped_anomalous_count > 0: logger.info(f"    - Reason Anomaly: {skipped_anomalous_count}")
            # if skipped_excluded_ticker_count > 0: logger.info(f"    - Reason Excluded Ticker: {skipped_excluded_ticker_count}") # Added reason log

            logger.info(f"[{input_gz_path.name}] Individual skip details ({len(skipped_indices)} total):")
            sorted_skips = sorted(list(skipped_indices))
            for idx in sorted_skips:
                reason = skipped_reasons.get(idx, "Unknown reason")
                date_str = idx[0].strftime("%Y-%m-%d")
                ticker_str = idx[1]
                logger.info(f"    - Skipped: {date_str}_{ticker_str}.csv | Reason: {reason}")

        # --- Filter DataFrame and Save Kept Files ---
        if not indices_to_keep:
            logger.info(f"[{input_gz_path.name}] No day-ticker combinations remaining after all filters.")
            total_filter_skips = len(skipped_indices)
            # Return counts for this file, including new counter
            return (
                0,
                total_filter_skips,
                skipped_incomplete_count,
                skipped_non_usd_count,
                skipped_anomalous_count,
                save_error_skips,
                skipped_excluded_ticker_count,
            )

        # Filter the dataframe down to only the rows that should be kept
        df_filtered = df[df.set_index(["date", "ticker"]).index.isin(indices_to_keep)]

        # --- Group by Ticker and Date, then Save ---
        for (file_date, ticker_name), ticker_day_df in df_filtered.groupby(["date", "ticker"]):
            # Groupby might yield empty groups if indices_to_keep logic had issues, add safety check
            if ticker_day_df.empty:
                continue

            try:
                date_str = file_date.strftime("%Y-%m-%d")
                safe_ticker = re.sub(r'[\\\\/*?"<>|]', "_", ticker_name)  # Clean ticker for filename
                output_file = output_dir / f"{date_str}_{safe_ticker}.csv"

                # Save the filtered, non-anomalous raw data
                ticker_day_df.drop(columns=["date"]).sort_values("window_start").to_csv(output_file, index=False)
                saved_count += 1
                # logger.debug(f"[{input_gz_path.name}] Saved extracted file: {output_file.name}")

            except Exception as e:
                # Log remains ERROR, clearly indicates a save failure skip
                logger.error(f"[{input_gz_path.name}] Skipped (Save Error): Ticker {ticker_name} for date {date_str}. Reason: {e}")
                save_error_skips += 1  # Use the dedicated counter

        # Calculate total skips for this file
        total_filter_skips = len(skipped_indices)
        total_skipped_in_file = total_filter_skips + save_error_skips

        # Remove per-file save error summary log
        # if save_error_skips > 0:
        #     logger.info(f"[{input_gz_path.name}] Skipped (Save Error): {save_error_skips} files (see ERROR logs for details).")

        logger.debug(f"[{input_gz_path.name}] GZ summary - Saved: {saved_count}, Total Skipped: {total_skipped_in_file}")

        # Return detailed counts, including new counter
        return (
            saved_count,
            total_skipped_in_file,
            skipped_incomplete_count,
            skipped_non_usd_count,
            skipped_anomalous_count,
            save_error_skips,
            skipped_excluded_ticker_count,
        )

    except pd.errors.EmptyDataError:
        logger.warning(f"[{input_gz_path.name}] Skipping: Input GZ file is empty.")
        # Update return to include new count
        return 0, 0, 0, 0, 0, 0, 0
    except FileNotFoundError:
        logger.error(f"[{input_gz_path.name}] Skipping: Input GZ file not found.")
        # Update return to include new count
        return 0, 0, 0, 0, 0, 0, 0
    except Exception as e:
        logger.error(f"Unexpected error processing GZ file {input_gz_path.name}: {e}", exc_info=True)
        # Cannot reliably determine counts, return zeros, including new counter
        return 0, 0, 0, 0, 0, 0, 0


def run_extraction(
    raw_dir: str,
    extracted_dir: str,
    max_workers: int,
    clear_output: bool,
    filter_usd_only: bool,
    filter_complete_days: bool,
    exact_datapoints: int,
    # Add anomaly filter parameters
    filter_anomalies: bool,
    anomaly_threshold: float,
    year_filter: Optional[str],
    # Add exclude_tickers list parameter
    exclude_tickers: List[str],
    ohlc_range_anomaly_threshold: float,
    ohlc_range_anomaly_occurrence_threshold: int,
):
    """
    Finds raw .csv.gz files and runs the extraction process in parallel.
    Sets up logging queue using Manager for worker processes.

    Args:
        raw_dir (str): Directory containing raw .csv.gz files.
        extracted_dir (str): Directory to save extracted files.
        max_workers (int): Number of parallel workers.
        clear_output (bool): Whether to clear the output directory before extraction.
        filter_usd_only (bool): Only process tickers ending in "-USD".
        filter_complete_days (bool): Only process days with exact datapoints.
        exact_datapoints (int): Number of expected datapoints per day.
        filter_anomalies (bool): Whether to filter anomalies.
        anomaly_threshold (float): Z-score threshold for anomaly detection.
        year_filter (Optional[str]): Year to filter files by.
        exclude_tickers (List[str]): List of tickers to exclude.
        ohlc_range_anomaly_threshold (float): Threshold for OHLC range anomaly detection ((high-low)/high).
        ohlc_range_anomaly_occurrence_threshold (int): Min occurrences for OHLC range anomaly to flag file.
    """
    # Use the main process logger instance
    logger = logging.getLogger("ExtractRawData")

    raw_path = Path(raw_dir)
    extracted_path = Path(extracted_dir)

    if not raw_path.exists() or not raw_path.is_dir():
        logger.error(f"Raw data directory not found: {raw_path}")
        return

    if clear_output:
        clear_directory(extracted_path)
    else:
        extracted_path.mkdir(parents=True, exist_ok=True)  # Ensure exists

    # --- File Discovery ---
    files_to_process = []
    search_path = raw_path
    if year_filter:
        year_dir = raw_path / year_filter
        if year_dir.exists() and year_dir.is_dir():
            logger.info(f"Filtering for files from year: {year_filter}")
            search_path = year_dir
        else:
            logger.warning(f"Year directory '{year_filter}' not found in {raw_path}, searching base raw directory.")  # Changed to warning

    logger.info(f"Searching for *.csv.gz files in {search_path} and its subdirectories...")
    files_to_process = sorted(list(search_path.rglob("*.csv.gz")))  # Use rglob for recursive search

    if not files_to_process:
        # Changed to warning as it might be expected if year filter is specific
        logger.warning(f"No *.csv.gz files found in {search_path} or subdirectories matching the criteria.")
        return

    logger.info(f"Found {len(files_to_process)} raw files to process.")

    # --- Set up Logging Queue and Listener ---
    # Use a Manager to create a shared queue
    with Manager() as manager:
        log_queue = manager.Queue(-1)

        root_logger = logging.getLogger()
        handlers = root_logger.handlers[:]
        if not handlers:
            logger.warning("No handlers found on root logger. Logging from workers might be lost.")
            # Consider adding a basic console handler if needed

        listener = logging.handlers.QueueListener(log_queue, *handlers, respect_handler_level=True)
        listener.start()
        logger.info("Log listener started for worker processes.")

        # --- Parallel Extraction ---
        total_saved = 0
        total_skipped = 0
        total_skipped_incomplete = 0
        total_skipped_non_usd = 0
        total_skipped_anomalous = 0
        total_save_errors = 0
        total_skipped_excluded = 0  # Initialize new total counter
        processed_files_count = 0

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        extract_gz_and_split_by_ticker,
                        log_queue,  # Pass the manager queue
                        file,
                        extracted_path,
                        filter_usd_only,
                        filter_complete_days,
                        exact_datapoints,
                        filter_anomalies,
                        anomaly_threshold,
                        ohlc_range_anomaly_threshold,
                        ohlc_range_anomaly_occurrence_threshold,
                        exclude_tickers,  # Pass the exclude list
                    ): file
                    for file in files_to_process
                }

                for future in futures:
                    gz_file = futures[future]
                    processed_files_count += 1
                    try:
                        # Unpack the detailed results, including new counter
                        saved, skipped, incomplete, non_usd, anomalous, save_err, excluded_err = future.result()
                        # Aggregate totals
                        total_saved += saved
                        total_skipped += skipped
                        total_skipped_incomplete += incomplete
                        total_skipped_non_usd += non_usd
                        total_skipped_anomalous += anomalous
                        total_save_errors += save_err
                        total_skipped_excluded += excluded_err  # Aggregate new count

                        # Update progress log to include cumulative reason counts
                        if processed_files_count % 100 == 0 or (saved > 0 or skipped > 0):
                            skip_summary = (
                                f"TOTALS - Saved: {total_saved}, Skipped: {total_skipped} | "
                                f"Incomplete: {total_skipped_incomplete}, NonUSD: {total_skipped_non_usd}, "
                                f"Anomaly: {total_skipped_anomalous}, SaveErr: {total_save_errors}, "  # Added comma
                                f"Excluded Ticker: {total_skipped_excluded}"  # Added new count
                            )
                            logger.info(f"Processed GZ {processed_files_count}/{len(files_to_process)} ({gz_file.name}) -> {skip_summary}")

                    except Exception as exc:
                        logger.error(f"Error getting result for GZ file {gz_file.name}: {exc}", exc_info=True)

        finally:
            # --- Stop the Logging Listener ---
            # Ensure listener stops even if errors occur in the pool
            listener.stop()
            logger.info("Log listener stopped.")

    # Final summary log with cumulative breakdown
    logger.info("Extraction complete.")
    logger.info(f"  TOTAL Files Saved: {total_saved}")
    logger.info(f"  TOTAL Combinations Skipped: {total_skipped}")
    logger.info(f"    - Reason Incomplete: {total_skipped_incomplete}")
    logger.info(f"    - Reason Non-USD: {total_skipped_non_usd}")
    logger.info(f"    - Reason Anomaly: {total_skipped_anomalous}")
    logger.info(f"    - Reason Save Error: {total_save_errors}")
    logger.info(f"    - Reason Excluded Ticker: {total_skipped_excluded}")  # Added new summary line
    logger.info(f"Extracted data saved to: {extracted_path}")


if __name__ == "__main__":
    # --- Logging Setup ---
    log_file = Path("logs") / "extract_raw_data.log"
    setup_logging(
        log_file_path=log_file,
        root_level=logging.INFO,
        level_overrides={
            "ExtractRawData": logging.INFO,
        },
        console_level=logging.INFO,  # Ensure console shows INFO
    )
    # ---------------------

    # --- Configuration Loading ---
    base_dir = Path(__file__).resolve().parent.parent.parent
    # Assume config is named extract_raw_config.yaml
    config_path = base_dir / "config" / "extract_raw_config.yaml"
    config = {}
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if config is None:
                config = {}  # Handle empty config file
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/extract_raw_config.yaml'.")
            # Provide a more complete example including new params
            logger.info(
                "Example `extract_raw_config.yaml`:\n"
                "raw_dir: data/polygon/stocks_raw # Or polygon/crypto_raw etc.\n"
                "extracted_dir: data/extracted\n"
                "max_workers: 4\n"
                "clear_output: true\n"
                "filter_usd_only: true\n"
                "filter_complete_days: true\n"
                "exact_datapoints: 1440 # Example for minute data\n"
                "filter_anomalies: true             # Added\n"
                "anomaly_std_dev_threshold: 4.0   # Added\n"
                "year_filter: null # Optional: e.g., '2023' to process only that year\n"
                "exclude_tickers: [] # Optional: List of tickers to exclude, e.g., ['SHIB-USD', 'DOGE-USD']"
            )  # Added example
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
        filter_anomalies = config["filter_anomalies"]
        anomaly_threshold = config["anomaly_std_dev_threshold"]
        year_filter = config["year_filter"]
        exclude_tickers = config["exclude_tickers"]
        ohlc_range_anomaly_threshold = config["ohlc_range_anomaly_threshold"]
        ohlc_range_anomaly_occurrence_threshold = config["ohlc_range_anomaly_occurrence_threshold"]

    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ---------------------------

    # --- Path Construction ---
    raw_dir = base_dir / raw_dir_rel
    extracted_dir = base_dir / extracted_dir_rel
    # -------------------------

    logger.info(f"Starting raw data extraction. Raw dir: {raw_dir}, Extracted dir: {extracted_dir}")
    logger.info(
        f"Settings: USD Only={filter_usd_only}, Complete Days={filter_complete_days} ({exact_datapoints} points), Filter Anomalies={filter_anomalies} (threshold {anomaly_threshold}), Year={year_filter or 'All'}"
    )
    if exclude_tickers:
        logger.info(
            f"Excluding {len(exclude_tickers)} tickers: {', '.join(exclude_tickers[:10])}{'...' if len(exclude_tickers) > 10 else ''}"
        )  # Log excluded tickers

    run_extraction(
        raw_dir=str(raw_dir),
        extracted_dir=str(extracted_dir),
        max_workers=max_workers,
        clear_output=clear_output,
        filter_usd_only=filter_usd_only,
        filter_complete_days=filter_complete_days,
        exact_datapoints=exact_datapoints,
        filter_anomalies=filter_anomalies,
        anomaly_threshold=anomaly_threshold,
        year_filter=year_filter,
        exclude_tickers=exclude_tickers,  # Pass the exclude list
        ohlc_range_anomaly_threshold=ohlc_range_anomaly_threshold,
        ohlc_range_anomaly_occurrence_threshold=ohlc_range_anomaly_occurrence_threshold,
    )

    logger.info("Extraction script finished.")
