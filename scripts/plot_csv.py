import argparse
import logging
import os
from typing import Optional

import mplfinance as mpf  # For financial plotting
import pandas as pd

try:
    from momentum_core.logging import get_logger, setup_package_logging
except ImportError:  # pragma: no cover - fallback when runtime package missing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.warning("Could not find momentum_core.logging, using basic config.")
    get_logger = logging.getLogger

logger = get_logger("scripts.plot_csv")


def configure_logging(log_level: Optional[str] = None) -> None:
    """Configure logging for the plot_csv utility."""

    if "setup_package_logging" not in globals():
        return

    setup_package_logging(
        "scripts.plot_csv",
        log_filename="plot_csv.log",
        root_level=log_level if log_level is not None else logging.INFO,
        console_level=log_level if log_level is not None else logging.INFO,
        level_overrides={
            "scripts.plot_csv": logging.INFO,
        },
    )


def find_file_in_dir(filename_to_find, search_dir):
    """
    Searches for a file within the specified directory and its subdirectories.
    Returns the full path to the first match found, or None if not found.
    """
    for root, dirs, files in os.walk(search_dir):
        if filename_to_find in files:
            return os.path.join(root, filename_to_find)
    return None


def visualize_csv(csv_filename):
    """
    Finds a CSV file by its name within the 'data/processed/' directory (and subdirectories)
    and plots OHLC candles and volume bars.
    """
    search_path = os.path.join("data", "processed")
    full_file_path = find_file_in_dir(csv_filename, search_path)

    if full_file_path is None:
        logger.error("File '%s' not found within '%s' or its subdirectories.", csv_filename, search_path)
        return

    try:
        # Load the CSV file
        df = pd.read_csv(full_file_path)
        logger.info("Successfully loaded CSV: %s", full_file_path)

        file_name_for_title = csv_filename.replace(".csv", "")

        # --- Data Preparation for mplfinance ---
        required_ohlc_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_ohlc_cols):
            missing_cols = [col for col in required_ohlc_cols if col not in df.columns]
            logger.error(
                "Missing one or more required OHLC columns (%s) in %s. Cannot plot candlestick chart.",
                ", ".join(missing_cols),
                full_file_path,
            )
            return

        # mplfinance expects the index to be DatetimeIndex for time series plots
        if "window_start" in df.columns:
            try:
                df["window_start"] = pd.to_datetime(df["window_start"])
                df.set_index("window_start", inplace=True)
            except Exception as e:
                logger.warning(
                    "Could not convert 'window_start' to DatetimeIndex or set it as index: %s. Plotting may not work as expected.",
                    e,
                )
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                "DataFrame index is not a DatetimeIndex and 'window_start' column is not available or couldn't be converted. Time axis may not be displayed correctly."
            )

        # --- Plotting with mplfinance ---
        plot_kwargs = {
            "type": "candle",
            "title": f"{file_name_for_title} - OHLCV Chart",
            "ylabel": "Price",
            "style": "yahoo",  # Common style, others include: 'charles', 'mike', 'nightclouds'
            # 'mav': (5, 10),  # Example: Add 5 and 10 period moving averages, uncomment to use
            "figsize": (16, 8),  # Adjusted figure size
        }

        if "volume" in df.columns:
            plot_kwargs["volume"] = True
            plot_kwargs["ylabel_lower"] = "Volume"
        else:
            logger.warning("'volume' column not found in %s. Plotting without volume.", full_file_path)

        mpf.plot(df, **plot_kwargs)

    except FileNotFoundError:  # Should be caught by the check above, but good for robustness
        logger.error("File not found at %s", full_file_path)
    except pd.errors.EmptyDataError:
        logger.error("The file %s is empty.", full_file_path)
    except Exception as e:
        logger.exception("An error occurred while processing %s: %s", full_file_path, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize OHLCV data (candlestick and volume) from a CSV file found within the 'data/processed/' directory and its subdirectories."
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Filename of the CSV (e.g., '2021-05-31_ETC-USD.csv'). The script will search for it in 'data/processed/' and its subdirectories.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (e.g. DEBUG, INFO, WARNING). Overrides MOMENTUM_LOG_LEVEL* environment variables.",
    )

    args = parser.parse_args()
    configure_logging(args.log_level)
    logger.info("Starting visualization for %s", args.filename)
    visualize_csv(args.filename)
