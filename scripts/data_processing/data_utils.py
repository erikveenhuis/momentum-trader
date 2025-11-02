import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Use root logger - configuration handled by main script or calling script
logger = logging.getLogger(__name__)  # Use __name__ for logger

# --- Constants ---
PRICE_COLS: List[str] = ["open", "close", "high", "low"]
EXPECTED_DAILY_ENTRIES: int = 1440

# --- Helper Functions ---


def clear_directory(directory_path: Path):
    """
    Clear all contents of a directory. Creates the directory if it doesn't exist.
    """
    if directory_path.exists():
        logger.info(f"Clearing directory: {directory_path}")
        try:
            shutil.rmtree(directory_path)
            logger.info(f"Directory cleared: {directory_path}")
        except OSError as e:
            logger.error(f"Error clearing directory {directory_path}: {e}")
            # Attempt individual item removal as fallback
            try:
                for item in directory_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                logger.info(f"Cleared contents of directory: {directory_path}")
            except Exception as inner_e:
                logger.error(f"Could not clear contents of directory {directory_path}: {inner_e}")
                # Proceed to recreate, might fail if permissions are the issue
    else:
        logger.info(f"Directory not found, will create: {directory_path}")

    # Recreate the empty directory
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        logger.error(f"Could not create directory {directory_path}: {e}")


def calculate_return(file_path: Path) -> Optional[float]:
    """
    Calculate buy-and-hold return (last_close / first_close - 1) for a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        if "close" not in df.columns:
            logger.warning(f"Skipping return calc: 'close' column missing in {file_path.name}")
            return None
        if len(df) < 2:
            logger.warning(f"Skipping return calc: < 2 data points in {file_path.name}")
            return None

        # Use .loc for potentially non-contiguous indices after filtering/cleaning
        first_valid_index = df["close"].first_valid_index()
        last_valid_index = df["close"].last_valid_index()

        if first_valid_index is None or last_valid_index is None:
            logger.warning(f"Skipping return calc: No valid close prices found in {file_path.name}")
            return None

        first_close = df.loc[first_valid_index, "close"]
        last_close = df.loc[last_valid_index, "close"]

        # Check for NaN/None again just in case .loc returned something unexpected
        if pd.isna(first_close) or first_close == 0:
            logger.warning(f"Skipping return calc: Invalid first close ({first_close}) in {file_path.name}")
            return None
        if pd.isna(last_close):
            logger.warning(f"Skipping return calc: Invalid last close ({last_close}) in {file_path.name}")
            return None

        return (last_close / first_close) - 1.0
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping return calc: File is empty {file_path.name}")
        return None
    except FileNotFoundError:
        logger.error(f"Skipping return calc: File not found {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error calculating return for {file_path.name}: {e}")
        return None


def detect_anomalies(df: pd.DataFrame, threshold: float) -> bool:
    """
    Check for anomalies in price columns based on std dev threshold.
    """
    try:
        if not all(col in df.columns for col in PRICE_COLS):
            logger.warning(f"Cannot detect anomalies: Missing one or more price columns ({PRICE_COLS}).")
            return False  # Cannot determine, assume no anomalies

        price_data = df[PRICE_COLS].dropna()
        if price_data.empty:
            return False  # No data to check

        # Use robust statistics (median absolute deviation) if desired, but std dev is simpler for now
        all_prices = price_data.values.flatten()
        if len(all_prices) < 2:
            return False  # Need at least 2 points for std dev

        mean_price = np.mean(all_prices)
        std_dev_price = np.std(all_prices)

        # Avoid division by zero or near-zero std dev
        if std_dev_price < 1e-9:
            return False  # No significant variation

        z_scores = np.abs((price_data - mean_price) / std_dev_price)

        # Check if ANY z-score exceeds the threshold
        is_anomalous = (z_scores > threshold).any().any()

        return is_anomalous
    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}")
        return False  # Assume not anomalous on error


def detect_ohlc_range_anomalies(df: pd.DataFrame, range_threshold: float, occurrence_threshold: int) -> bool:
    """
    Detects anomalies based on the intra-candle range (high - low) / high.
    Flags the DataFrame as anomalous if the number of such candles exceeds
    the occurrence_threshold.
    """
    try:
        if not all(col in df.columns for col in ["high", "low"]):
            logger.warning("Cannot detect OHLC range anomalies: Missing 'high' or 'low' column.")
            return False  # Cannot determine

        if df[["high", "low"]].isnull().any().any():
            logger.debug("Dropping rows with NaN in high/low for OHLC range anomaly check.")
            # Create a copy to avoid SettingWithCopyWarning if df is a slice
            df_checked = df.dropna(subset=["high", "low"]).copy()
        else:
            df_checked = df.copy()  # Use a copy to add new column safely

        if df_checked.empty:
            return False  # No data to check

        # Avoid division by zero or very small 'high' values leading to huge ratios
        # Replace 'high' values that are zero or very close to zero with np.nan
        # so they are ignored in the percentage calculation or result in NaN percentage_range
        df_checked["high_safe"] = df_checked["high"].replace(0, np.nan)
        # Set 'high_safe' to np.nan if it's too close to zero to avoid extreme ratios
        df_checked.loc[np.abs(df_checked["high_safe"]) < 1e-9, "high_safe"] = np.nan

        # Calculate percentage range, will be NaN if high_safe is NaN (e.g. original high was 0 or too small)
        df_checked["percentage_range"] = (df_checked["high"] - df_checked["low"]) / df_checked["high_safe"]

        # Count anomalies:
        # - percentage_range is not NaN (i.e., high_safe was valid)
        # - percentage_range exceeds the threshold
        # - low must not be greater than high (basic sanity check)
        anomalous_candles_count = df_checked[
            (df_checked["percentage_range"].notna())
            & (df_checked["percentage_range"] > range_threshold)
            & (df_checked["low"] <= df_checked["high"])  # Ensure low is not above high
        ].shape[0]

        if anomalous_candles_count > occurrence_threshold:
            logger.debug(f"Detected {anomalous_candles_count} OHLC range anomalies, exceeding threshold of {occurrence_threshold}.")
            return True
        return False

    except Exception as e:
        logger.error(f"Error during OHLC range anomaly detection: {e}")
        return False  # Assume not anomalous on error
