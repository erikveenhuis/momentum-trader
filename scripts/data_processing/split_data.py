import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor # Keep for potential future parallel return calculation
import logging
import yaml
import sys
import shutil
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Add src directory to Python path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import logging configuration
from momentum_train.logging_config import setup_logging, get_logger

# Import shared functions/constants
try:
    from .data_utils import clear_directory, calculate_return
except ImportError:
    logging.error("Could not import utility functions from .data_utils. Ensure data_utils.py exists in the same directory.")
    sys.exit(1)

# Get logger instance
logger = get_logger("SplitData")


def perform_train_val_test_split(
    all_files: List[Path],
    output_base_dir: Path, # Base directory where subdirs will be created
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
    test_months: int,  # Changed from test_ratio to test_months
    validation_size: int,
    seed: int,
    clear_output: bool,
):
    """
    Splits the processed CSV files into train, validation, and test sets.
    - Test set is the last N months of data
    - Validation set (validation_size) is sampled from the remaining files
      based on buy-and-hold return distribution.
    - Training set gets the rest.
    Moves files to respective subdirectories under output_base_dir.
    """
    if not all_files:
        logger.warning("No processed files provided for splitting.")
        return

    # Extract dates from filenames and sort files by date
    def get_date_from_filename(file_path: Path) -> datetime:
        # Handle both formats:
        # - YYYY-MM-DD_SYMBOL-USD.csv (e.g. 2018-03-13_EDO-USD.csv)
        # - SYMBOL-USD_YYYY-MM-DD.csv (e.g. BTC-USD_2023-01-01.csv)
        try:
            filename = file_path.stem  # Get filename without extension
            parts = filename.split('_')
            if len(parts) != 2:
                logger.warning(f"Could not parse date from filename: {file_path.name} - Expected format YYYY-MM-DD_SYMBOL-USD.csv or SYMBOL-USD_YYYY-MM-DD.csv")
                return datetime.min
            
            # Try both parts to find the date
            for part in parts:
                try:
                    return datetime.strptime(part, '%Y-%m-%d')
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date from filename: {file_path.name} - No valid date found")
            return datetime.min
            
        except Exception as e:
            logger.warning(f"Could not parse date from filename: {file_path.name} - {str(e)}")
            return datetime.min  # Place files with invalid dates at the start

    # Sort files by date
    files_with_dates = [(f, get_date_from_filename(f)) for f in all_files]
    files_with_dates.sort(key=lambda x: x[1])
    sorted_files = [f for f, _ in files_with_dates]

    # Find the cutoff date for test set (last N months)
    if not files_with_dates:
        logger.error("No valid files found after date parsing.")
        return

    last_date = files_with_dates[-1][1]
    cutoff_date = last_date - relativedelta(months=test_months)
    
    # Split into train+val and test sets
    test_files = []
    initial_train_files = []
    
    for file_path, file_date in files_with_dates:
        if file_date >= cutoff_date:
            test_files.append(file_path)
        else:
            initial_train_files.append(file_path)

    n_total = len(all_files)
    n_test = len(test_files)
    n_initial_train = len(initial_train_files)

    if n_test == 0:
        logger.error(f"No files found in the last {test_months} months. Check your data range.")
        return
    if n_initial_train == 0:
        logger.error("No files found for training/validation. Check your data range.")
        return
    if validation_size <= 0:
        logger.error(f"validation_size ({validation_size}) must be positive.")
        return
    if validation_size >= n_initial_train:
        logger.error(f"validation_size ({validation_size}) must be less than the number of potential training files ({n_initial_train}).")
        return

    logger.info(f"Starting split of {n_total} processed files into Train/Validation/Test.")
    logger.info(f"Test set: Last {test_months} months ({n_test} files)")
    logger.info(f"Target validation set size: {validation_size} (selected from remaining {n_initial_train})")

    # --- Calculate returns for the potential training set ---
    returns_map: List[Tuple[Path, float]] = []
    logger.info("Calculating buy-and-hold returns for potential train/val files...")
    skipped_calc_count = 0
    for file_path in initial_train_files:
        ret = calculate_return(file_path)
        if ret is not None:
            returns_map.append((file_path, ret))
        else:
            skipped_calc_count += 1

    if skipped_calc_count > 0:
        logger.warning(f"Skipped return calculation for {skipped_calc_count} files due to errors or invalid data.")

    # --- Log Top/Bottom 5 Returns --- 
    if returns_map:
        logger.info("--- Top/Bottom 5 Calculated Returns (for Train/Val pool) ---")
        returns_map.sort(key=lambda x: x[1]) # Ensure sorted by return
        
        # Bottom 5
        bottom_5 = returns_map[:5]
        logger.info("Bottom 5 Returns:")
        for file_path, ret_val in bottom_5:
            logger.info(f"  - {file_path.name}: {ret_val:.6f}")
            
        # Top 5
        top_5 = returns_map[-5:]
        logger.info("Top 5 Returns:")
        # Iterate in reverse to show highest first
        for file_path, ret_val in reversed(top_5):
            logger.info(f"  - {file_path.name}: {ret_val:.6f}")
        logger.info("----------------------------------------------------------")
    else:
        logger.warning("No returns calculated, cannot show top/bottom.")
        # Fallback or error handling logic follows

    if not returns_map:
         logger.error("Could not calculate returns for any potential train/val files. Cannot select validation set based on returns.")
         # Fallback: Randomly select validation set
         logger.warning("Falling back to random selection for validation set.")
         try:
             # Split initial_train further into final_train and validation
             final_train_files, validation_files = train_test_split(
                 initial_train_files, test_size=validation_size, random_state=seed, shuffle=True
             )
         except Exception as e:
              logger.error(f"Error during fallback validation split: {e}")
              return
    elif len(returns_map) <= validation_size:
        logger.warning(f"Number of files with valid returns ({len(returns_map)}) is <= validation_size ({validation_size}). Using all files with valid returns for validation.")
        validation_files = [item[0] for item in returns_map]
        # Files without valid returns go to training
        valid_files_set = set(validation_files)
        final_train_files = [f for f in initial_train_files if f not in valid_files_set]
    else:
        # --- Select Validation files based on return distribution ---
        logger.info(f"Selecting {validation_size} validation files based on return distribution...")
        returns_map.sort(key=lambda x: x[1]) # Sort by return value

        # Select indices evenly distributed across the sorted returns
        indices = np.linspace(0, len(returns_map) - 1, validation_size, dtype=int)
        # Ensure unique indices, especially if validation_size is large relative to len(returns_map)
        unique_indices = np.unique(indices)
        validation_files_set = set(returns_map[i][0] for i in unique_indices)
        validation_files = list(validation_files_set)

        # If unique indices resulted in fewer files than requested, add more strategically
        if len(validation_files) < validation_size:
            logger.warning(f"Selected fewer validation files ({len(validation_files)}) than requested ({validation_size}) due to non-unique indices from linspace.")
            needed = validation_size - len(validation_files)
            # Add remaining from those not already selected, trying to maintain spread
            all_return_files = [item[0] for item in returns_map]
            candidates = [f for f in all_return_files if f not in validation_files_set]
            if len(candidates) >= needed:
                 additional_files = candidates[:needed]
                 validation_files.extend(additional_files)
                 validation_files_set.update(additional_files)
                 logger.info(f"Added {len(additional_files)} more files to validation set.")
            else:
                 logger.error(f"Cannot add more files to validation set, only {len(candidates)} candidates available.")
                 # Proceeding with fewer validation files

        # --- Final Training set is the remainder ---
        final_train_files = [
            f for f in initial_train_files if f not in validation_files_set
        ]

        # Log some info about selected validation files' returns
        val_returns = [ret for path, ret in returns_map if path in validation_files_set]
        if val_returns:
            logger.info(f"Selected validation file returns - Min: {min(val_returns):.4f}, Max: {max(val_returns):.4f}, Mean: {np.mean(val_returns):.4f}, Std: {np.std(val_returns):.4f}")

    # --- Final Size Calculation and Consistency Check ---
    n_train = len(final_train_files)
    n_val = len(validation_files)
    logger.info(f"Final split sizes: Train={n_train}, Validation={n_val}, Test={n_test}")
    if n_train + n_val + n_test != n_total:
         logger.warning(f"Consistency check failed: Train({n_train}) + Val({n_val}) + Test({n_test}) = {n_train+n_val+n_test} != Total({n_total})")

    # --- Create Directories and Move Files ---
    train_path = output_base_dir / train_subdir
    val_path = output_base_dir / val_subdir
    test_path = output_base_dir / test_subdir

    if clear_output:
        logger.info(f"Clearing output subdirectories under {output_base_dir}...")
        clear_directory(train_path)
        clear_directory(val_path)
        clear_directory(test_path)
    else:
        # Ensure directories exist even if not clearing
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)

    moved_train, moved_val, moved_test = 0, 0, 0
    errors_moving = 0

    def move_files(file_list: List[Path], dest_path: Path) -> Tuple[int, int]:
        count = 0
        err_count = 0
        dest_path.mkdir(parents=True, exist_ok=True) # Ensure destination exists
        for file_path in file_list:
            try:
                if file_path.exists(): # Check if source exists
                     dest_file = dest_path / file_path.name
                     # Change move to move instead of copy2 to remove source files
                     shutil.move(str(file_path), str(dest_file))
                     count += 1
                else:
                    logger.warning(f"Source file {file_path} not found for moving. Check if input dir overlaps with cleared output.") # Changed wording
                    err_count += 1
            except Exception as e:
                logger.error(f"Error moving file {file_path.name} to {dest_path}: {e}") # Changed wording
                err_count += 1
        return count, err_count

    logger.info(f"Copying {n_train} files to {train_path}...") # Changed wording
    moved, errors = move_files(final_train_files, train_path)
    moved_train += moved
    errors_moving += errors

    logger.info(f"Copying {n_val} files to {val_path}...") # Changed wording
    moved, errors = move_files(validation_files, val_path)
    moved_val += moved
    errors_moving += errors

    logger.info(f"Copying {n_test} files to {test_path}...") # Changed wording
    moved, errors = move_files(test_files, test_path)
    moved_test += moved
    errors_moving += errors

    logger.info(
        f"File copying complete. Copied {moved_train}/{n_train} to Train, {moved_val}/{n_val} to Validation, {moved_test}/{n_test} to Test." # Changed wording
    )
    if errors_moving > 0:
        logger.error(f"{errors_moving} errors occurred during file copying.") # Changed wording


def run_split(
    extracted_dir: str,
    split_output_base_dir: str,
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
    test_months: int,  # Changed from test_ratio to test_months
    validation_size: int,
    seed: int,
    clear_output: bool,
):
    """
    Finds extracted files and runs the train/val/test split.
    """
    extracted_path = Path(extracted_dir)
    output_base_path = Path(split_output_base_dir)

    if not extracted_path.exists() or not extracted_path.is_dir():
        logger.error(f"Extracted data directory not found: {extracted_path}")
        return

    # --- File Discovery ---
    # Find all extracted CSV files (assuming they are directly in extracted_path)
    all_extracted_files = sorted(list(extracted_path.glob('*.csv')))

    if not all_extracted_files:
        logger.warning(f"No *.csv files found in {extracted_path} to split. Did the extraction step run successfully?")
        return

    logger.info(f"Found {len(all_extracted_files)} extracted files to split.")

    # Check for overlap between input and output directories if clearing
    if clear_output and output_base_path.resolve() == extracted_path.resolve():
        logger.error(f"Input directory ({extracted_path}) and output directory ({output_base_path}) are the same. Cannot clear output without deleting input files before splitting.")
        logger.error("Please specify a different split_output_base_dir or set clear_output to false.")
        return

    # --- Perform Split --- 
    perform_train_val_test_split(
        all_files=all_extracted_files,
        output_base_dir=output_base_path,
        train_subdir=train_subdir,
        val_subdir=val_subdir,
        test_subdir=test_subdir,
        test_months=test_months,
        validation_size=validation_size,
        seed=seed,
        clear_output=clear_output,
    )

    logger.info(f"Splitting process complete. Output saved under: {output_base_path}")


if __name__ == "__main__":
    # --- Logging Setup ---
    log_file = Path("logs") / "split_data.log"
    setup_logging(
        log_file_path=log_file,
        root_level=logging.INFO,
        level_overrides={
            "SplitData": logging.INFO,
            "DataManager": logging.INFO
        }
    )
    # ---------------------

    # --- Configuration Loading ---
    base_dir = Path(__file__).resolve().parent.parent.parent  # Go up one more level to reach project root
    config_path = base_dir / "config" / "split_config.yaml"  # Config file name
    config = {}
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                 config = yaml.safe_load(f)
            if config is None: config = {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/split_config.yaml'.")
            # Example config structure:
            logger.info("Example `split_config.yaml`:\n"
                        "extracted_dir: data/extracted\n"
                        "split_output_base_dir: data/processed # Output train/val/test subdirs here\n"
                        "train_subdir: train\n"
                        "val_subdir: validation\n"
                        "test_subdir: test\n"
                        "test_months: 3\n"
                        "validation_size: 500\n"
                        "seed: 42\n"
                        "clear_output: true # Clears train/val/test subdirs before moving")
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
        extracted_dir_rel = config["extracted_dir"]
        split_output_base_dir_rel = config["split_output_base_dir"]
        train_subdir = config.get("train_subdir", "train")
        val_subdir = config.get("val_subdir", "validation")
        test_subdir = config.get("test_subdir", "test")
        test_months = config["test_months"]
        validation_size = config["validation_size"]
        seed = config.get("seed", 42)
        clear_output = config.get("clear_output", True)
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ---------------------------

    # --- Validation ---
    if not isinstance(test_months, int) or test_months <= 0:
        logger.error(f"test_months ({test_months}) must be a positive integer.")
        sys.exit(1)
    if not isinstance(validation_size, int) or validation_size <= 0:
        logger.error(f"validation_size ({validation_size}) must be a positive integer.")
        sys.exit(1)
    # ----------------

    # --- Path Construction ---
    extracted_dir = base_dir / extracted_dir_rel
    split_output_base_dir = base_dir / split_output_base_dir_rel
    # -------------------------

    logger.info(
        f"Starting data splitting. Extracted dir: {extracted_dir}, Output base dir: {split_output_base_dir}"
    )

    run_split(
        extracted_dir=str(extracted_dir),
        split_output_base_dir=str(split_output_base_dir),
        train_subdir=train_subdir,
        val_subdir=val_subdir,
        test_subdir=test_subdir,
        test_months=test_months,
        validation_size=validation_size,
        seed=seed,
        clear_output=clear_output,
    )

    logger.info("Splitting script finished.") 