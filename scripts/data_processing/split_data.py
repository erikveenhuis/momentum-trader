import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import yaml

# Import logging configuration
try:
    from momentum_core.logging import get_logger, setup_package_logging
except ImportError:  # pragma: no cover - fallback when package missing
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.warning("Could not find momentum_core.logging, using basic config.")
    get_logger = logging.getLogger

# Add src directory to Python path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import shared functions/constants - import from same directory
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
try:
    import data_utils

    clear_directory = data_utils.clear_directory
except ImportError:
    logging.error("Could not import utility functions from data_utils. Ensure data_utils.py exists in the same directory.")
    sys.exit(1)

# Get logger instance
logger = get_logger("data_processing.split_data")


def configure_logging(log_level: str | None = None) -> None:
    """Configure logging for the split_data script."""

    if "setup_package_logging" not in globals():
        return

    setup_package_logging(
        "data_processing.split_data",
        log_filename="split_data.log",
        root_level=log_level if log_level is not None else logging.INFO,
        console_level=log_level if log_level is not None else logging.INFO,
        level_overrides={
            "data_processing.split_data": logging.INFO,
            "DataManager": logging.INFO,
        },
    )


def perform_train_val_test_split(
    all_files: List[Path],
    output_base_dir: Path,  # Base directory where subdirs will be created
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
    test_months: int,  # Changed from test_ratio to test_months
    validation_months: int,
    seed: int,
    clear_output: bool,
):
    """
    Splits the processed CSV files into train, validation, and test sets.
    - Test set is the last `test_months` calendar months of data.
    - Validation set is the contiguous block of `validation_months` that
      immediately precedes the test window.
    - Training set receives all data prior to the validation window.
    Copies files into respective subdirectories under output_base_dir.
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
            parts = filename.split("_")
            if len(parts) != 2:
                logger.warning(
                    f"Could not parse date from filename: {file_path.name} - Expected format YYYY-MM-DD_SYMBOL-USD.csv or SYMBOL-USD_YYYY-MM-DD.csv"
                )
                return datetime.min

            # Try both parts to find the date
            for part in parts:
                try:
                    return datetime.strptime(part, "%Y-%m-%d")
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

    if not files_with_dates:
        logger.error("No valid files found after date parsing.")
        return

    invalid_date_files = [f for f, d in files_with_dates if d == datetime.min]
    if invalid_date_files:
        logger.warning(f"Found {len(invalid_date_files)} files with unparseable dates; assigning them to the training set.")

    valid_entries = [(f, d) for f, d in files_with_dates if d != datetime.min]
    if not valid_entries:
        logger.error("No files with valid dates found; cannot perform chronological split.")
        return

    # Build ordered list of distinct (year, month) pairs
    distinct_months: List[Tuple[int, int]] = []
    for _, date in valid_entries:
        month_key = (date.year, date.month)
        if not distinct_months or distinct_months[-1] != month_key:
            distinct_months.append(month_key)

    total_months = len(distinct_months)
    if validation_months <= 0:
        logger.error(f"validation_months ({validation_months}) must be positive.")
        return
    if total_months <= validation_months + test_months:
        logger.error(
            "Not enough historical months to create contiguous train/validation/test windows."
            f" Distinct months: {total_months}, requested validation_months={validation_months}, test_months={test_months}."
        )
        return

    test_month_keys = distinct_months[-test_months:]
    validation_month_keys = distinct_months[-(test_months + validation_months) : -test_months]

    if not validation_month_keys:
        logger.error("Validation window resolved to zero months. Reduce test_months or increase available history.")
        return

    train_month_keys = distinct_months[: -(validation_months + test_months)]

    test_month_set = set(test_month_keys)
    validation_month_set = set(validation_month_keys)

    train_files: List[Path] = list(invalid_date_files)
    validation_files: List[Path] = []
    test_files: List[Path] = []

    for file_path, file_date in valid_entries:
        month_key = (file_date.year, file_date.month)
        if month_key in test_month_set:
            test_files.append(file_path)
        elif month_key in validation_month_set:
            validation_files.append(file_path)
        else:
            train_files.append(file_path)

    n_total = len(all_files)
    n_test = len(test_files)
    n_val = len(validation_files)
    n_train = len(train_files)

    if n_test == 0:
        logger.error(f"Resolved test window (last {test_months} months) produced zero files. Check data coverage or adjust configuration.")
        return
    if n_val == 0:
        logger.error(f"Resolved validation window ({validation_months} months before test window) produced zero files.")
        return
    if n_train == 0:
        logger.error("Training split is empty. Increase available history or shrink validation/test windows.")
        return

    logger.info(
        f"Contiguous split across {total_months} distinct months -> Train: {len(train_month_keys)} months, Validation: {len(validation_month_keys)} months, Test: {len(test_month_keys)} months."
    )
    logger.info(f"Validation window: preceding {validation_months} months (seed {seed} recorded for reproducibility notes).")
    logger.info(f"Final split sizes: Train={n_train}, Validation={n_val}, Test={n_test}")
    if n_train + n_val + n_test != n_total:
        logger.warning(
            f"Consistency check failed: Train({n_train}) + Val({n_val}) + Test({n_test}) = {n_train + n_val + n_test} != Total({n_total})."
        )

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

    copied_train, copied_val, copied_test = 0, 0, 0
    errors_copying = 0

    def copy_files(file_list: List[Path], dest_path: Path) -> Tuple[int, int]:
        count = 0
        err_count = 0
        dest_path.mkdir(parents=True, exist_ok=True)  # Ensure destination exists
        for file_path in file_list:
            try:
                if file_path.exists():  # Check if source exists
                    dest_file = dest_path / file_path.name
                    shutil.copy2(str(file_path), str(dest_file))
                    count += 1
                else:
                    logger.warning(
                        f"Source file {file_path} not found for copying. Check if input dir overlaps with cleared output."
                    )  # Changed wording
                    err_count += 1
            except Exception as e:
                logger.error(f"Error copying file {file_path.name} to {dest_path}: {e}")  # Changed wording
                err_count += 1
        return count, err_count

    logger.info(f"Copying {n_train} files to {train_path}...")  # Changed wording
    moved, errors = copy_files(train_files, train_path)
    copied_train += moved
    errors_copying += errors

    logger.info(f"Copying {n_val} files to {val_path}...")  # Changed wording
    moved, errors = copy_files(validation_files, val_path)
    copied_val += moved
    errors_copying += errors

    logger.info(f"Copying {n_test} files to {test_path}...")  # Changed wording
    moved, errors = copy_files(test_files, test_path)
    copied_test += moved
    errors_copying += errors

    logger.info(
        f"File copying complete. Copied {copied_train}/{n_train} to Train, {copied_val}/{n_val} to Validation, {copied_test}/{n_test} to Test."  # Changed wording
    )
    if errors_copying > 0:
        logger.error(f"{errors_copying} errors occurred during file copying.")  # Changed wording


def run_split(
    extracted_dir: str,
    split_output_base_dir: str,
    train_subdir: str,
    val_subdir: str,
    test_subdir: str,
    test_months: int,  # Changed from test_ratio to test_months
    validation_months: int,
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
    all_extracted_files = sorted(list(extracted_path.glob("*.csv")))

    if not all_extracted_files:
        logger.warning(f"No *.csv files found in {extracted_path} to split. Did the extraction step run successfully?")
        return

    logger.info(f"Found {len(all_extracted_files)} extracted files to split.")

    # Check for overlap between input and output directories if clearing
    if clear_output and output_base_path.resolve() == extracted_path.resolve():
        logger.error(
            f"Input directory ({extracted_path}) and output directory ({output_base_path}) are the same. Cannot clear output without deleting input files before splitting."
        )
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
        validation_months=validation_months,
        seed=seed,
        clear_output=clear_output,
    )

    logger.info(f"Splitting process complete. Output saved under: {output_base_path}")


if __name__ == "__main__":
    # --- Logging Setup ---
    configure_logging()
    # ---------------------

    # --- Configuration Loading ---
    base_dir = Path(__file__).resolve().parent.parent.parent  # Go up one more level to reach project root
    config_path = base_dir / "config" / "split_config.yaml"  # Config file name
    config = {}
    try:
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if config is None:
                config = {}
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.error(f"Configuration file {config_path} not found. Create 'config/split_config.yaml'.")
            # Example config structure:
            logger.info(
                "Example `split_config.yaml`:\n"
                "extracted_dir: data/extracted\n"
                "split_output_base_dir: data/processed # Output train/val/test subdirs here\n"
                "train_subdir: train\n"
                "val_subdir: validation\n"
                "test_subdir: test\n"
                "test_months: 3\n"
                "validation_months: 3\n"
                "seed: 42\n"
                "clear_output: true # Clears train/val/test subdirs before copying"
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
        extracted_dir_rel = config["extracted_dir"]
        split_output_base_dir_rel = config["split_output_base_dir"]
        train_subdir = config["train_subdir"]
        val_subdir = config["val_subdir"]
        test_subdir = config["test_subdir"]
        test_months = config["test_months"]
        validation_months = config["validation_months"]
        seed = config["seed"]
        clear_output = config["clear_output"]
    except KeyError as e:
        logger.error(f"Missing required configuration parameter in {config_path}: {e}")
        sys.exit(1)
    # ---------------------------

    # --- Validation ---
    if not isinstance(test_months, int) or test_months <= 0:
        logger.error(f"test_months ({test_months}) must be a positive integer.")
        sys.exit(1)
    if not isinstance(validation_months, int) or validation_months <= 0:
        logger.error(f"validation_months ({validation_months}) must be a positive integer.")
        sys.exit(1)
    # ----------------

    # --- Path Construction ---
    extracted_dir = base_dir / extracted_dir_rel
    split_output_base_dir = base_dir / split_output_base_dir_rel
    # -------------------------

    logger.info(f"Starting data splitting. Extracted dir: {extracted_dir}, Output base dir: {split_output_base_dir}")

    run_split(
        extracted_dir=str(extracted_dir),
        split_output_base_dir=str(split_output_base_dir),
        train_subdir=train_subdir,
        val_subdir=val_subdir,
        test_subdir=test_subdir,
        test_months=test_months,
        validation_months=validation_months,
        seed=seed,
        clear_output=clear_output,
    )

    logger.info("Splitting script finished.")
