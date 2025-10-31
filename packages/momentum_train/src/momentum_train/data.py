from pathlib import Path
from typing import List
import logging
import random
from .utils.logging_config import get_logger

# Get logger instance
logger = get_logger("DataManager")


class DataManager:
    """Manages data loading and organization for training, validation, and test sets."""

    def __init__(self, base_dir: str = "data", processed_dir_name: str = "processed"):
        """Initializes the DataManager.

        Args:
            base_dir (str): The base directory containing the data structure.
                            Defaults to "data". Can be overridden (e.g., for tests).
            processed_dir_name (str): Subdirectory name for processed data (relative to base_dir).
                                      Defaults to "processed".
        """
        self.base_dir = Path(base_dir)

        # Check for processed data location
        processed_path_direct = self.base_dir / processed_dir_name
        processed_path_nested = self.base_dir / "data" / processed_dir_name

        if processed_path_direct.is_dir():
            self.processed_dir = processed_path_direct
            logger.info(
                f"Found processed data directly under base: {self.processed_dir}"
            )
        elif processed_path_nested.is_dir():
            self.processed_dir = processed_path_nested
            logger.info(
                f"Found processed data nested under base/data: {self.processed_dir}"
            )
        else:
            # Raise error if neither is found
            err_msg = (
                f"Processed data directory '{processed_dir_name}' not found directly under "
                f"'{self.base_dir}' or under '{self.base_dir / 'data'}'. "
                f"Checked: {processed_path_direct}, {processed_path_nested}"
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        # No longer explicitly storing raw_dir unless needed later
        # self.processed_dir = self.base_dir / processed_dir_name <-- Old logic removed

        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "validation"
        self.test_dir = self.processed_dir / "test"

        logger.info(
            f"DataManager initialized with base directory: {self.base_dir.resolve()}"
        )
        logger.info(f"  Processed data directory: {self.processed_dir.resolve()}")
        logger.info(f"  Train directory: {self.train_dir.resolve()}")
        logger.info(f"  Validation directory: {self.val_dir.resolve()}")
        logger.info(f"  Test directory: {self.test_dir.resolve()}")

        self._data_organized = False
        self.train_files: List[Path] = []

    # Removed get_file_date method as sorting/splitting is external
    # Removed split_data_by_time method as splitting is external

    def organize_data(self):
        """Load file lists from pre-split train/validation/test directories."""
        if self._data_organized:
            return  # Avoid re-organizing

        logger.info("Loading file lists from pre-split directories...")

        # Check if directories exist
        assert (
            self.train_dir.is_dir()
        ), f"Train directory not found or is not a directory: {self.train_dir}"
        assert (
            self.val_dir.is_dir()
        ), f"Validation directory not found or is not a directory: {self.val_dir}"
        assert (
            self.test_dir.is_dir()
        ), f"Test directory not found or is not a directory: {self.test_dir}"

        try:
            # Load and sort files from each directory by filename (assumes YYYY-MM-DD format)
            self.train_files = sorted(list(self.train_dir.glob("*.csv")))
            self.val_files = sorted(list(self.val_dir.glob("*.csv")))
            self.test_files = sorted(list(self.test_dir.glob("*.csv")))

            # Assert that files were actually found if the directories exist
            assert (
                len(self.train_files) > 0
            ), f"No training CSV files found in {self.train_dir}"
            assert (
                len(self.val_files) > 0
            ), f"No validation CSV files found in {self.val_dir}"
            # Test files might be optional, so only warn if missing
            if len(self.test_files) == 0:
                logger.warning(f"No test CSV files found in {self.test_dir}")

            logger.info(f"Found {len(self.train_files)} training files.")
            logger.info(f"Found {len(self.val_files)} validation files.")
            logger.info(f"Found {len(self.test_files)} test files.")

            self._data_organized = True  # Mark as organized

        except Exception as e:
            logger.error(
                f"Error organizing data from subdirectories: {e}", exc_info=True
            )
            self.train_files, self.val_files, self.test_files = [], [], []
            self._data_organized = False
            raise  # Re-raise the exception

    def _ensure_organized(self):
        """Internal helper to ensure data is organized before accessing files."""
        if not self._data_organized:
            self.organize_data()
        assert self._data_organized, "Data organization failed, cannot retrieve files."

    def get_training_files(self) -> List[Path]:
        """Get training files."""
        self._ensure_organized()
        assert len(self.train_files) > 0, "No training files loaded."
        return self.train_files

    def get_validation_files(self) -> List[Path]:
        """Get validation files."""
        self._ensure_organized()
        assert len(self.val_files) > 0, "No validation files loaded."
        return self.val_files

    def get_test_files(self) -> List[Path]:
        """Get test files."""
        self._ensure_organized()
        # No assertion here, test files might be empty
        return self.test_files

    def get_random_training_file(self) -> Path:
        """Get a random training file."""
        self._ensure_organized()
        num_files = len(self.train_files)
        logger.debug(f"[DataManager] Choosing random file from {num_files} training files.")
        assert (
            num_files > 0
        ), "Cannot get random file: No training files available."
        random_file = random.choice(self.train_files)
        logger.debug(f"[DataManager] Selected file: {random_file.name}")
        assert (
            random_file.exists()
        ), f"Chosen random training file does not exist: {random_file}"
        return random_file
