"""Shared data management for training, validation, and test file discovery."""

import random
from pathlib import Path

from .logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """Manages data loading and organization for training, validation, and test sets."""

    def __init__(self, base_dir: str = "data", processed_dir_name: str = "processed"):
        """Initializes the DataManager.

        Args:
            base_dir: The base directory containing the data structure.
            processed_dir_name: Subdirectory name for processed data (relative to base_dir).
        """
        self.base_dir = Path(base_dir)

        processed_path_direct = self.base_dir / processed_dir_name
        processed_path_nested = self.base_dir / "data" / processed_dir_name

        if processed_path_direct.is_dir():
            self.processed_dir = processed_path_direct
            logger.info(f"Found processed data directly under base: {self.processed_dir}")
        elif processed_path_nested.is_dir():
            self.processed_dir = processed_path_nested
            logger.info(f"Found processed data nested under base/data: {self.processed_dir}")
        else:
            err_msg = (
                f"Processed data directory '{processed_dir_name}' not found directly under "
                f"'{self.base_dir}' or under '{self.base_dir / 'data'}'. "
                f"Checked: {processed_path_direct}, {processed_path_nested}"
            )
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        self.train_dir = self.processed_dir / "train"
        self.val_dir = self.processed_dir / "validation"
        self.test_dir = self.processed_dir / "test"

        logger.info(f"DataManager initialized with base directory: {self.base_dir.resolve()}")
        logger.info(f"  Processed data directory: {self.processed_dir.resolve()}")
        logger.info(f"  Train directory: {self.train_dir.resolve()}")
        logger.info(f"  Validation directory: {self.val_dir.resolve()}")
        logger.info(f"  Test directory: {self.test_dir.resolve()}")

        self._data_organized = False
        self.train_files: list[Path] = []

    def organize_data(self):
        """Load file lists from pre-split train/validation/test directories."""
        if self._data_organized:
            return

        logger.info("Loading file lists from pre-split directories...")

        assert self.train_dir.is_dir(), f"Train directory not found or is not a directory: {self.train_dir}"
        assert self.val_dir.is_dir(), f"Validation directory not found or is not a directory: {self.val_dir}"
        assert self.test_dir.is_dir(), f"Test directory not found or is not a directory: {self.test_dir}"

        try:
            self.train_files = sorted(list(self.train_dir.glob("*.npz")))
            self.val_files = sorted(list(self.val_dir.glob("*.npz")))
            self.test_files = sorted(list(self.test_dir.glob("*.npz")))

            if not self.train_files:
                self.train_files = sorted(list(self.train_dir.glob("*.csv")))
                self.val_files = sorted(list(self.val_dir.glob("*.csv")))
                self.test_files = sorted(list(self.test_dir.glob("*.csv")))
                if self.train_files:
                    logger.info("No .npz files found, falling back to CSV files.")

            assert len(self.train_files) > 0, f"No training data files found in {self.train_dir}"
            assert len(self.val_files) > 0, f"No validation data files found in {self.val_dir}"
            if len(self.test_files) == 0:
                logger.warning(f"No test data files found in {self.test_dir}")

            logger.info(f"Found {len(self.train_files)} training files.")
            logger.info(f"Found {len(self.val_files)} validation files.")
            logger.info(f"Found {len(self.test_files)} test files.")

            self._data_organized = True

        except Exception as e:
            logger.error(f"Error organizing data from subdirectories: {e}", exc_info=True)
            self.train_files, self.val_files, self.test_files = [], [], []
            self._data_organized = False
            raise

    def _ensure_organized(self):
        """Internal helper to ensure data is organized before accessing files."""
        if not self._data_organized:
            self.organize_data()
        assert self._data_organized, "Data organization failed, cannot retrieve files."

    def get_training_files(self) -> list[Path]:
        """Get training files."""
        self._ensure_organized()
        assert len(self.train_files) > 0, "No training files loaded."
        return self.train_files

    def get_validation_files(self) -> list[Path]:
        """Get validation files."""
        self._ensure_organized()
        assert len(self.val_files) > 0, "No validation files loaded."
        return self.val_files

    def get_test_files(self) -> list[Path]:
        """Get test files."""
        self._ensure_organized()
        return self.test_files

    def get_random_training_file(self, curriculum_frac: float = 1.0) -> Path:
        """Get a random training file, optionally restricted by curriculum fraction.

        Args:
            curriculum_frac: Fraction of training files to sample from (0..1).
                The files are sorted by name (chronologically), so a fraction < 1
                restricts to earlier (typically lower-volatility) data.
        """
        self._ensure_organized()
        num_files = len(self.train_files)
        assert num_files > 0, "Cannot get random file: No training files available."

        pool_size = max(1, int(num_files * min(max(curriculum_frac, 0.0), 1.0)))
        pool = self.train_files[:pool_size]
        random_file = random.choice(pool)
        logger.debug(f"[DataManager] Selected file: {random_file.name} (pool={pool_size}/{num_files})")
        assert random_file.exists(), f"Chosen random training file does not exist: {random_file}"
        return random_file
