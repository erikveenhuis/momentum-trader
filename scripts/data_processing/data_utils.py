import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

EXPECTED_DAILY_ENTRIES: int = 1440


def clear_directory(directory_path: Path):
    """Remove and recreate a directory so it ends up empty."""
    if directory_path.exists():
        logger.info(f"Clearing directory: {directory_path}")
        try:
            shutil.rmtree(directory_path)
            logger.info(f"Directory cleared: {directory_path}")
        except OSError as e:
            logger.error(f"Error clearing directory {directory_path}: {e}")
            try:
                for item in directory_path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                logger.info(f"Cleared contents of directory: {directory_path}")
            except Exception as inner_e:
                logger.error(f"Could not clear contents of directory {directory_path}: {inner_e}")
    else:
        logger.info(f"Directory not found, will create: {directory_path}")

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory_path}")
    except Exception as e:
        logger.error(f"Could not create directory {directory_path}: {e}")
