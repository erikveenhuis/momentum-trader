import pytest

# import tempfile <-- Remove

# Remove sys.path manipulation
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
# if src_path not in sys.path:
#     sys.path.insert(0, src_path)

# Direct import from package
from momentum_train.data import DataManager


# --- Fixture to create dummy data structure ---
@pytest.fixture
def dummy_data_structure(tmp_path):
    """Creates a temporary directory structure mimicking the expected data layout."""
    base_dir = tmp_path
    processed_dir = base_dir / "data" / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Define minimal valid CSV content
    csv_header = "timestamp,open,high,low,close,volume\n"
    csv_row = "1678886400,1700.0,1710.0,1690.0,1705.0,1000\n"
    csv_content = csv_header + csv_row

    # Create dummy files
    (train_dir / "train1.csv").write_text(csv_content)
    (train_dir / "train2.csv").write_text(csv_content)
    (val_dir / "val1.csv").write_text(csv_content)
    (test_dir / "test1.csv").write_text(csv_content)

    return base_dir  # Return the path to the base directory


# --- Test Cases ---


@pytest.mark.unittest
def test_data_manager_init_success(dummy_data_structure):
    """Test successful initialization when data directories exist."""
    base_dir = dummy_data_structure
    # print(f"[Test Init Success] Using base_dir: {base_dir}")
    try:
        manager = DataManager(base_dir=str(base_dir))
        assert manager.base_dir == base_dir
        assert manager.processed_dir.exists()
        assert manager.train_dir.exists()
        assert manager.val_dir.exists()
        assert manager.test_dir.exists()
        # print(f"[Test Init Success] Manager initialized successfully")
    except Exception as e:
        # print(f"[Test Init Success] Exception during init: {e}")
        pytest.fail(f"DataManager initialization failed unexpectedly: {e}")


@pytest.mark.unittest
def test_data_manager_init_missing_processed(tmp_path, caplog):
    """Test initialization raises FileNotFoundError if the processed directory is missing."""
    base_dir = tmp_path
    # Don't create data/processed
    # (base_dir / "data").mkdir() # Don't even create base/data
    # print(f"[Test Missing Processed] Using base_dir: {base_dir}")

    # Check that FileNotFoundError is raised
    with pytest.raises(
        FileNotFoundError, match="Processed data directory .* not found"
    ):
        _ = DataManager(base_dir=str(base_dir))  # Initialize with the temp base path

    # Remove caplog check as we expect an exception now
    # caplog.set_level(logging.ERROR)
    # _ = DataManager(base_dir=str(base_dir)) # Initialize with the temp base path
    # assert "Processed data directory 'processed' not found" in caplog.text


@pytest.mark.unittest
def test_data_manager_organize_data_success(dummy_data_structure):
    """Test that organize_data correctly finds the files."""
    base_dir = dummy_data_structure
    manager = DataManager(base_dir=str(base_dir))
    try:
        manager.organize_data()
        assert len(manager.train_files) == 2
        assert len(manager.val_files) == 1
        assert len(manager.test_files) == 1  # Test dir was not empty
        assert manager._data_organized is True
        assert manager.get_training_files()[0].name == "train1.csv"  # Check sorting
        assert manager.get_training_files()[1].name == "train2.csv"
        assert manager.get_validation_files()[0].name == "val1.csv"
        assert manager.get_test_files()[0].name == "test1.csv"
        # print(f"[Test Organize Success] Data organized correctly")
    except Exception as e:
        # print(f"[Test Organize Success] Exception during organize: {e}")
        pytest.fail(f"organize_data failed unexpectedly: {e}")


@pytest.mark.unittest
def test_data_manager_organize_data_missing_train_dir(dummy_data_structure):
    """Test organize_data fails if the train directory is missing."""
    base_dir = dummy_data_structure
    # Remove the train directory created by the fixture
    train_dir = base_dir / "data" / "processed" / "train"
    if train_dir.exists():
        import shutil

        shutil.rmtree(train_dir)
    # print(f"[Test Missing Train Dir] Removed train_dir: {train_dir}")

    manager = DataManager(base_dir=str(base_dir))
    with pytest.raises(AssertionError, match="Train directory not found"):
        manager.organize_data()
    # print(f"[Test Missing Train Dir] Correctly raised AssertionError")


@pytest.mark.unittest
def test_data_manager_get_random_training_file(dummy_data_structure):
    """Test retrieving a random training file."""
    base_dir = dummy_data_structure
    manager = DataManager(base_dir=str(base_dir))
    manager.organize_data()  # Load the files first

    try:
        random_file = manager.get_random_training_file()
        assert random_file.name in ["train1.csv", "train2.csv"]
        assert random_file.exists()
        # print(f"[Test Random File] Got random file: {random_file.name}")
    except Exception as e:
        # print(f"[Test Random File] Exception getting random file: {e}")
        pytest.fail(f"get_random_training_file failed unexpectedly: {e}")
