# tests/test_trainer_train_loop.py
from pathlib import Path


# Helper for creating dummy CSV files in tests that need them
def create_dummy_csv(filepath: Path, rows: int = 20):
    header = "timestamp,open,high,low,close,volume\n"
    rows_data = [f"2023-01-01T00:{i:02d}:00Z,100,101,99,100,1000\n" for i in range(rows)]
    csv_content = header + "".join(rows_data)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(csv_content)


# Note: Fixtures (trainer, mock_agent, mock_data_manager, etc.) are in conftest.py

# --- Mocking removed, tests below are removed --- #
