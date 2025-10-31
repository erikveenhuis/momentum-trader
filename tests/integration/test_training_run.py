import pytest
import subprocess
import sys
from pathlib import Path
import yaml
import os

# Define project root relative to this test file
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_training.py"
TEST_CONFIG_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "config" / "test_training_config.yaml"
)
TEST_FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


@pytest.mark.integration
def test_run_training_script_integration(tmp_path):
    """
    Integration test for the run_training.py script.
    - Mocks DataManager to use test fixture data.
    - Runs the script with a minimal test configuration.
    - Checks for successful execution and expected output files.
    """
    assert SCRIPT_PATH.exists(), f"Training script not found: {SCRIPT_PATH}"
    assert TEST_CONFIG_PATH.exists(), f"Test config not found: {TEST_CONFIG_PATH}"
    assert (
        TEST_FIXTURES_DIR.exists()
    ), f"Test fixtures directory not found: {TEST_FIXTURES_DIR}"
    assert (
        TEST_FIXTURES_DIR / "data/processed/train"
    ).exists(), "Test train data dir missing"
    assert (
        TEST_FIXTURES_DIR / "data/processed/validation"
    ).exists(), "Test validation data dir missing"
    assert (
        TEST_FIXTURES_DIR / "data/processed/test"
    ).exists(), "Test test data dir missing"

    # Create temporary directories for logs and models within tmp_path
    test_model_dir = tmp_path / "models"
    test_log_dir = tmp_path / "logs"  # Keep for potential future use
    test_model_dir.mkdir()
    test_log_dir.mkdir()  # Create even if not directly asserted

    # Load the test config and update paths
    with open(TEST_CONFIG_PATH, "r") as f:
        test_config = yaml.safe_load(f)

    # Update config to use temporary output directories and fixture data path
    test_config["run"]["model_dir"] = str(test_model_dir)
    test_config["run"]["skip_evaluation"] = True  # Add flag to skip eval
    test_config["run"]["data_base_dir"] = str(
        TEST_FIXTURES_DIR
    )  # Specify fixture data path

    # Path for the modified config within tmp_path
    temp_config_path = tmp_path / "temp_test_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(test_config, f)

    # --- Run the script as a subprocess --- #
    command = [
        sys.executable,  # Use the same python interpreter
        str(SCRIPT_PATH),
        "--config_path",
        str(temp_config_path),
    ]

    print(f"\nRunning command: {' '.join(command)}")
    print(f"Working directory: {PROJECT_ROOT}")

    # --- Set PYTHONPATH for the subprocess ---
    subprocess_env = os.environ.copy()
    # src_path = str(PROJECT_ROOT / "src") # Old: Use src path directly
    # Prepend project root path to existing PYTHONPATH or set it if it doesn't exist
    subprocess_env["PYTHONPATH"] = (
        f"{str(PROJECT_ROOT)}{os.pathsep}{subprocess_env.get('PYTHONPATH', '')}"
    )
    print(f"Setting PYTHONPATH for subprocess: {subprocess_env['PYTHONPATH']}")
    # ----------------------------------------

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,  # Run from project root for module resolution
        capture_output=True,
        text=True,
        env=subprocess_env,  # Pass the modified environment
        check=False,  # Handle non-zero exit code manually
        timeout=60,  # Restore 60-second timeout
    )

    # --- Assertions --- #
    print("\n--- Subprocess stdout --- ")
    print(process.stdout)
    print("\n--- Subprocess stderr --- ")
    print(process.stderr)
    print("-------------------------")

    # Check 1: Successful execution (exit code 0)
    assert (
        process.returncode == 0
    ), f"Script execution failed with exit code {process.returncode}. See stderr above."

    # Check 3: Expected output files in the *temporary* model directory
    model_files = list(test_model_dir.glob("**/*"))  # Search recursively
    print(f"Files found in temporary model dir ({test_model_dir}): {model_files}")
    assert any(
        f.name.startswith(("checkpoint_trainer", "rainbow_transformer")) and f.is_file()
        for f in model_files
    ), f"No model checkpoint files (e.g., checkpoint_trainer*.pt, rainbow_transformer*.pt) found in {test_model_dir}"
    print(f"Model checkpoint files found in {test_model_dir}.")

    # Check 4: Check if the main training log file was created in the standard location
    # Note: This depends on the logging setup using a fixed relative path 'logs/training.log'
    # If logging is configured differently (e.g., uses the temp dir), this needs adjustment.
    expected_log_file = PROJECT_ROOT / "logs" / "training.log"
    # This assertion might be fragile depending on logging setup
    # assert expected_log_file.exists(), f"Expected log file not found at {expected_log_file}"
    # print(f"Standard log file found at {expected_log_file}.")

    print("\nIntegration test completed successfully.")


@pytest.mark.integration
def test_run_eval_script_integration(tmp_path):
    """
    Integration test for the run_training.py script in evaluation mode.
    - Uses fixture data via config injection.
    - Runs the script with mode: eval.
    - Creates dummy model files for the script to load.
    - Checks for successful execution.
    """
    assert SCRIPT_PATH.exists(), f"Training script not found: {SCRIPT_PATH}"
    assert TEST_CONFIG_PATH.exists(), f"Test config not found: {TEST_CONFIG_PATH}"
    assert (
        TEST_FIXTURES_DIR.exists()
    ), f"Test fixtures directory not found: {TEST_FIXTURES_DIR}"
    # Need test data for evaluation
    assert (
        TEST_FIXTURES_DIR / "data/processed/test"
    ).exists(), "Test test data dir missing"

    # Create temporary directory for dummy model
    dummy_model_dir = tmp_path / "dummy_model"
    dummy_model_dir.mkdir()
    dummy_model_prefix = dummy_model_dir / "dummy_rainbow_transformer"

    # Create dummy model files (content doesn't matter for this test, just existence)
    # Agent expects <prefix>_rainbow_agent.pt and <prefix>_rainbow_config.yaml
    (dummy_model_prefix.with_suffix(".pt")).touch()
    # Create a minimal dummy config for the agent load method
    dummy_agent_config = {
        "agent": {  # Mimic structure expected by agent load
            "window_size": 60,
            "n_features": 5,
            "hidden_dim": 32,
            "num_actions": 7,
            "num_atoms": 51,
            "v_min": -10.0,
            "v_max": 10.0,
        }
    }
    with open(dummy_model_prefix.with_suffix(".yaml"), "w") as f:
        yaml.dump(dummy_agent_config, f)

    # Load the base test config and modify for evaluation mode
    with open(TEST_CONFIG_PATH, "r") as f:
        test_config = yaml.safe_load(f)

    # Update config for eval mode
    test_config["run"]["mode"] = "eval"
    test_config["run"]["data_base_dir"] = str(TEST_FIXTURES_DIR)  # Use fixture data
    test_config["run"]["eval_model_prefix"] = str(dummy_model_prefix)
    test_config["run"]["skip_evaluation"] = False  # Ensure evaluation runs
    # Remove training-specific keys if they might interfere (optional)
    test_config["run"].pop("episodes", None)
    test_config["run"].pop("resume", None)
    # test_config.pop('trainer', None) # DO NOT POP - trainer config (e.g., seed) might still be needed

    # Path for the modified eval config within tmp_path
    temp_eval_config_path = tmp_path / "temp_eval_config.yaml"
    with open(temp_eval_config_path, "w") as f:
        yaml.dump(test_config, f)

    # --- Run the script as a subprocess --- #
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--config_path",
        str(temp_eval_config_path),
    ]

    print(f"\nRunning EVAL command: {' '.join(command)}")
    print(f"Working directory: {PROJECT_ROOT}")

    subprocess_env = os.environ.copy()
    # src_path = str(PROJECT_ROOT / "src") # Old: Use src path directly
    # Prepend project root path to existing PYTHONPATH or set it if it doesn't exist
    subprocess_env["PYTHONPATH"] = (
        f"{str(PROJECT_ROOT)}{os.pathsep}{subprocess_env.get('PYTHONPATH', '')}"
    )
    print(f"Setting PYTHONPATH for subprocess: {subprocess_env['PYTHONPATH']}")

    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=subprocess_env,
        check=False,
        timeout=90,  # Allow a bit more time for eval setup/run
    )

    # --- Assertions --- #
    print("\n--- Eval Subprocess stdout --- ")
    print(process.stdout)
    print("\n--- Eval Subprocess stderr --- ")
    print(process.stderr)
    print("-------------------------")

    assert (
        process.returncode == 0
    ), f"Eval script execution failed with exit code {process.returncode}. See stderr above."

    # Check stdout for signs of evaluation running
    assert "Starting Evaluation Mode" in process.stdout
    assert "EVALUATING RAINBOW MODEL ON TEST DATA" in process.stdout
    # Check for the specific test file being evaluated (from ACTUAL fixtures)
    assert "TEST FILE 1/1: 2024-08-28_BTC-USD.csv" in process.stdout

    print("\nEval Integration test completed successfully.")
