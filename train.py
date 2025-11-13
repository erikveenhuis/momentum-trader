#!/usr/bin/env python3
"""
Momentum Trader Training Entrypoint

This script provides an interactive interface for training the momentum trader model.
It automatically starts TensorBoard for monitoring and prompts the user to choose
between resuming training or starting fresh.
"""

import subprocess
import sys
import time
from pathlib import Path


def start_tensorboard():
    """Start TensorBoard in the background."""
    runs_dir = Path("models/runs")
    if not runs_dir.exists():
        print(f"⚠️  Warning: TensorBoard logs directory '{runs_dir}' not found.")
        print("   TensorBoard will start but may not have logs to display yet.")
        runs_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 Starting TensorBoard...")
    print("   📊 Dashboard will be available at: http://localhost:6006")
    print("   📁 Monitoring logs in: models/runs")
    print()

    # Start TensorBoard in background
    try:
        tensorboard_cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir=models/runs",
            "--port=6006",
            "--host=localhost"
        ]
        tensorboard_process = subprocess.Popen(
            tensorboard_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✅ TensorBoard started successfully")
        return tensorboard_process
    except Exception as e:
        print(f"❌ Failed to start TensorBoard: {e}")
        print("   You can start it manually later with:")
        print("   tensorboard --logdir=models/runs --port=6006")
        return None


def get_training_choice():
    """Prompt user for training choice with resume as default."""
    print("🎯 Training Options:")
    print("   1. Resume training from checkpoint (default)")
    print("   2. Start new training")
    print()

    # Check if running in interactive mode
    import sys
    if not sys.stdin.isatty():
        print("🤖 Non-interactive mode detected, using default: Resume training")
        return True  # resume = True (default)

    while True:
        try:
            choice = input("Enter your choice (1 or 2) [default: 1]: ").strip()

            if choice == "" or choice == "1":
                print("▶️  Resuming training from checkpoint...")
                return True  # resume = True
            elif choice == "2":
                print("🆕 Starting new training session...")
                return False  # resume = False
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\n🤖 Using default: Resume training")
            return True  # resume = True (default)


def run_training(resume_training):
    """Run the training script with appropriate flags."""
    cmd = [
        sys.executable, "-m", "momentum_train.run_training",
        "--config_path", "config/training_config.yaml"
    ]

    if resume_training:
        cmd.append("--resume")

    print(f"🏃 Executing: {' '.join(cmd)}")
    print()

    try:
        # Run training (this will block until training completes)
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        if "ModuleNotFoundError: No module named 'momentum_train'" in error_msg:
            print("❌ Training failed: momentum_train package not installed!")
            print()
            print("🔧 To fix this issue:")
            print("   1. Install the required packages: python install_packages.py")
            print("   2. Then run training again: python train.py")
            print()
        else:
            print(f"❌ Training failed with exit code: {e.returncode}")
            if error_msg.strip():
                print(f"   Error details: {error_msg}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False


def check_package_imports():
    """Check if required packages can be imported."""
    required_packages = [
        "momentum_train",
        "momentum_core",
        "momentum_env",
        "momentum_agent"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Required packages not installed:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("🔧 To fix this issue, run:")
        print("   python install_packages.py")
        print()
        return False

    return True


def main():
    """Main entrypoint function."""
    print("=" * 60)
    print("🤖 MOMENTUM TRADER TRAINING ENTRYPOINT")
    print("=" * 60)
    print()

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected!")
        print("   Make sure to activate your venv before running this script.")
        print("   Run: source venv/bin/activate")
        print()

    # Check if required packages are installed
    if not check_package_imports():
        sys.exit(1)

    # Check if config file exists
    config_path = Path("config/training_config.yaml")
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        print("   Please ensure the config file exists.")
        sys.exit(1)

    # Start TensorBoard
    tensorboard_process = start_tensorboard()

    # Give TensorBoard a moment to start
    time.sleep(2)

    # Get user's training choice
    resume_training = get_training_choice()
    print()

    # Run training
    success = run_training(resume_training)

    # Cleanup TensorBoard if it's still running
    if tensorboard_process and tensorboard_process.poll() is None:
        print("🛑 Stopping TensorBoard...")
        tensorboard_process.terminate()
        try:
            tensorboard_process.wait(timeout=5)
            print("✅ TensorBoard stopped")
        except subprocess.TimeoutExpired:
            tensorboard_process.kill()
            print("⚠️  TensorBoard force-killed")

    print()
    if success:
        print("🎉 Session completed successfully!")
    else:
        print("💥 Session ended with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
