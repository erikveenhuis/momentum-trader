#!/usr/bin/env python3
"""Install Momentum Trader packages in editable mode."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"[install] {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[ok] {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[error] {description}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return False


def install_package(package_path, package_name):
    """Install a single package in editable mode."""
    if not package_path.exists():
        print(f"[error] Package directory not found: {package_path}")
        return False

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(package_path)]
    return run_command(cmd, f"Installing {package_name}")


def verify_installation():
    """Verify that all packages can be imported."""
    print("[verify] Checking package imports...")

    packages_to_check = [
        "momentum_core",
        "momentum_env",
        "momentum_agent",
        "momentum_train",
        "momentum_live",
    ]

    failed = []
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"  [ok] {package}")
        except ImportError as e:
            print(f"  [error] {package}: {e}")
            failed.append(package)

    if failed:
        print(f"\n[error] Failed to import: {', '.join(failed)}")
        return False

    print("[ok] All packages imported successfully")
    return True


def main():
    print("Momentum Trader Package Installer")
    print("=" * 40)

    if not hasattr(sys, "real_prefix") and not (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print("[warn] Virtual environment not detected. Run: source venv/bin/activate")
        print()

    project_root = Path(__file__).parent

    packages = [
        ("momentum_core", project_root / "packages" / "momentum_core"),
        ("momentum_env", project_root / "packages" / "momentum_env"),
        ("momentum_agent", project_root / "packages" / "momentum_agent"),
        ("momentum_train", project_root / "packages" / "momentum_train"),
        ("momentum_live", project_root / "packages" / "momentum_live"),
    ]

    success = True
    for package_name, package_path in packages:
        if not install_package(package_path, package_name):
            success = False
            break

    if success and verify_installation():
        print("\nInstallation complete.")
        return 0
    else:
        print("\nInstallation failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
