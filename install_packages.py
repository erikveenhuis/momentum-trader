#!/usr/bin/env python3
"""
Install Momentum Trader packages in editable mode.

This script installs all the required packages for the Momentum Trader project
in development mode, allowing for live editing of the source code.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Error: {e.stderr}")
        return False


def install_package(package_path, package_name):
    """Install a single package in editable mode."""
    if not package_path.exists():
        print(f"❌ Package directory not found: {package_path}")
        return False

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(package_path)]
    return run_command(cmd, f"Installing {package_name}")


def verify_installation():
    """Verify that all packages can be imported."""
    print("🔍 Verifying package installation...")

    packages_to_check = [
        "momentum_core",
        "momentum_env",
        "momentum_agent",
        "momentum_train",
        "momentum_live"
    ]

    failed_imports = []

    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} package(s): {', '.join(failed_imports)}")
        return False

    print("✅ All packages imported successfully!")
    return True


def main():
    """Main installation function."""
    print("=" * 60)
    print("📦 MOMENTUM TRADER PACKAGE INSTALLER")
    print("=" * 60)
    print()

    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected!")
        print("   Make sure to activate your venv before running this script.")
        print("   Run: source venv/bin/activate")
        print()

    project_root = Path(__file__).parent

    packages = [
        ("momentum_core", project_root / "packages" / "momentum_core"),
        ("momentum_env", project_root / "packages" / "momentum_env"),
        ("momentum_agent", project_root / "packages" / "momentum_agent"),
        ("momentum_train", project_root / "packages" / "momentum_train"),
        ("momentum_live", project_root / "packages" / "momentum_live"),
    ]

    print("🚀 Installing packages in editable mode...")
    print()

    success = True
    for package_name, package_path in packages:
        if not install_package(package_path, package_name):
            success = False
            break

    if success:
        print()
        if verify_installation():
            print()
            print("🎉 Installation completed successfully!")
            print("   You can now run training with: python train.py")
            return 0
        else:
            print()
            print("❌ Installation verification failed!")
            print("   Try running this script again or check the error messages above.")
            return 1
    else:
        print()
        print("❌ Installation failed!")
        print("   Check the error messages above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
