# Momentum Core

Core utilities and shared components for the momentum trader project.

## Features

- **Repository-wide logging**: Consistent logging configuration across all packages
  - Console logging
  - File logging with rotation (10 files × 1MB each)
  - Support for multiple log levels
  - Easy-to-use API for all packages
  - Environment variable overrides for per-package log levels (`MOMENTUM_LOG_LEVEL_*`)

## Usage

### Basic Logging Setup

```python
from momentum_core.logging import setup_logging, get_logger

# Setup logging (console only)
setup_logging()

# Setup logging with file rotation
setup_logging(log_file_path="logs/my_app.log")

# Get a logger for your module
logger = get_logger(__name__)
logger.info("This is an info message")
```

### Package-Specific Logging

```python
from momentum_core.logging import setup_package_logging

# Setup logging for a specific package
setup_package_logging("my_package")

# Environment overrides:
#   MOMENTUM_LOG_DIR=/var/log/app
#   MOMENTUM_LOG_LEVEL=WARNING
#   MOMENTUM_LOG_LEVEL_MY_PACKAGE=DEBUG
```

### Advanced Configuration

```python
from momentum_core.logging import setup_logging
import logging

# Advanced setup with custom levels and overrides
setup_logging(
    log_file_path="logs/app.log",
    root_level=logging.DEBUG,
    level_overrides={
        "my_module.debug_only": logging.DEBUG,
        "my_module.quiet": logging.WARNING,
    },
    max_bytes=2*1024*1024,  # 2MB per file
    backup_count=5,  # 5 backup files
)
```
