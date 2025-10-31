# Momentum Trader Monorepo

A monorepo containing all components for a reinforcement learning-based trading system using Rainbow DQN.

## 🏗️ Project Structure

This is a workspace-style monorepo organized into independent packages:

```
momentum-trader/
├── packages/
│   ├── momentum_env/       # Trading environment (gymnasium-based)
│   ├── momentum_agent/     # Rainbow DQN agent implementation
│   ├── momentum_train/     # Training module with trainer and metrics
│   └── momentum_live/      # Live trading (placeholder for future work)
├── config/                 # Configuration files (YAML)
├── data/                   # Data directories (raw, processed, extracted)
├── scripts/                # Data processing scripts
├── tests/                  # Integration tests
└── legacy_src_backup/     # Backup of original structure (for reference)
```

## 📦 Packages

### momentum_env
Trading environment package. A Gymnasium environment for reinforcement learning.

**Install:**
```bash
cd packages/momentum_env
pip install -e .
```

**Import:**
```python
from momentum_env import TradingEnv, TradingEnvConfig
```

**Dependencies:**
- gymnasium
- numpy
- pandas
- matplotlib

### momentum_agent
Rainbow DQN agent implementation with:
- Distributional RL (C51)
- Prioritized Experience Replay (PER)
- Multi-step returns
- Double Q-Learning
- Noisy Nets for exploration
- Dueling network architecture

**Install:**
```bash
cd packages/momentum_agent
pip install -e .
```

**Import:**
```python
from momentum_agent import RainbowDQNAgent, RainbowNetwork, PrioritizedReplayBuffer
```

**Dependencies:**
- torch
- numpy
- PyYAML
- momentum_env

### momentum_train
Training module with trainer, metrics, and data management.

**Install:**
```bash
cd packages/momentum_train
pip install -e .
```

**Import:**
```python
from momentum_train import RainbowTrainerModule, DataManager, PerformanceTracker
from momentum_train import calculate_sharpe_ratio, calculate_max_drawdown
```

**Dependencies:**
- torch
- numpy
- pandas
- scikit-learn
- PyYAML
- momentum_agent
- momentum_env
- tensorboard
- wandb

### momentum_live
Live trading package (skeleton - under development).

## 🚀 Quick Start

### 1. Install All Packages

Install packages in dependency order:

```bash
# Base package
cd packages/momentum_env
pip install -e .

# Agent package
cd ../momentum_agent
pip install -e .

# Training package
cd ../momentum_train
pip install -e .

# Return to root
cd ../../..
```

### 2. Setup Data

Data should be organized as:
```
data/
├── raw/           # Raw market data (CSV files)
├── extracted/     # Extracted/processed raw data
└── processed/     # Final processed data
    ├── train/
    ├── validation/
    └── test/
```

### 3. Configure Training

Edit `config/training_config.yaml` to set your hyperparameters.

### 4. Run Training

```bash
python -m momentum_train.run_training --config_path config/training_config.yaml
```

Or with resume:
```bash
python -m momentum_train.run_training --config_path config/training_config.yaml --resume
```

### 5. Run Evaluation

Set `mode: eval` in config and specify `eval_model_prefix`.

## 📊 Features

### Trading Environment
- Normalized OHLCV observations with configurable window size
- Discrete action space: Hold, Buy (10%/25%/50%), Sell (10%/25%/100%)
- Account state tracking (position, balance, transaction costs)
- Configurable fees and initial balance

### Agent Architecture
- Rainbow DQN with full feature set
- Transformer-based state encoding
- Distributional value estimation (C51)
- Prioritized experience replay
- Multi-step returns
- Noisy linear layers for exploration

### Training Features
- Checkpoint management
- Early stopping
- Performance metrics (Sharpe ratio, max drawdown, avg returns)
- TensorBoard integration
- WandB logging support
- Validation on separate data splits

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific package tests
pytest tests/momentum_env/
pytest tests/momentum_agent/
pytest tests/momentum_train/

# Run with coverage
pytest --cov=momentum_env --cov=momentum_agent --cov=momentum_train
```

## 📝 Development

### Making Changes
1. Edit code in `packages/[package_name]/src/[package_name]/`
2. Changes are immediately available due to editable installs
3. Re-install package if you add new dependencies to `pyproject.toml`

### Adding a New Package
1. Create `packages/new_package/`
2. Add `src/new_package/` directory structure
3. Create `pyproject.toml` with dependencies
4. Install with `pip install -e .`

### Package Dependencies
- `momentum_env` has no internal dependencies
- `momentum_agent` depends on `momentum_env`
- `momentum_train` depends on `momentum_agent` and `momentum_env`
- `momentum_live` depends on `momentum_agent` and `momentum_env`

## 🔧 Configuration

Key configuration files:
- `config/training_config.yaml` - Training hyperparameters
- `config/extract_raw_config.yaml` - Data extraction settings
- `config/split_config.yaml` - Data splitting configuration

## 📈 Monitoring

- **TensorBoard**: Training metrics and losses
- **WandB**: Experiment tracking and hyperparameter tuning
- **Logs**: `logs/training.log` contains detailed training logs

## 🤝 Contributing

This is currently a personal project. If you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure nothing breaks
5. Submit a pull request

## 📄 License

MIT License

## 📚 Additional Notes

### Legacy Structure
The original code has been backed up to `legacy_src_backup/` directory for reference.

### Why Monorepo?
This monorepo structure was chosen because:
- ✅ Fast iteration (no need to publish packages)
- ✅ Cross-package changes are easy
- ✅ Single repository to clone
- ✅ Consistent version management
- ✅ Easier dependency management

### Migration from Separate Repos
This monorepo consolidates:
- `trader` → `packages/momentum_train`
- `trading-env` → `packages/momentum_env`
- `momentum-agent` → `packages/momentum_agent`
- `momentum-live` → `packages/momentum_live` (planned)
