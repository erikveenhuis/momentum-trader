# 🚀 Momentum Live Trading System

Real-time cryptocurrency trading using reinforcement learning on Alpaca's paper trading platform.

## Quick Start

### Option 1: Batch File (Windows)
Double-click `start_live_trading.bat` to launch the system.

### Option 2: PowerShell
```powershell
.\start_live_trading.ps1
```

### Option 3: Manual Command
```bash
# Set credentials first
.\set_alpaca_credentials.bat  # or .ps1

# Then run live trading
venv\Scripts\python.exe -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO
```

## What It Does

- **Connects** to Alpaca's live crypto data stream
- **Processes** real-time OHLCV bars for BTC/USD and ETH/USD
- **Makes decisions** using a Rainbow DQN agent trained on momentum patterns
- **Executes orders** on your Alpaca paper trading account
- **Tracks portfolio** with shared balance across symbols

## Key Features

- ✅ **Paper Trading**: All orders execute on paper trading account
- ✅ **Multi-Symbol**: Handles multiple cryptocurrencies simultaneously
- ✅ **Shared Balance**: Intelligent cash management across all positions
- ✅ **Real-time**: Processes live market data as it arrives
- ✅ **Safe**: Includes position limits and cash availability checks

## System Requirements

- Windows 10/11 with PowerShell
- Python 3.13+
- Alpaca paper trading account
- CUDA-compatible GPU (recommended for performance)

## Configuration

### Symbols
Modify the symbols in the batch files or CLI command:
```bash
--symbols BTC/USD,ETH/USD,SOL/USD
```

### Credentials
Credentials are automatically set by the launcher scripts. For manual setup:
```batch
set ALPACA_API_KEY=your_key_here
set ALPACA_API_SECRET=your_secret_here
set ALPACA_PAPER_TRADING=true
```

## Live Output

When running, you'll see logs like:
```
BTC/USD BUY 25% at 45123.45 | Shared PV 1050.67 (Balance: 750.23) | valid=True | Order: abc123
ETH/USD HOLD 0% at 2456.78 | Shared PV 1050.67 (Balance: 750.23) | valid=True
```

## Stopping the System

- Press `Ctrl+C` in the terminal to gracefully stop
- The system will close connections and save final state

## Troubleshooting

### Common Issues

1. **"Missing Alpaca credentials"**
   - Run the credential setup script first
   - Check that environment variables are set

2. **"Checkpoint loading failed"**
   - System automatically falls back to fresh agent
   - This is normal for new installations

3. **"CUDA not available"**
   - System falls back to CPU automatically
   - Performance may be slower

### Performance Notes

- **GPU**: ~100-200ms per decision (recommended)
- **CPU**: ~500-1000ms per decision
- **Network**: Dependent on Alpaca's data feed latency

## Safety Features

- **Paper Trading**: Never uses real money
- **Position Limits**: Cannot sell more than owned
- **Cash Checks**: Cannot buy with insufficient balance
- **Minimum Orders**: Prevents tiny, costly orders
- **Error Handling**: Graceful failure recovery

## Architecture

```
Alpaca Data Stream → Feature Normalizer → Rainbow Agent → Order Execution → Portfolio Update
     ↓                    ↓                     ↓              ↓                ↓
 BTC/USD Bars     60-bar OHLCV windows    BUY/SELL/HOLD   Market Orders    Shared Balance
 ETH/USD Bars     Rolling normalization   decisions       Paper Trading    Position Tracking
```

## Next Steps

1. **Monitor Performance**: Check Alpaca dashboard for paper trading results
2. **Adjust Symbols**: Add or remove cryptocurrencies from watchlist
3. **Model Training**: Retrain agent on current market data for better performance
4. **Risk Management**: Implement position sizing and stop-loss logic

---

**⚠️ Disclaimer**: This is experimental software for educational purposes. Always verify trades and use at your own risk.
