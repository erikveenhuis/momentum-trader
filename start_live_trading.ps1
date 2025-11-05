# Live Momentum Trading Launcher
# This script sets up credentials and starts live trading

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 🚀 MOMENTUM LIVE TRADING SYSTEM" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking Alpaca API credentials..." -ForegroundColor Yellow
Write-Host ""

# Check if credentials are set
if (-not $env:ALPACA_API_KEY) {
    Write-Host "ERROR: ALPACA_API_KEY environment variable not set" -ForegroundColor Red
    Write-Host "Please set your Alpaca credentials using set_alpaca_credentials.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
if (-not $env:ALPACA_API_SECRET) {
    Write-Host "ERROR: ALPACA_API_SECRET environment variable not set" -ForegroundColor Red
    Write-Host "Please set your Alpaca credentials using set_alpaca_credentials.ps1" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ Credentials found (Paper Trading: $env:ALPACA_PAPER_TRADING)" -ForegroundColor Green
Write-Host ""
Write-Host "Starting live trading for BTC/USD and ETH/USD..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the system gracefully" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Change to script directory and run the live trading system
Set-Location $PSScriptRoot
& "venv\Scripts\python.exe" -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO

Write-Host ""
Write-Host "Live trading session ended." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
