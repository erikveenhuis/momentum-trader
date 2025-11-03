# Live Momentum Trading Launcher
# This script sets up credentials and starts live trading

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 🚀 MOMENTUM LIVE TRADING SYSTEM" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Setting Alpaca API credentials..." -ForegroundColor Yellow
Write-Host ""

# Set Alpaca API credentials
$env:ALPACA_API_KEY = "PKWKLNE6QHTADHRTUTB22EOYWW"
$env:ALPACA_API_SECRET = "AUjDNbw5Atb8jGxvraRnhs1xg4P81mgKwzoQFRTS2Znw"
$env:ALPACA_PAPER_TRADING = "true"

Write-Host "✅ Credentials configured (Paper Trading: $env:ALPACA_PAPER_TRADING)" -ForegroundColor Green
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
