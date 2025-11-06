# Live Momentum Trading Launcher
# This script sets up credentials and starts live trading

# Check for lock file (indicates another instance is running)
if (Test-Path "momentum_trading.lock") {
    Write-Host "⚠️  WARNING: Found momentum_trading.lock file" -ForegroundColor Yellow
    Write-Host "This indicates another momentum trading session may be running." -ForegroundColor Yellow
    Write-Host "This could cause connection limit issues with Alpaca." -ForegroundColor Yellow
    Write-Host ""
    $lockContent = Get-Content "momentum_trading.lock"
    Write-Host "Lock file created: $lockContent" -ForegroundColor Gray
    Write-Host ""
    Write-Host "If you're sure no other instance is running, you can delete the lock file." -ForegroundColor Yellow
    $response = Read-Host "Press Enter to continue anyway, or 'q' to quit"
    if ($response -eq 'q') {
        exit 1
    }
    Write-Host ""
} else {
    Get-Date -Format "yyyy-MM-dd HH:mm:ss" | Out-File -FilePath "momentum_trading.lock"
}

# Load environment variables from .env file if it exists
if (Test-Path ".env") {
    Write-Host "Loading environment variables from .env file..." -ForegroundColor Yellow
    Get-Content ".env" | ForEach-Object {
        if ($_ -notmatch '^\s*#' -and $_ -match '^([^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
    Write-Host "✅ Environment variables loaded from .env file" -ForegroundColor Green
    Write-Host ""
}

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

# Reset Alpaca account before starting trading (paper trading only)
if ($env:ALPACA_PAPER_TRADING -eq "true") {
    Write-Host "Resetting Alpaca paper trading account..." -ForegroundColor Yellow
    & "venv\Scripts\python.exe" -m momentum_live.reset_account --log-level INFO --wait-interval 2 --timeout 120
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to reset Alpaca paper trading account." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "⚠️  REAL MONEY TRADING MODE: Skipping account reset for safety." -ForegroundColor Yellow
    Write-Host "   Please ensure your live trading account is in the desired state before proceeding." -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "Starting live trading for BTC/USD and ETH/USD..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the system gracefully" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Change to script directory and run the live trading system
Set-Location $PSScriptRoot
& "venv\Scripts\python.exe" -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO

Write-Host ""
if ($LASTEXITCODE -eq 1) {
    Write-Host "❌ Live trading failed to start. Check the logs above for details." -ForegroundColor Red
    Write-Host "This may be due to connection limits or authentication issues." -ForegroundColor Yellow
} else {
    Write-Host "Live trading session ended." -ForegroundColor Yellow
}

# Clean up lock file
Write-Host "Cleaning up lock file..." -ForegroundColor Gray
if (Test-Path "momentum_trading.lock") {
    Remove-Item "momentum_trading.lock"
}

Read-Host "Press Enter to exit"
