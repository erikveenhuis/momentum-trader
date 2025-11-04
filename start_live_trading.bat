@echo off
REM Live Momentum Trading Launcher
REM This script sets up credentials and starts live trading

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    echo Please run setup first or check your installation.
    echo.
    pause
    exit /b 1
)

echo ========================================
echo  🚀 MOMENTUM LIVE TRADING SYSTEM
echo ========================================
echo.
echo Setting Alpaca API credentials...
echo.

REM Set Alpaca API credentials
set ALPACA_API_KEY=PKWKLNE6QHTADHRTUTB22EOYWW
set ALPACA_API_SECRET=AUjDNbw5Atb8jGxvraRnhs1xg4P81mgKwzoQFRTS2Znw
set ALPACA_PAPER_TRADING=true

echo ✅ Credentials configured (Paper Trading: %ALPACA_PAPER_TRADING%)
echo.

REM Prevent PC from sleeping during live trading (monitor can sleep)
echo Preventing system sleep during live trading (monitor can still sleep)...
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change standby-timeout-dc 0 >nul 2>&1
powercfg /change hibernate-timeout-dc 0 >nul 2>&1
echo.

REM Check if virtual environment exists and activate it
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please check your virtual environment setup.
    echo.
    echo Restoring default power settings...
    powercfg /change standby-timeout-ac 30 >nul 2>&1
    powercfg /change standby-timeout-dc 15 >nul 2>&1
    echo Power settings restored.
    echo.
    pause
    exit /b 1
)

REM Verify python is available and can import the module
echo Checking if momentum_live module is available...
python -c "import momentum_live.cli" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot import momentum_live.cli module
    echo Attempting to refresh editable packages automatically...
    echo.
    call refresh_packages.bat --no-pause
    if errorlevel 1 (
        echo ERROR: Automatic package refresh failed
        echo Please run refresh_packages.bat manually and check for errors.
        echo.
        echo Restoring default power settings...
        powercfg /change standby-timeout-ac 30 >nul 2>&1
        powercfg /change standby-timeout-dc 15 >nul 2>&1
        echo Power settings restored.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Checking module import again after refresh...
    python -c "import momentum_live.cli" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Still cannot import momentum_live.cli module after refresh
        echo Please check the refresh logs above for errors.
        echo.
        echo Restoring default power settings...
        powercfg /change standby-timeout-ac 30 >nul 2>&1
        powercfg /change standby-timeout-dc 15 >nul 2>&1
        echo Power settings restored.
        echo.
        pause
        exit /b 1
    )
    echo ✓ momentum_live module imported successfully after refresh
)

echo Starting live trading for BTC/USD and ETH/USD...
echo Press Ctrl+C to stop the system gracefully
echo.
echo ========================================

REM Run the live trading system
python -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO

echo.
echo Live trading session ended.
echo Restoring default power settings...
powercfg /change standby-timeout-ac 30 >nul 2>&1
powercfg /change standby-timeout-dc 15 >nul 2>&1
echo Power settings restored.
echo.
pause
