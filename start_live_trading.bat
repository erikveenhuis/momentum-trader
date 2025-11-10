@echo off
REM Live Momentum Trading Launcher
REM This script sets up credentials and starts live trading

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Terminate any existing momentum_live.cli processes before starting
echo Checking for existing live trading Python processes...
set "KILLED_PROCESSES=0"
for /f "usebackq tokens=*" %%P in (`powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*momentum_live.cli*' } | ForEach-Object { $_.ProcessId }"`) do (
    echo Terminating existing momentum_live.cli process ID %%P
    taskkill /PID %%P /F >nul 2>&1
    set "KILLED_PROCESSES=1"
)
if "%KILLED_PROCESSES%"=="0" (
    echo No existing live trading processes found.
) else (
    echo Existing live trading processes terminated.
)
echo.

REM Load environment variables from .env file if it exists
if exist ".env" (
    echo Loading environment variables from .env file...
    for /f "tokens=*" %%i in (.env) do (
        echo %%i | findstr /r "^#" >nul 2>&1
        if errorlevel 1 (
            for /f "tokens=1,* delims==" %%a in ("%%i") do set "%%a=%%b" 2>nul
        )
    )
    echo ✅ Environment variables loaded from .env file
    echo.
)

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
echo Checking Alpaca API credentials...
echo.

REM Check if credentials are set
if "%ALPACA_API_KEY%"=="" (
    echo ERROR: ALPACA_API_KEY environment variable not set
    echo Please set your Alpaca credentials using set_alpaca_credentials.bat
    pause
    exit /b 1
)
if "%ALPACA_API_SECRET%"=="" (
    echo ERROR: ALPACA_API_SECRET environment variable not set
    echo Please set your Alpaca credentials using set_alpaca_credentials.bat
    pause
    exit /b 1
)

echo ✅ Credentials found (Paper Trading: %ALPACA_PAPER_TRADING%)
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

REM Reset Alpaca account before starting trading (paper trading only)
if "%ALPACA_PAPER_TRADING%"=="true" (
    echo Resetting Alpaca paper trading account...
    python -m momentum_live.reset_account --log-level INFO --wait-interval 2 --timeout 120
    if errorlevel 1 (
        echo ERROR: Failed to reset Alpaca paper trading account.
        echo.
        echo Restoring default power settings...
        powercfg /change standby-timeout-ac 30 >nul 2>&1
        powercfg /change standby-timeout-dc 15 >nul 2>&1
        echo Power settings restored.
        echo.
        pause
        exit /b 1
    )
) else (
    echo ⚠️  REAL MONEY TRADING MODE: Skipping account reset for safety.
    echo    Please ensure your live trading account is in the desired state before proceeding.
    echo.
)

echo Starting live trading for BTC/USD and ETH/USD...
echo Press Ctrl+C to stop the system gracefully
echo.
echo ========================================

REM Run the live trading system
python -m momentum_live.cli --symbols BTC/USD,ETH/USD --log-level INFO
set EXITCODE=%errorlevel%

echo.
if %EXITCODE% equ 1 (
    echo ❌ Live trading failed to start. Check the logs above for details.
    echo This may be due to connection limits or authentication issues.
) else (
    echo Live trading session ended.
)
echo Restoring default power settings...
powercfg /change standby-timeout-ac 30 >nul 2>&1
powercfg /change standby-timeout-dc 15 >nul 2>&1
echo Power settings restored.
echo.
pause
