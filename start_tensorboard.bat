@echo off
REM TensorBoard launcher for Momentum Trader
REM Double-click this file to open TensorBoard and view training metrics

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

REM Check if virtual environment exists and activate it
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please check your virtual environment setup.
    echo.
    pause
    exit /b 1
)

REM Check if tensorboard is available
echo Checking if tensorboard is available...
tensorboard --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: TensorBoard is not installed or not available
    echo Please run refresh_packages.bat first to install the required packages.
    echo.
    pause
    exit /b 1
)

REM Check if models/runs directory exists
if not exist "models\runs" (
    echo Error: models\runs directory not found!
    echo Make sure you have run training at least once.
    pause
    exit /b 1
)

REM Prevent PC from sleeping during TensorBoard session (monitor can sleep)
echo Preventing system sleep during TensorBoard session (monitor can still sleep)...
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change standby-timeout-dc 0 >nul 2>&1
powercfg /change hibernate-timeout-dc 0 >nul 2>&1
echo.

REM Start TensorBoard pointing to the runs directory
echo Starting TensorBoard...
echo TensorBoard will be available at: http://localhost:6006
echo.
echo Press Ctrl+C to stop TensorBoard when done.
echo.

tensorboard --logdir=models\runs --port=6006 --host=localhost

echo.
echo TensorBoard session ended.
echo Restoring default power settings...
powercfg /change standby-timeout-ac 30 >nul 2>&1
powercfg /change standby-timeout-dc 15 >nul 2>&1
echo Power settings restored.
echo.
pause
