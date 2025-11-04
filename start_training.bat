@echo off
REM Training shortcut for Momentum Trader
REM Double-click this file to start training

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

REM Check if config file exists
if not exist "config\training_config.yaml" (
    echo ERROR: Training config file not found at config\training_config.yaml
    echo Please ensure the config file exists.
    echo.
    pause
    exit /b 1
)

REM Prevent PC from sleeping during training (monitor can sleep)
echo Preventing system sleep during training (monitor can still sleep)...
powercfg /change standby-timeout-ac 0 >nul 2>&1
powercfg /change hibernate-timeout-ac 0 >nul 2>&1
powercfg /change standby-timeout-dc 0 >nul 2>&1
powercfg /change hibernate-timeout-dc 0 >nul 2>&1

REM Check if virtual environment exists and activate it
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Please check your virtual environment setup.
    echo.
    goto :cleanup_and_exit
)

REM Verify python is available and can import the module
echo Checking if momentum_train module is available...
python -c "import momentum_train.run_training" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Cannot import momentum_train.run_training module
    echo Attempting to refresh editable packages automatically...
    echo.
    call refresh_packages.bat --no-pause
    if errorlevel 1 (
        echo ERROR: Automatic package refresh failed
        echo Please run refresh_packages.bat manually and check for errors.
        echo.
        goto :cleanup_and_exit
    )
    echo.
    echo Checking module import again after refresh...
    python -c "import momentum_train.run_training" >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Still cannot import momentum_train.run_training module after refresh
        echo Please check the refresh logs above for errors.
        echo.
        goto :cleanup_and_exit
    )
    echo ✓ momentum_train module imported successfully after refresh
)

REM Start training with default config in background
echo.
echo Starting training in background...
echo Training logs will be written to logs/training.log
echo You can monitor progress by opening the "Momentum Trader Training" window
echo.
start "Momentum Trader Training" cmd /c "cd /d %~dp0 && call venv\Scripts\activate.bat && python -m momentum_train.run_training --config_path config/training_config.yaml && echo. && echo Training completed successfully! && echo Restoring default power settings... && powercfg /change standby-timeout-ac 30 >nul 2>&1 && powercfg /change standby-timeout-dc 15 >nul 2>&1 && echo Power settings restored. && pause"

REM Check if start command succeeded
if errorlevel 1 (
    echo ERROR: Failed to start training process
    echo.
    goto :cleanup_and_exit
)

echo.
echo SUCCESS: Training started in background with sleep prevention enabled!
echo Power settings will be restored when training completes.
echo You can now close this window safely.
echo.
pause
exit /b 0

:cleanup_and_exit
REM Restore default power settings on error
echo Restoring default power settings...
powercfg /change standby-timeout-ac 30 >nul 2>&1
powercfg /change standby-timeout-dc 15 >nul 2>&1
echo.
echo Press any key to exit...
pause >nul
exit /b 1
