@echo off
REM Resume training while resetting optimizer and LR scheduler state to config defaults.

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

REM Check if checkpoint files exist for resuming
echo Checking for available checkpoints...
for %%f in ("models\checkpoint_trainer_*.pt") do (
    goto :found_checkpoint
)
echo No checkpoint files found.
goto :no_checkpoints

:no_checkpoints
    echo.
    echo ====================================================
    echo   ❌ NO CHECKPOINTS FOUND FOR RESUME
    echo ====================================================
    echo.
    echo The resume feature requires checkpoint files to continue training.
    echo Checkpoints are automatically saved during training based on:
    echo   - checkpoint_save_freq setting in config\training_config.yaml
    echo   - Currently set to save every 30 episodes
    echo.
    echo To create checkpoints for resuming:
    echo   1. Start fresh training with start_training.bat
    echo   2. Let it run for at least 30 episodes to save first checkpoint
    echo   3. Interrupt training manually (Ctrl+C) to create resume point
    echo   4. Then use this script to continue with a reset learning rate
    echo.
    goto :cleanup_and_exit

:found_checkpoint
goto :start_training

:start_training
echo.
echo Found checkpoint files! Resuming training with LR reset in background...
echo The optimizer and LR scheduler states will be re-initialised to match the current config.
echo Training logs will be written to logs\training.log
echo You can monitor progress by opening the "Momentum Trader Training (Resume - Reset LR)" window
echo.
start "Momentum Trader Training (Resume - Reset LR)" cmd /c "cd /d %~dp0 && call venv\Scripts\activate.bat && python -m momentum_train.run_training --config_path config/training_config.yaml --resume --reset-lr-on-resume && echo. && echo Training completed successfully! && echo Restoring default power settings... && powercfg /change standby-timeout-ac 30 >nul 2>&1 && powercfg /change standby-timeout-dc 15 >nul 2>&1 && echo Power settings restored. && pause"

REM Check if start command succeeded
if errorlevel 1 (
    echo ERROR: Failed to start training resume process
    echo.
    goto :cleanup_and_exit
)

echo.
echo SUCCESS: Training resumed in background with learning rate reset and sleep prevention enabled!
echo Power settings will be restored when training completes.
echo You can now close this window safely.
echo.
pause
exit /b 0

:cleanup_and_exit
REM Restore default power settings on error (only if we changed them)
echo Restoring default power settings...
powercfg /change standby-timeout-ac 30 >nul 2>&1
powercfg /change standby-timeout-dc 15 >nul 2>&1
echo.
echo Press any key to exit...
pause >nul
exit /b 1
