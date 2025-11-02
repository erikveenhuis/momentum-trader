@echo off
REM Training shortcut for Momentum Trader - Resume Mode
REM Double-click this file to resume training from the latest checkpoint

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start training with resume flag
echo Resuming training from latest checkpoint...
echo.
python -m momentum_train.run_training --config_path config/training_config.yaml --resume

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo Training ended with an error.
    pause
)
