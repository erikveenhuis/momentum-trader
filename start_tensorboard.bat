@echo off
REM TensorBoard launcher for Momentum Trader
REM Double-click this file to open TensorBoard and view training metrics

REM Get the directory where this batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check if models/runs directory exists
if not exist "models\runs" (
    echo Error: models\runs directory not found!
    echo Make sure you have run training at least once.
    pause
    exit /b 1
)

REM Start TensorBoard pointing to the runs directory
echo Starting TensorBoard...
echo TensorBoard will be available at: http://localhost:6006
echo.
echo Press Ctrl+C to stop TensorBoard when done.
echo.

tensorboard --logdir=models\runs --port=6006 --host=localhost

pause
