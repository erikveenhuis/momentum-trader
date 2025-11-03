@echo off
REM Check training progress
REM Shows recent training log entries

echo Momentum Trader Training Progress
echo ==================================
echo.

if exist "logs\training.log" (
    echo Recent training activity:
    echo --------------------------
    powershell -Command "Get-Content -Path 'logs\training.log' -Tail 20"
    echo.
    echo Full log file: logs\training.log
) else (
    echo No training log found. Start training first.
)

echo.
echo Press any key to continue...
pause >nul
