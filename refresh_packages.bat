@echo off
REM Refresh editable installs of Momentum Trader packages by uninstalling any
REM previously installed distributions and reinstalling in editable mode.

set NO_PAUSE=0
if /I "%~1"=="--no-pause" set NO_PAUSE=1

REM Ensure script runs from repo root
cd /d "%~dp0"

REM Check for virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv\Scripts\activate.bat
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo Removing any previously installed momentum packages...
for %%P in (momentum-train momentum-env momentum-agent momentum-core momentum-live) do (
    pip show %%P >nul 2>&1
    if errorlevel 1 (
        echo   %%P not currently installed. Skipping uninstall.
    ) else (
        echo   Uninstalling %%P ...
        pip uninstall -y %%P >nul
        if errorlevel 1 (
            echo ERROR: Failed to uninstall %%P
            if "%NO_PAUSE%"=="0" pause
            exit /b 1
        )
    )
)

echo.
echo Reinstalling editable packages...
call install_packages.bat --no-pause
if errorlevel 1 (
    echo ERROR: Unable to reinstall editable packages
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo Momentum Trader packages refreshed successfully.
if "%NO_PAUSE%"=="0" (
    echo.
    echo Press any key to continue...
    pause >nul
)

exit /b 0
