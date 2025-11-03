@echo off
REM Install momentum trader packages in virtual environment

set NO_PAUSE=0
if /I "%~1"=="--no-pause" set NO_PAUSE=1

cd /d "%~dp0"

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo Checking currently installed packages...
pip list
echo.

echo Installing momentum packages...
echo Installing momentum-core...
pip install -e packages/momentum_core
if errorlevel 1 (
    echo ERROR: Failed to install momentum-core
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Installing momentum-env...
pip install -e packages/momentum_env
if errorlevel 1 (
    echo ERROR: Failed to install momentum-env
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Installing momentum-agent...
pip install -e packages/momentum_agent
if errorlevel 1 (
    echo ERROR: Failed to install momentum-agent
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Installing momentum-train...
pip install -e packages/momentum_train
if errorlevel 1 (
    echo ERROR: Failed to install momentum-train
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo Installing momentum-live...
pip install -e packages/momentum_live
if errorlevel 1 (
    echo ERROR: Failed to install momentum-live
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo All packages installed successfully!
echo.
echo Verifying installation...
python -c "import momentum_train.run_training; print('✓ momentum_train module imported successfully')"
if errorlevel 1 (
    echo ERROR: Still cannot import momentum_train
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo Installation completed successfully!
if "%NO_PAUSE%"=="0" pause
