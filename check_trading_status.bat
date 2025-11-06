@echo off
REM Check for running momentum trading processes and lock files

echo ========================================
echo  🔍 MOMENTUM TRADING STATUS CHECK
echo ========================================
echo.

echo Checking for lock file...
if exist "momentum_trading.lock" (
    echo ❌ Found momentum_trading.lock file
    for /f "tokens=*" %%i in (momentum_trading.lock) do echo    Created: %%i
    echo.
    echo This indicates a trading session may still be running or was not closed properly.
    echo.
    set /p choice="Delete lock file? (y/n): "
    if /i "!choice!"=="y" (
        del "momentum_trading.lock"
        echo ✅ Lock file deleted.
    ) else (
        echo Lock file kept.
    )
) else (
    echo ✅ No lock file found.
)
echo.

echo Checking for running Python processes...
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE | findstr /C:"python.exe" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️  Found Python processes:
    tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
    echo.
    echo If any of these are momentum trading processes, close them before starting a new session.
) else (
    echo ✅ No Python processes found.
)
echo.

echo Checking for trading log activity...
if exist "logs\live_trading.log" (
    echo Recent log activity:
    powershell -command "Get-Content 'logs\live_trading.log' -Tail 5"
) else (
    echo No trading logs found.
)
echo.

echo ========================================
echo  STATUS CHECK COMPLETE
echo ========================================
echo.
pause
