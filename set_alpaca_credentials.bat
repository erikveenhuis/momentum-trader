@echo off
REM Set Alpaca API credentials as environment variables
REM Run this once per command prompt session
REM
REM IMPORTANT: Replace YOUR_ALPACA_API_KEY_HERE and YOUR_ALPACA_API_SECRET_HERE
REM with your actual Alpaca API credentials from https://alpaca.markets/

set ALPACA_API_KEY=YOUR_ALPACA_API_KEY_HERE
set ALPACA_API_SECRET=YOUR_ALPACA_API_SECRET_HERE
set ALPACA_PAPER_TRADING=true

echo Alpaca API credentials set successfully!
echo ALPACA_API_KEY: %ALPACA_API_KEY%
echo ALPACA_API_SECRET: [HIDDEN FOR SECURITY]
echo ALPACA_PAPER_TRADING: %ALPACA_PAPER_TRADING%
