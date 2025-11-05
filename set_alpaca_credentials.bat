@echo off
REM Set Alpaca API credentials as environment variables
REM Run this once per command prompt session
REM
REM These credentials are for paper trading only
REM Make sure crypto trading permissions are enabled in your Alpaca dashboard

set ALPACA_API_KEY=YOUR_ALPACA_API_KEY_HERE
set ALPACA_API_SECRET=YOUR_ALPACA_API_SECRET_HERE
set ALPACA_PAPER_TRADING=true

echo Alpaca API credentials set successfully!
echo ALPACA_API_KEY: %ALPACA_API_KEY%
echo ALPACA_API_SECRET: [HIDDEN FOR SECURITY]
echo ALPACA_PAPER_TRADING: %ALPACA_PAPER_TRADING%
