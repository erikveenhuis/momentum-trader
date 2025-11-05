@echo off
REM Set Alpaca API credentials as environment variables
REM Run this once per command prompt session
REM
REM These credentials are for paper trading only
REM Make sure crypto trading permissions are enabled in your Alpaca dashboard

set ALPACA_API_KEY=PKISXJBRRJT7MTI553CDXGNHOJ
set ALPACA_API_SECRET=9yZ7cNgr7StgNctiWBedoWpcpHjX5nxkBdP9s1JSbW2L
set ALPACA_PAPER_TRADING=true

echo Alpaca API credentials set successfully!
echo ALPACA_API_KEY: %ALPACA_API_KEY%
echo ALPACA_API_SECRET: [HIDDEN FOR SECURITY]
echo ALPACA_PAPER_TRADING: %ALPACA_PAPER_TRADING%
