# Set Alpaca API credentials as environment variables
# Run this once per PowerShell session
#
# IMPORTANT: Replace YOUR_ALPACA_API_KEY_HERE and YOUR_ALPACA_API_SECRET_HERE
# with your actual Alpaca API credentials from https://alpaca.markets/

$env:ALPACA_API_KEY = "YOUR_ALPACA_API_KEY_HERE"
$env:ALPACA_API_SECRET = "YOUR_ALPACA_API_SECRET_HERE"
$env:ALPACA_PAPER_TRADING = "true"

Write-Host "Alpaca API credentials set successfully!"
Write-Host "ALPACA_API_KEY: $env:ALPACA_API_KEY"
Write-Host "ALPACA_API_SECRET: [HIDDEN FOR SECURITY]"
Write-Host "ALPACA_PAPER_TRADING: $env:ALPACA_PAPER_TRADING"
