# Set Alpaca API credentials as environment variables
# Run this once per PowerShell session
#
# These credentials are for paper trading only
# Make sure crypto trading permissions are enabled in your Alpaca dashboard

$env:ALPACA_API_KEY = "YOUR_ALPACA_API_KEY_HERE"
$env:ALPACA_API_SECRET = "YOUR_ALPACA_API_SECRET_HERE"
$env:ALPACA_PAPER_TRADING = "true"

Write-Host "Alpaca API credentials set successfully!"
Write-Host "ALPACA_API_KEY: $env:ALPACA_API_KEY"
Write-Host "ALPACA_API_SECRET: [HIDDEN FOR SECURITY]"
Write-Host "ALPACA_PAPER_TRADING: $env:ALPACA_PAPER_TRADING"
