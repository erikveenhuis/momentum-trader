# Set Alpaca API credentials as environment variables
# Run this once per PowerShell session

$env:ALPACA_API_KEY = "PKWKLNE6QHTADHRTUTB22EOYWW"
$env:ALPACA_API_SECRET = "AUjDNbw5Atb8jGxvraRnhs1xg4P81mgKwzoQFRTS2Znw"
$env:ALPACA_PAPER_TRADING = "true"

Write-Host "Alpaca API credentials set successfully!"
Write-Host "ALPACA_API_KEY: $env:ALPACA_API_KEY"
Write-Host "ALPACA_API_SECRET: [HIDDEN FOR SECURITY]"
Write-Host "ALPACA_PAPER_TRADING: $env:ALPACA_PAPER_TRADING"
