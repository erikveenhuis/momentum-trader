#!/usr/bin/env bash
# Load Alpaca *paper* trading credentials from the repository root .env (gitignored).
# Two API surfaces are loaded:
#   1. Data API (ALPACA_API_KEY / ALPACA_API_SECRET) — used by CryptoDataStream
#      to subscribe to live minute bars. Required.
#   2. Broker API (ALPACA_BROKER_*) — used by BrokerClient to manage per-pair
#      sandbox sub-accounts and to JNLC-fund them. Required for momentum_live.
#
#   source scripts/env-paper.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
_ENV_FILE="$_REPO_ROOT/.env"

if [[ ! -f "$_ENV_FILE" ]]; then
  echo "env-paper.sh: missing $_ENV_FILE — see README 'Live Trading > Credentials'" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
source "$_ENV_FILE"
set +a

export ALPACA_PAPER_TRADING="${ALPACA_PAPER_TRADING:-true}"
export ALPACA_CRYPTO_FEED="${ALPACA_CRYPTO_FEED:-us}"
export ALPACA_BROKER_BASE_URL="${ALPACA_BROKER_BASE_URL:-https://broker-api.sandbox.alpaca.markets}"

_missing=()
# ALPACA_BROKER_ACCOUNT_ID is optional: it's only needed for --reset-mode={soft,hard}
# (JNLC-based funding). Leave it blank when you just want to adopt an existing
# crypto-funded sub-account and run the agent with --reset-mode none.
for _var in ALPACA_API_KEY ALPACA_API_SECRET \
            ALPACA_BROKER_API_KEY ALPACA_BROKER_API_SECRET; do
  if [[ -z "${!_var:-}" ]]; then
    _missing+=("$_var")
  fi
done

if (( ${#_missing[@]} > 0 )); then
  echo "env-paper.sh: missing required vars in .env: ${_missing[*]}" >&2
  return 1 2>/dev/null || exit 1
fi

_funding_display="${ALPACA_BROKER_ACCOUNT_ID:-<unset>}"
echo "Alpaca env loaded (data=...${ALPACA_API_KEY: -4}, broker=...${ALPACA_BROKER_API_KEY: -4}, funding=${_funding_display})"
