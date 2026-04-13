#!/usr/bin/env bash
# Load Alpaca *paper* trading credentials from the repository root .env (gitignored).
# Paper accounts can be reset (close positions / cancel orders) in the Alpaca dashboard;
# no real funds are at risk.
#
#   source scripts/env-paper.sh

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
_ENV_FILE="$_REPO_ROOT/.env"

if [[ ! -f "$_ENV_FILE" ]]; then
  echo "env-paper.sh: missing $_ENV_FILE — add ALPACA_API_KEY and ALPACA_API_SECRET there" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
# shellcheck disable=SC1090
source "$_ENV_FILE"
set +a

export ALPACA_PAPER_TRADING="${ALPACA_PAPER_TRADING:-true}"
export ALPACA_CRYPTO_FEED="${ALPACA_CRYPTO_FEED:-us}"

if [[ -z "${ALPACA_API_KEY:-}" || -z "${ALPACA_API_SECRET:-}" ]]; then
  echo "env-paper.sh: ALPACA_API_KEY and ALPACA_API_SECRET must be set in .env" >&2
  return 1 2>/dev/null || exit 1
fi

echo "Alpaca paper trading env vars loaded (key=...${ALPACA_API_KEY: -4})"
