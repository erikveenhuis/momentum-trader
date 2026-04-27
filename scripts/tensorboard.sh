#!/usr/bin/env bash
# Launch TensorBoard in an isolated venv to avoid dependency conflicts
# with the training environment.
#
# Usage:  ./scripts/tensorboard.sh [--port PORT] [--logdir DIR]
#
# Defaults:  port=6006  logdir=models/runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT=6006
LOGDIR="$REPO_ROOT/models/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    --logdir) LOGDIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

TB_VENV="$REPO_ROOT/.tb-venv"

# Detect a stale venv (e.g. base Python upgraded out from under it) and rebuild.
if [[ -d "$TB_VENV" ]] && ! "$TB_VENV/bin/python" -c "import tensorboard" >/dev/null 2>&1; then
  echo "TensorBoard venv at $TB_VENV is stale; recreating ..."
  rm -rf "$TB_VENV"
fi

if [[ ! -d "$TB_VENV" ]]; then
  echo "Creating TensorBoard venv at $TB_VENV ..."
  if command -v uv >/dev/null 2>&1; then
    uv venv "$TB_VENV" -q
    VIRTUAL_ENV="$TB_VENV" uv pip install -q 'setuptools<81' tensorboard
  else
    python3 -m venv "$TB_VENV"
    "$TB_VENV/bin/pip" install --upgrade pip -q
    "$TB_VENV/bin/pip" install 'setuptools<81' tensorboard -q
  fi
  echo "TensorBoard venv ready."
fi

echo "Launching TensorBoard on port $PORT — logdir: $LOGDIR"
exec "$TB_VENV/bin/tensorboard" --logdir "$LOGDIR" --port "$PORT" --bind_all
