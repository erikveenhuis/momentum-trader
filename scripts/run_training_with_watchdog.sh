#!/usr/bin/env bash
# Start training_watchdog.sh in the background, then run training in the foreground.
#
# Usage:
#   ./scripts/run_training_with_watchdog.sh --resume
#   ./scripts/run_training_with_watchdog.sh --health-interval 5 --checkpoint-interval 15 -- --resume
#
# Options before "--" are passed to training_watchdog.sh; everything after "--"
# goes to ``python -m momentum_train.run_training``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WATCHDOG_ARGS=()
TRAIN_ARGS=()

if [[ $# -eq 0 ]]; then
  TRAIN_ARGS=(--resume)
else
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      TRAIN_ARGS+=("$@")
      break
    fi
    WATCHDOG_ARGS+=("$1")
    shift
  done
fi

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  TRAIN_ARGS=(--resume)
fi

cd "$REPO_ROOT"

python -m momentum_train.run_training "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!

cleanup() {
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  wait "$TRAIN_PID"
  exit $?
fi

"$SCRIPT_DIR/training_watchdog.sh" --pid "$TRAIN_PID" "${WATCHDOG_ARGS[@]}" &
WATCHDOG_PID=$!

wait "$TRAIN_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
