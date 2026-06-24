#!/usr/bin/env bash
# Launch TensorBoard in a separate terminal, then resume training in this one.
#
# Usage:  ./train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

launch_in_terminal() {
  if command -v xdg-terminal-exec >/dev/null 2>&1; then
    xdg-terminal-exec --dir="$REPO_ROOT" -- ./scripts/tensorboard.sh
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator --new-window --working-directory="$REPO_ROOT" -- ./scripts/tensorboard.sh &
  elif command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --working-directory="$REPO_ROOT" -- ./scripts/tensorboard.sh
  elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --working-directory="$REPO_ROOT" --hold -e ./scripts/tensorboard.sh &
  elif command -v konsole >/dev/null 2>&1; then
    konsole --workdir "$REPO_ROOT" --hold -e ./scripts/tensorboard.sh &
  elif command -v xterm >/dev/null 2>&1; then
    xterm -hold -e bash -lc "cd $(printf '%q' "$REPO_ROOT") && ./scripts/tensorboard.sh" &
  else
    echo "No supported terminal emulator found; running TensorBoard in the background." >&2
    ./scripts/tensorboard.sh &
  fi
}

launch_in_terminal

# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
exec python -m momentum_train.run_training --resume
