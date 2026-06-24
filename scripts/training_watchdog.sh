#!/usr/bin/env bash
# Monitor a running momentum_train job and log system/GPU health.
#
# Optionally sends SIGUSR1 on an interval so the trainer persists an emergency
# checkpoint (see CheckpointMixin._install_emergency_checkpoint_signal_handler).
#
# Usage:
#   ./scripts/training_watchdog.sh --pid <TRAIN_PID>
#   ./scripts/training_watchdog.sh                    # auto-detect training PID
#
# Options:
#   --pid PID                  Training process PID (default: auto-detect)
#   --health-interval MIN      Health log interval in minutes (default: 5)
#   --checkpoint-interval MIN  SIGUSR1 emergency-save interval; 0=disable (default: 20)
#   --model-dir DIR            Model directory for progress/checkpoint probes (default: models)
#   --log-file PATH            Watchdog log path (default: logs/watchdog.log)
#
# Run alongside training (separate terminal):
#   python -m momentum_train.run_training --resume
#   ./scripts/training_watchdog.sh
#
# Or use the wrapper:
#   ./scripts/run_training_with_watchdog.sh --resume

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_PID=""
HEALTH_INTERVAL_MIN=5
CHECKPOINT_INTERVAL_MIN=20
MODEL_DIR="$REPO_ROOT/models"
LOG_FILE="$REPO_ROOT/logs/watchdog.log"
HEARTBEAT_FILE="$REPO_ROOT/logs/watchdog.heartbeat"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid) TRAIN_PID="$2"; shift 2 ;;
    --health-interval) HEALTH_INTERVAL_MIN="$2"; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL_MIN="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$HEALTH_INTERVAL_MIN" -le 0 ]]; then
  echo "--health-interval must be > 0" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$HEARTBEAT_FILE")"

find_train_pid() {
  pgrep -f 'python -m momentum_train\.run_training' | head -1 || true
}

if [[ -z "$TRAIN_PID" ]]; then
  TRAIN_PID="$(find_train_pid)"
fi

if [[ -z "$TRAIN_PID" ]]; then
  echo "No training PID found. Start training first or pass --pid." >&2
  exit 1
fi

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  echo "PID $TRAIN_PID is not running." >&2
  exit 1
fi

json_escape() {
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  printf '%s' "$s"
}

log_event() {
  local event="$1"
  local extra="${2:-}"
  local ts
  ts="$(date -Is)"
  if [[ -n "$extra" ]]; then
    printf '{"ts":"%s","event":"%s","train_pid":%s,%s}\n' \
      "$(json_escape "$ts")" \
      "$(json_escape "$event")" \
      "$TRAIN_PID" \
      "$extra" >> "$LOG_FILE"
  else
    printf '{"ts":"%s","event":"%s","train_pid":%s}\n' \
      "$(json_escape "$ts")" \
      "$(json_escape "$event")" \
      "$TRAIN_PID" >> "$LOG_FILE"
  fi
  date -Is > "$HEARTBEAT_FILE"
}

read_loadavg() {
  awk '{print $1","$2","$3}' /proc/loadavg 2>/dev/null || echo ",,"
}

read_mem_mb() {
  awk '/MemTotal:/ {t=$2} /MemAvailable:/ {a=$2} END {printf "%d,%d", t/1024, a/1024}' /proc/meminfo 2>/dev/null || echo ","
}

read_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "null"
    return
  fi
  local line
  if ! line="$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)"; then
    echo "null"
    return
  fi
  line="${line// /}"
  IFS=',' read -r temp util mem_used mem_total power <<< "$line"
  printf '{"temp_c":%s,"util_pct":%s,"mem_mib":%s,"mem_total_mib":%s,"power_w":%s}' \
    "${temp:-null}" "${util:-null}" "${mem_used:-null}" "${mem_total:-null}" "${power:-null}"
}

read_progress() {
  local progress_file="$MODEL_DIR/progress.jsonl"
  if [[ ! -f "$progress_file" ]]; then
    echo "null"
    return
  fi
  local line
  line="$(tail -1 "$progress_file")"
  local ep steps ts_event
  ep="$(printf '%s' "$line" | sed -n 's/.*"episode": \([0-9]*\).*/\1/p')"
  steps="$(printf '%s' "$line" | sed -n 's/.*"total_steps": \([0-9]*\).*/\1/p')"
  ts_event="$(printf '%s' "$line" | sed -n 's/.*"ts": "\([^"]*\)".*/\1/p')"
  printf '{"episode":%s,"total_steps":%s,"progress_ts":"%s"}' \
    "${ep:-null}" "${steps:-null}" "$(json_escape "${ts_event:-}")"
}

read_latest_checkpoint() {
  local ckpt
  ckpt="$(ls -t "$MODEL_DIR"/checkpoint_trainer_latest_*.pt 2>/dev/null | head -1 || true)"
  if [[ -z "$ckpt" ]]; then
    echo "null"
    return
  fi
  local ep mtime
  ep="$(basename "$ckpt" | sed -n 's/.*_ep\([0-9]*\)_reward.*/\1/p')"
  mtime="$(date -Is -r "$ckpt" 2>/dev/null || stat -c '%y' "$ckpt" 2>/dev/null | cut -d. -f1)"
  printf '{"path":"%s","episode":%s,"mtime":"%s"}' \
    "$(json_escape "$ckpt")" "${ep:-null}" "$(json_escape "$mtime")"
}

read_kernel_alerts() {
  if ! command -v journalctl >/dev/null 2>&1; then
    echo "[]"
    return
  fi
  local alerts
  alerts="$(journalctl -k --since "10 min ago" --no-pager 2>/dev/null \
    | grep -iE 'Xid|Out of memory|Killed process|soft lockup|hard lockup|hung_task|blocked for more than|GPU has fallen off' \
    | grep -viE 'NVRM: loading|Listening on systemd-oomd' \
    | tail -3 \
    | sed 's/"/\\"/g' \
    | awk '{printf "\"%s\",", $0}' \
    | sed 's/,$//')"
  if [[ -z "$alerts" ]]; then
    echo "[]"
  else
    printf '[%s]' "$alerts"
  fi
}

emit_health() {
  local load mem gpu progress ckpt alerts alive
  load="$(read_loadavg)"
  mem="$(read_mem_mb)"
  gpu="$(read_gpu)"
  progress="$(read_progress)"
  ckpt="$(read_latest_checkpoint)"
  alerts="$(read_kernel_alerts)"

  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    alive=true
  else
    alive=false
  fi

  log_event "health" \
    "\"train_alive\":$alive,\"load_1_5_15\":\"$load\",\"mem_total_available_mib\":\"$mem\",\"gpu\":$gpu,\"progress\":$progress,\"latest_checkpoint\":$ckpt,\"kernel_alerts\":$alerts"
}

request_emergency_checkpoint() {
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    log_event "checkpoint_skipped" "\"reason\":\"train_pid_not_running\""
    return
  fi
  if kill -USR1 "$TRAIN_PID" 2>/dev/null; then
    log_event "checkpoint_requested" "\"signal\":\"SIGUSR1\""
  else
    log_event "checkpoint_failed" "\"reason\":\"kill_USR1_failed\""
  fi
}

HEALTH_SEC=$((HEALTH_INTERVAL_MIN * 60))
CHECKPOINT_SEC=$((CHECKPOINT_INTERVAL_MIN * 60))
NEXT_HEALTH=$SECONDS
NEXT_CHECKPOINT=$((SECONDS + CHECKPOINT_SEC))

log_event "watchdog_start" \
  "\"health_interval_min\":$HEALTH_INTERVAL_MIN,\"checkpoint_interval_min\":$CHECKPOINT_INTERVAL_MIN,\"model_dir\":\"$(json_escape "$MODEL_DIR")\",\"log_file\":\"$(json_escape "$LOG_FILE")\""

emit_health

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 10

  if (( SECONDS >= NEXT_HEALTH )); then
    emit_health
    NEXT_HEALTH=$((SECONDS + HEALTH_SEC))
  fi

  if (( CHECKPOINT_INTERVAL_MIN > 0 && SECONDS >= NEXT_CHECKPOINT )); then
    request_emergency_checkpoint
    NEXT_CHECKPOINT=$((SECONDS + CHECKPOINT_SEC))
  fi
done

log_event "watchdog_stop" "\"reason\":\"train_pid_exited\""
