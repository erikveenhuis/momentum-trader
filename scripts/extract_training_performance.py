import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_ROOT = Path("logs")
REPORTS_ROOT = Path("reports")

# Regex patterns for parsing logs
EPISODE_START_RE = re.compile(r"--- Starting Episode (\d+)/(\d+) using file: (.+) ---")
EPISODE_END_RE = re.compile(r"Episode (\d+): Ended\.")
TS_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([^-]+) - ([^-]+) - (.*)$")
STEPS_RE = re.compile(r"Steps: (\d+)")
REWARD_RE = re.compile(r"Reward: (-?\d+\.\d+)")
AVG_LOSS_RE = re.compile(r"Avg Loss: (\d+\.\d+)")


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime object."""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")


def _ordered_training_logs() -> List[Path]:
    """Get all training log files in chronological order."""
    if not LOG_ROOT.exists():
        return []
    logs = []
    for path in LOG_ROOT.glob("training.log*"):
        name = path.name
        if name == "training.log":
            order = 0
        else:
            match = re.match(r"training\.log\.(\d+)$", name)
            order = int(match.group(1)) if match else 10_000
        logs.append((order, path))
    logs.sort(key=lambda pair: (pair[0], pair[1].name))
    return [path for _, path in logs]


def _write_csv(path: Path, rows: List[Dict[str, Optional[str]]], fieldnames: List[str]) -> None:
    """Write rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def extract_training_performance() -> None:
    """Extract training performance metrics from logs."""
    performance_data: List[Dict[str, Optional[str]]] = []
    episode_starts: Dict[int, Tuple[str, str]] = {}  # episode_num -> (timestamp, dataset_file)

    for log_path in _ordered_training_logs():
        current_episode_start: Optional[str] = None
        current_dataset: Optional[str] = None

        with log_path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n\r")
                if not line:
                    continue

                message = line
                timestamp_match = TS_LINE_RE.match(line)
                current_timestamp = None
                if timestamp_match:
                    current_timestamp = timestamp_match.group(1)
                    message = timestamp_match.group(4)
                else:
                    message = line.strip()

                normalized_message = message.strip()

                # Check for episode start
                start_match = EPISODE_START_RE.match(normalized_message)
                if start_match:
                    episode_num = int(start_match.group(1))
                    dataset_file = start_match.group(3).strip()
                    if current_timestamp:
                        episode_starts[episode_num] = (current_timestamp, dataset_file)
                        current_episode_start = current_timestamp
                        current_dataset = dataset_file
                    continue

                # Check for episode end
                end_match = EPISODE_END_RE.match(normalized_message)
                if end_match:
                    episode_num = int(end_match.group(1))
                    if episode_num in episode_starts and current_timestamp:
                        start_time_str, dataset_file = episode_starts[episode_num]
                        end_time_str = current_timestamp

                        try:
                            start_time = parse_timestamp(start_time_str)
                            end_time = parse_timestamp(end_time_str)
                            duration_seconds = (end_time - start_time).total_seconds()

                            # Extract additional metrics from the following lines
                            episode_data = {
                                "source_log": log_path.name,
                                "episode": str(episode_num),
                                "dataset_file": dataset_file,
                                "start_timestamp": start_time_str,
                                "end_timestamp": end_time_str,
                                "duration_seconds": f"{duration_seconds:.3f}",
                                "duration_minutes": f"{duration_seconds / 60:.3f}",
                                "steps": None,
                                "reward": None,
                                "avg_loss": None,
                            }

                            performance_data.append(episode_data)

                            # Clear the start data
                            del episode_starts[episode_num]

                        except ValueError as e:
                            print(f"Error parsing timestamps for episode {episode_num}: {e}")
                            continue

                    # Reset current episode tracking
                    current_episode_start = None
                    current_dataset = None
                    continue

                # Extract additional metrics for the current episode being processed
                if performance_data and performance_data[-1]["episode"] == str(episode_num if "episode_num" in locals() else 0):
                    if "Steps:" in normalized_message:
                        steps_match = STEPS_RE.search(normalized_message)
                        if steps_match:
                            performance_data[-1]["steps"] = steps_match.group(1)
                    elif "Reward:" in normalized_message:
                        reward_match = REWARD_RE.search(normalized_message)
                        if reward_match:
                            performance_data[-1]["reward"] = reward_match.group(1)
                    elif "Avg Loss:" in normalized_message:
                        loss_match = AVG_LOSS_RE.search(normalized_message)
                        if loss_match:
                            performance_data[-1]["avg_loss"] = loss_match.group(1)

    # Sort by end timestamp
    performance_data.sort(key=lambda x: x.get("end_timestamp") or "")

    # Write to CSV
    fieldnames = [
        "source_log",
        "episode",
        "dataset_file",
        "start_timestamp",
        "end_timestamp",
        "duration_seconds",
        "duration_minutes",
        "steps",
        "reward",
        "avg_loss",
    ]

    _write_csv(REPORTS_ROOT / "training_performance.csv", performance_data, fieldnames)

    # Print summary statistics
    if performance_data:
        durations = [float(row["duration_seconds"]) for row in performance_data if row["duration_seconds"]]
        if durations:
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)

            print(f"Average episode duration: {avg_duration:.3f} seconds ({avg_duration/60:.3f} minutes)")
            print(f"Min episode duration: {min_duration:.3f} seconds")
            print(f"Max episode duration: {max_duration:.3f} seconds")


if __name__ == "__main__":
    extract_training_performance()
