"""
Event Logger
============
Writes every access decision to a timestamped CSV log file.

Log file location: data/logs/alpr_YYYY-MM-DD.csv
Columns: timestamp, plate, confidence, decision, owner, notes

A new file is created each day so logs stay manageable.
"""

import os
import csv
from datetime import datetime

import config


class EventLogger:
    """Appends access events to a daily CSV log file."""

    FIELDNAMES = ["timestamp", "plate", "confidence", "decision", "owner", "notes"]

    def __init__(self, log_dir: str = config.LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def log(self,
            plate:      str,
            confidence: float,
            decision:   str,
            owner:      str = "",
            notes:      str = "") -> None:
        """
        Append a single access event to today's log file.

        Parameters
        ----------
        plate      : normalised plate string
        confidence : aggregated confidence (0–1)
        decision   : 'ALLOWED' | 'DENIED' | 'UNCERTAIN'
        owner      : optional owner name from whitelist
        notes      : optional notes from whitelist or extra context
        """
        row = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "plate":      plate,
            "confidence": f"{confidence:.3f}",
            "decision":   decision,
            "owner":      owner,
            "notes":      notes,
        }

        log_path = self._today_path()
        file_exists = os.path.exists(log_path)

        with open(log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self._print_event(row)

    def recent(self, n: int = 20) -> list[dict]:
        """Return the last n events from today's log (newest first)."""
        log_path = self._today_path()
        if not os.path.exists(log_path):
            return []

        with open(log_path, newline="") as fh:
            rows = list(csv.DictReader(fh))

        return list(reversed(rows[-n:]))

    # ── Private helpers ────────────────────────────────────────────────────

    def _today_path(self) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"alpr_{date_str}.csv")

    @staticmethod
    def _print_event(row: dict) -> None:
        decision = row["decision"]
        symbol   = {"ALLOWED": "✓", "DENIED": "✗", "UNCERTAIN": "?"}.get(decision, "?")
        print(
            f"[{row['timestamp']}] {symbol} {decision:9s} | "
            f"plate={row['plate']:10s} conf={row['confidence']} "
            f"owner={row['owner']}"
        )
