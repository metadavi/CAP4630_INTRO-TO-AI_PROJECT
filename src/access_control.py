"""
Access Control
==============
Compares a voted plate reading against the authorised whitelist and returns
one of three decisions:

    ALLOWED   — plate is on the whitelist AND confidence ≥ VOTE_CONF_THRESHOLD
    DENIED    — plate is NOT on the whitelist AND confidence ≥ UNCERTAIN_THRESHOLD
    UNCERTAIN — confidence is too low to make a reliable call

The whitelist is a CSV file with (at minimum) a 'plate' column.
Optional columns: 'owner', 'notes', 'expires'.
"""

from __future__ import annotations

import os
from datetime import date
from enum import Enum

import pandas as pd

import config


class Decision(str, Enum):
    ALLOWED   = "ALLOWED"
    DENIED    = "DENIED"
    UNCERTAIN = "UNCERTAIN"


class AccessController:
    """
    Loads the whitelist CSV and evaluates plate decisions.

    The whitelist is reloaded each time `check()` is called only if the
    file modification time has changed, so hot-edits to the CSV take effect
    without restarting the process.
    """

    def __init__(self, whitelist_path: str = config.WHITELIST_PATH):
        self.whitelist_path = whitelist_path
        self._mtime: float | None = None
        self._plates: set[str]    = set()
        self._df: pd.DataFrame    = pd.DataFrame()
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def check(self,
              plate_text: str,
              confidence: float) -> tuple[Decision, dict]:
        """
        Evaluate a plate reading.

        Parameters
        ----------
        plate_text  : normalised plate string, e.g. 'ABC1234'
        confidence  : aggregated confidence from FrameVoter (0–1)

        Returns
        -------
        (decision, info_dict)
            decision  — Decision enum value
            info_dict — extra details (owner, notes, expires, confidence)
        """
        self._reload_if_stale()

        plate = plate_text.upper().strip()
        info  = {"plate": plate, "confidence": round(confidence, 3)}

        # Too uncertain to decide
        if confidence < config.UNCERTAIN_THRESHOLD:
            return Decision.UNCERTAIN, info

        # Look up in whitelist
        if plate in self._plates:
            row = self._df[self._df["plate"] == plate].iloc[0]
            info["owner"]  = row.get("owner",  "")
            info["notes"]  = row.get("notes",  "")
            info["expires"] = str(row.get("expires", ""))

            # Check expiry if column exists
            if "expires" in self._df.columns and info["expires"]:
                try:
                    exp = date.fromisoformat(info["expires"])
                    if exp < date.today():
                        info["reason"] = "expired"
                        return Decision.DENIED, info
                except ValueError:
                    pass  # non-date value → ignore expiry check

            if confidence >= config.VOTE_CONF_THRESHOLD:
                return Decision.ALLOWED, info
            else:
                # On whitelist but confidence not high enough
                return Decision.UNCERTAIN, info

        # Not on whitelist
        return Decision.DENIED, info

    def add_plate(self, plate: str, owner: str = "", notes: str = "",
                  expires: str = "") -> None:
        """Add a plate to the whitelist and persist to CSV."""
        self._reload_if_stale()
        plate = plate.upper().strip()
        if plate in self._plates:
            return

        new_row = pd.DataFrame([{
            "plate":   plate,
            "owner":   owner,
            "notes":   notes,
            "expires": expires,
        }])
        self._df     = pd.concat([self._df, new_row], ignore_index=True)
        self._plates.add(plate)
        self._save()

    def remove_plate(self, plate: str) -> bool:
        """Remove a plate from the whitelist. Returns True if it existed."""
        self._reload_if_stale()
        plate = plate.upper().strip()
        if plate not in self._plates:
            return False

        self._df     = self._df[self._df["plate"] != plate].reset_index(drop=True)
        self._plates.discard(plate)
        self._save()
        return True

    # ── Private helpers ────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load (or initialise) the whitelist CSV."""
        if not os.path.exists(self.whitelist_path):
            self._df     = pd.DataFrame(columns=["plate", "owner", "notes", "expires"])
            self._plates = set()
            self._save()
            return

        self._df     = pd.read_csv(self.whitelist_path, dtype=str).fillna("")
        self._df["plate"] = self._df["plate"].str.upper().str.strip()
        self._plates = set(self._df["plate"].tolist())
        self._mtime  = os.path.getmtime(self.whitelist_path)

    def _reload_if_stale(self) -> None:
        if not os.path.exists(self.whitelist_path):
            return
        mtime = os.path.getmtime(self.whitelist_path)
        if mtime != self._mtime:
            self._load()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.whitelist_path), exist_ok=True)
        self._df.to_csv(self.whitelist_path, index=False)
        if os.path.exists(self.whitelist_path):
            self._mtime = os.path.getmtime(self.whitelist_path)
