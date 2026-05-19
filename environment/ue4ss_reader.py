"""
Reads live game state written by the UE4SS Lua mod (mods/StateReader).

The Lua mod serialises game state to a JSON file in the OS temp directory on
every tick.  This reader opens that file, parses one JSON object, and returns
a typed GameState dict.

File path used on each platform:
  Windows : %TEMP%\\expedition33_state.json
  Linux   : /tmp/expedition33_state.json   (for headless testing / dev)

Returns _DEFAULT_STATE silently when the file is missing or malformed so that
the environment degrades gracefully before UE4SS is installed.
"""

import json
import os
import sys
from typing import TypedDict

# ── Types ────────────────────────────────────────────────────────────────────


class GameState(TypedDict):
    player_hp: float
    player_hp_max: float
    enemy_hp: float
    enemy_hp_max: float
    player_ap: int          # discrete 0–9
    enemy_break: float
    enemy_break_max: float
    in_battle: bool
    is_offensive_phase: bool


_DEFAULT_STATE: GameState = {
    "player_hp": 0.0,
    "player_hp_max": 1.0,
    "enemy_hp": 0.0,
    "enemy_hp_max": 1.0,
    "player_ap": 0,
    "enemy_break": 0.0,
    "enemy_break_max": 1.0,
    "in_battle": False,
    "is_offensive_phase": False,
}

# ── Default path ─────────────────────────────────────────────────────────────


def _default_state_path() -> str:
    if sys.platform == "win32":
        tmp = os.environ.get("TEMP", os.environ.get("TMP", "C:\\Windows\\Temp"))
    else:
        tmp = os.environ.get("TMPDIR", "/tmp")
    return os.path.join(tmp, "expedition33_state.json")


STATE_PATH: str = _default_state_path()


# ── StateReader ───────────────────────────────────────────────────────────────


class StateReader:
    """
    Reads the latest game state from the file written by the UE4SS Lua mod.

    Args:
        path: Full path to the JSON state file.  Defaults to the OS temp file
              written by mods/StateReader/Scripts/main.lua.
    """

    def __init__(self, path: str = STATE_PATH):
        self._path = path

    def read(self) -> GameState:
        """
        Return the current game state.

        Returns _DEFAULT_STATE (all zeros / False) if the file is missing,
        unreadable, or contains invalid JSON.
        """
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.loads(f.read())
            return _merge_defaults(data)
        except (OSError, json.JSONDecodeError):
            return dict(_DEFAULT_STATE)

    def is_available(self) -> bool:
        """True if the UE4SS mod has written at least one state file."""
        return os.path.isfile(self._path)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _merge_defaults(data: dict) -> GameState:
    """Fill missing keys with defaults so downstream code never KeyErrors."""
    result = dict(_DEFAULT_STATE)
    result.update({k: v for k, v in data.items() if k in _DEFAULT_STATE})
    return result  # type: ignore[return-value]
