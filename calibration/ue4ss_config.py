"""
UE4SS Blueprint class names and property names for Expedition 33.

These values must be discovered by running UE4SS Live Property Viewer inside
an active battle.  Replace every "TODO" string below with the real names.

How to discover them
--------------------
1. Install UE4SS (Nexus Mods mod 630 for Expedition 33) into
   ...\\Expedition 33\\Sandfall\\Binaries\\Win64\\
2. Launch the game and enter a battle.
3. Open the UE4SS console (default hotkey: F10 → UE4SS tab).
4. In the Live Property Viewer, search for class names containing:
   "Battle", "Combat", "Character", "Enemy", "Turn"
5. Expand matching instances to read their property names and live values.
6. Note the exact Blueprint class names and property names below.

Example (actual names will differ — do NOT copy this verbatim):
  PLAYER_CLASS = "BP_BattleCharacter_C"
  PLAYER_HP_PROPERTY = "CurrentHP"
"""

# ── Player character ──────────────────────────────────────────────────────────
PLAYER_CLASS: str = "TODO"           # e.g. "BP_BattleCharacter_C"
PLAYER_HP_PROPERTY: str = "TODO"     # e.g. "CurrentHP"
PLAYER_HP_MAX_PROPERTY: str = "TODO" # e.g. "MaxHP"
PLAYER_AP_PROPERTY: str = "TODO"     # Action Points, integer 0–9

# ── Enemy ─────────────────────────────────────────────────────────────────────
ENEMY_CLASS: str = "TODO"            # e.g. "BP_EnemyBase_C"
ENEMY_HP_PROPERTY: str = "TODO"
ENEMY_HP_MAX_PROPERTY: str = "TODO"
ENEMY_BREAK_PROPERTY: str = "TODO"   # break / stun meter fill
ENEMY_BREAK_MAX_PROPERTY: str = "TODO"

# ── Battle state ──────────────────────────────────────────────────────────────
BATTLE_CONTROLLER_CLASS: str = "TODO"   # e.g. "BP_BattleController_C"
IN_BATTLE_PROPERTY: str = "TODO"        # bool
IS_OFFENSIVE_PHASE_PROPERTY: str = "TODO"  # bool: True = player's action turn

# ── State file path (must match mods/StateReader/Scripts/main.lua OUTPUT_FILE)
# Leave as empty string to use the platform default (%TEMP%\expedition33_state.json).
STATE_FILE_OVERRIDE: str = ""
