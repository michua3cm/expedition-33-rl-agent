import os

# Paths
ASSETS_DIR = 'assets'
LOG_DIR = os.path.join('data', 'logs')
SCREENSHOT_DIR = os.path.join('data', 'screenshots')
YOLO_RAW_DIR = os.path.join('data', 'yolo_dataset', 'images', 'raw')
YOLO_MODEL_PATH = os.path.join('data', 'yolo_dataset', 'train', 'weights', 'best.pt')
DEMO_DIR = os.path.join('data', 'demos')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(YOLO_RAW_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

# Global Settings
DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_MATCHES = 12

# 0: All monitors
# 1: monitor 1
# 2: monitor 2 ... and so on
MONITOR_INDEX = 1

# --- TARGETS DEFINITION ---
#
# Three categories:
#   Reward signals  — text/icon confirming a successful defensive action
#   Action cues     — icon/effect prompting the player to act
#   Phase signals   — UI elements indicating the current combat phase
#
# NOTE: template files for the 5 new targets (JUMP_CUE, BATTLE_WHEEL,
# TURN_ALLY, TURN_ENEMY, GRADIENT_INCOMING) must be collected in-game
# before PIXEL/SIFT/ORB engines can use them.
#
TARGETS = {
    # ------------------------------------------------------------------ #
    # Reward Signals                                                       #
    # Text that appears on screen confirming a successful action.          #
    # ------------------------------------------------------------------ #
    "PERFECT": {
        # Reward +10: perfect dodge achieved (tight timing window).
        # Used to train parry timing — PERFECT dodge window ≈ parry window.
        "file": "template_perfect.png",
        "color": "lime",
        "threshold": 0.65,
        "min_matches": 10
    },
    "DODGE": {
        # Reward +5: successful dodge (any timing).
        "file": "template_dodge.png",
        "color": "yellow",
        "threshold": 0.65,
        "min_matches": 10
    },
    "JUMP": {
        # Reward +5: text confirming a successful jump.
        # NOT the action cue — see JUMP_CUE for the prompt icon.
        "file": "template_jump.png",
        "color": "magenta",
        "threshold": 0.75,
        "min_matches": 15
    },
    "PARRIED": {
        # Reward +8: text confirming a successful gradient attack parry.
        # Normal attack parries produce no visual signal.
        "file": "template_parried.png",
        "color": "cyan",
        "threshold": 0.65,
        "min_matches": 10
    },

    # ------------------------------------------------------------------ #
    # Action Cues                                                          #
    # Icons/effects that prompt the player to take a specific action.      #
    # ------------------------------------------------------------------ #
    "JUMP_CUE": {
        # Action cue: golden starburst/cross icon on screen — this attack
        # can only be avoided by jumping (SPACE). Collect template in-game.
        "file": "template_jump_cue.png",
        "color": "gold",
        "threshold": 0.70,
        "min_matches": 10
    },
    "MOUSE": {
        # Action cue: mouse cursor icon — appears after a successful jump,
        # signals the jump attack window is open (left click to counter).
        "file": "template_mouse.png",
        "color": "orange",
        "threshold": 0.90,
        "min_matches": 10
    },

    # ------------------------------------------------------------------ #
    # Phase Signals                                                        #
    # UI elements that identify the current combat phase.                  #
    # Collect all templates in-game.                                       #
    # ------------------------------------------------------------------ #
    "BATTLE_WHEEL": {
        # Phase signal: circular action menu visible during the player's
        # selection turn. When present: offensive phase, no defense needed.
        "file": "template_battle_wheel.png",
        "color": "white",
        "threshold": 0.75,
        "min_matches": 12
    },
    "TURN_ALLY": {
        # Phase signal: the top (active) card in the turn order UI has blue
        # border lines around it — a player character is currently acting.
        # Template: crop of the blue border strip only, no portrait.
        # color_mode=True: blue vs red is invisible in greyscale.
        "file": "template_turn_ally.png",
        "color": "blue",
        "color_mode": True,
        "threshold": 0.75,
        "min_matches": 12
    },
    "TURN_ENEMY": {
        # Phase signal: the top (active) card in the turn order UI has red
        # border lines around it — an enemy is currently acting. React now.
        # Template: crop of the red border strip only, no portrait.
        # color_mode=True: blue vs red is invisible in greyscale.
        "file": "template_turn_enemy.png",
        "color": "red",
        "color_mode": True,
        "threshold": 0.75,
        "min_matches": 12
    },
    "GRADIENT_INCOMING": {
        # Phase signal: grey screen overlay — enemy is launching a gradient
        # attack. Only occurs during TURN_ENEMY. Response: GRADIENT_PARRY (W).
        "file": "template_gradient_incoming.png",
        "color": "purple",
        "threshold": 0.70,
        "min_matches": 10
    },
}
