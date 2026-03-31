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
MONITOR_INDEX = 0

# --- TARGETS DEFINITION ---
#
# Three categories:
#   Reward signals  — text/icon confirming a successful defensive action
#   Action cues     — icon/effect prompting the player to act
#   Phase signals   — UI elements indicating the current combat phase
#
# NOTE: template files for 4 targets (JUMP_CUE, BATTLE_WHEEL, TURN_ALLY,
# TURN_ENEMY) must be collected in-game and saved to assets/ before
# PIXEL/SIFT/ORB engines can use them.
# GRADIENT_INCOMING uses frame-wide HSV saturation (no template file needed).
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
        #
        # engine / autolabel_engine: "SIFT" — PIXEL cannot match this target.
        # The icon animates (shrinks → grows → fades) and its size varies
        # with camera distance per enemy attack, so template matching at a
        # fixed pixel size will never reliably match it. SIFT is scale- and
        # rotation-invariant and can detect the icon across all size variants
        # from a single template crop. See docs/PROJECT_STRUCTURE.md for the
        # full design rationale.
        "file": "template_jump_cue.png",
        "color": "gold",
        "color_mask": True,
        "threshold": 0.70,
        "min_matches": 10,
        "engine": "SIFT",
        "autolabel_engine": "SIFT",
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
        # engine: "SIFT" — button positions shift per scenario; Attack button
        # crop is the stable anchor. SIFT handles the layout variation.
        "file": "template_battle_wheel_attack.png",
        "color": "white",
        "threshold": 0.75,
        "min_matches": 12,
        "engine": "SIFT",
        "autolabel_engine": "SIFT",
    },
    "TURN_ALLY": {
        # Phase signal: the top (active) card in the turn order UI has a gold
        # border around it — a player character is currently acting.
        # Template: crop of the gold border strip only, no portrait.
        # color_mode=True: gold vs red is invisible in greyscale.
        # color: "gold" — RGB (167,137,66), hue ~42° standard / ~21 OpenCV HSV.
        "file": "template_turn_ally.png",
        "color": "gold",
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
        # Detected by frame-wide saturation drop (no template file).
        # hsv_sat_max: mean S-channel (0–255) threshold; tune via calibration.
        "file": None,
        "color": "purple",
        "hsv_sat_max": 40,
        "min_matches": 10
    },
}
