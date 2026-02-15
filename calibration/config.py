import os

# Paths
ASSETS_DIR = 'assets'
LOG_DIR = os.path.join('data', 'logs')
SCREENSHOT_DIR = os.path.join('data', 'screenshots')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Global Settings
DEFAULT_THRESHOLD = 0.6

# 0: All monitors
# 1: monitor 1
# 2: monitor 2 ... and so on
MONITOR_INDEX = 1

# --- TARGETS DEFINITION ---
TARGETS = {
    "PERFECT": {
        "file": "template_perfect.png",
        "color": "lime",
        "threshold": 0.65  # Text needs lower threshold
    },
    "DODGE": {
        "file": "template_dodge.png",
        "color": "yellow",
        "threshold": 0.65
    },
    "PARRIED": {
        "file": "template_parried.png",
        "color": "cyan",
        "threshold": 0.65
    },
    "JUMP": {
        "file": "template_jump.png",
        "color": "magenta",
        "threshold": 0.75
    },
    # "MOUSE": {
    #     "file": "template_mouse.png",
    #     "color": "orange",
    #     "threshold": 0.90  # Icon needs strict threshold
    # }
}