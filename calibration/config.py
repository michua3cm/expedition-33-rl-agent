import os

# Paths
ASSETS_DIR = 'assets'
LOG_DIR = os.path.join('data', 'logs')
SCREENSHOT_DIR = os.path.join('data', 'screenshots')
YOLO_RAW_DIR = os.path.join('data', 'yolo_dataset', 'images', 'raw')
YOLO_MODEL_PATH = os.path.join('data', 'yolo_dataset', 'train', 'weights', 'best.pt')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(YOLO_RAW_DIR, exist_ok=True)

# Global Settings
DEFAULT_THRESHOLD = 0.6
DEFAULT_MIN_MATCHES = 12

# 0: All monitors
# 1: monitor 1
# 2: monitor 2 ... and so on
MONITOR_INDEX = 1

# --- TARGETS DEFINITION ---
TARGETS = {
    "PERFECT": {
        "file": "template_perfect.png",
        "color": "lime",
        "threshold": 0.65,  # Text needs lower threshold
        "min_matches": 10
    },
    "DODGE": {
        "file": "template_dodge.png",
        "color": "yellow",
        "threshold": 0.65,
        "min_matches": 10
    },
    "PARRIED": {
        "file": "template_parried.png",
        "color": "cyan",
        "threshold": 0.65,
        "min_matches": 10
    },
    "JUMP": {
        "file": "template_jump.png",
        "color": "magenta",
        "threshold": 0.75,
        "min_matches": 15
    },
    "MOUSE": {
        "file": "template_mouse.png",
        "color": "orange",
        "threshold": 0.90,  # Icon needs strict threshold — high precision for auto-labeling
        "min_matches": 10
    }
}