import cv2
import os
from .config import ASSETS_DIR, DEFAULT_THRESHOLD

def load_templates(target_config):
    """
    Loads templates based on the configuration dictionary.
    Returns a dictionary of processed template data.
    """
    templates = {}
    print(f"[Loader] Loading assets from: {ASSETS_DIR}")
    
    for name, config in target_config.items():
        path = os.path.join(ASSETS_DIR, config['file'])
        
        # Check if file exists to prevent crash
        if not os.path.exists(path):
            print(f"[Warning] File not found: {config['file']}. Skipping '{name}'.")
            continue
        
        img = cv2.imread(path, 0)
        if img is None:
            print(f"[Error] Failed to load {config['file']}")
            continue
        
        # Store image data and dimensions
        templates[name] = {
            "image": img,
            "w": img.shape[1],
            "h": img.shape[0],
            "color": config["color"],
            "threshold": config.get("threshold", DEFAULT_THRESHOLD) # Default if missing
        }
        print(f" -> Loaded {name} (Threshold: {templates[name]['threshold']})")
        
    if not templates:
        print("[Critical] No templates loaded. Exiting.")
        exit()
        
    return templates