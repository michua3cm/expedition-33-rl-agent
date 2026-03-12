import cv2
import os
from .config import ASSETS_DIR, DEFAULT_THRESHOLD, DEFAULT_MIN_MATCHES

def load_templates(target_config, engine="PIXEL"):
    """
    Loads templates based on the configuration dictionary and selected engine.
    """
    templates = {}
    print(f"[Loader] Loading assets from: {ASSETS_DIR}")

    engine = engine.upper()
    # Initialize SIFT locally just for loading templates if needed
    sift = cv2.SIFT_create() if engine == "SIFT" else None
    
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
        
        # Base data
        templates[name] = {
            "image": img,
            "w": img.shape[1],
            "h": img.shape[0],
            "color": config["color"]
        }

        # Engine specific data
        if engine == "PIXEL":
            templates[name]["threshold"] = config.get("threshold", DEFAULT_THRESHOLD)
            print(f" -> Loaded {name} (Threshold: {templates[name]['threshold']})")
            
        elif engine == "SIFT":
            kp, des = sift.detectAndCompute(img, None)
            templates[name]["kp"] = kp
            templates[name]["des"] = des
            templates[name]["min_matches"] = config.get("min_matches", DEFAULT_MIN_MATCHES)
            print(f" -> Loaded {name} (Min Matches: {templates[name]['min_matches']})")
        
    if not templates:
        print("[Critical] No templates loaded. Exiting.")
        exit()
        
    return templates