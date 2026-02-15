import cv2
import numpy as np

def match_target(name, template_data, img_gray, offset, overlay, logger):
    """
    Detects a specific target in the image.
    Updates the Overlay and the Logger directly.
    Returns: Max confidence score.
    """
    # 1. Matching
    res = cv2.matchTemplate(img_gray, template_data["image"], cv2.TM_CCOEFF_NORMED)
    
    # 2. Get Max Confidence
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # 3. Apply Specific Threshold
    loc = np.where(res >= template_data["threshold"])
    
    # 4. Extract Data
    w = template_data["w"]
    h = template_data["h"]
    color = template_data["color"]
    off_x, off_y = offset
    
    # 5. Process Matches
    # zip(*loc[::-1]) converts (y, x) arrays to (x, y) coordinates
    for pt in zip(*loc[::-1]):
        global_x = pt[0] + off_x
        global_y = pt[1] + off_y
        
        # Draw on UI
        overlay.draw_box(global_x, global_y, w, h, color, name)
        
        # Log Data
        logger.add_point(global_x, global_y, w, h, name)
            
    return max_val