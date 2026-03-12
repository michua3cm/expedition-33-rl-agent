import cv2
import numpy as np

# Initialize FLANN matcher globally for speed
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def match_target(name, template_data, img_gray, offset, overlay, logger, frame_data=None):
    """
    Detects a target using Scale-Invariant Feature Transform (SIFT).
    Returns the number of good matches found.
    """
    if not frame_data:
        return 0
        
    des_template = template_data["des"]
    kp_template = template_data["kp"]
    min_matches = template_data["min_matches"]
    color = template_data["color"]
    off_x, off_y = offset
    
    kp_live = frame_data.get("kp")
    des_live = frame_data.get("des")

    if des_template is None or des_live is None or len(des_live) < 2:
        return 0

    # Match descriptors
    matches = flann.knnMatch(des_template, des_live, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= min_matches:
        src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find mapping matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h_temp, w_temp = template_data["h"], template_data["w"]
            pts = np.float32([[0, 0], [0, h_temp - 1], [w_temp - 1, h_temp - 1], [w_temp - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            x_coords = [int(pt[0][0]) for pt in dst]
            y_coords = [int(pt[0][1]) for pt in dst]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            found_w = max_x - min_x
            found_h = max_y - min_y

            global_x = min_x + off_x
            global_y = min_y + off_y
            
            overlay.draw_box(global_x, global_y, found_w, found_h, color, name)
            logger.add_point(global_x, global_y, found_w, found_h, name)
            
    return len(good_matches)