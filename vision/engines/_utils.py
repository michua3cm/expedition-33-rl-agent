import cv2
import numpy as np


def _load_template_grey(
    path: str,
    hue_ranges: list[tuple[int, int]] | None,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Load a template as greyscale + optional colour mask.

    Returns ``(grey_img, mask)`` on success, ``None`` if the file cannot be
    read.  ``mask`` is a dilated uint8 binary array when ``hue_ranges`` is
    provided (restricts keypoints to coloured pixels), else ``None``.
    """
    if hue_ranges is not None:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in hue_ranges:
            mask |= cv2.inRange(hsv, (lo, 100, 200), (hi, 255, 255))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        mask = None
    return img, mask
