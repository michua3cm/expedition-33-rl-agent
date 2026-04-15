from __future__ import annotations

import os

import cv2
import numpy as np

from ..engine import HUE_RANGES, Detection, VisionEngine, apply_roi
from ..registry import register
from ._utils import _load_template_grey

_DEFAULT_MIN_MATCHES = 12


@register("ORB")
class ORBEngine(VisionEngine):
    """
    ORB (Oriented FAST and Rotated BRIEF) feature matching.

    Faster than SIFT, patent-free, and rotation invariant.
    The standard choice for real-time robotics pipelines (ORB-SLAM2/3).

    Uses BFMatcher with Hamming distance — the correct pairing for binary
    ORB descriptors (FLANN/KDTree is for float descriptors like SIFT).

    Ratio test threshold is 0.75 (vs 0.7 for SIFT) because binary Hamming
    distances are less smooth, requiring a slightly more lenient threshold.

    Confidence is normalised: min_matches → 0.5, 2×min_matches → 1.0.
    """

    def __init__(self) -> None:
        self._orb = cv2.ORB_create()
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._templates: dict = {}

    @property
    def name(self) -> str:
        return "ORB"

    def load(self, targets: dict, assets_dir: str) -> None:
        self._templates.clear()
        for label, cfg in targets.items():
            file_name = cfg.get("file")
            if file_name is None:
                print(f"[ORBEngine] Skipping '{label}' (no template file — not supported by ORB).")
                continue

            # Normalise: a single filename string is treated as a one-item list.
            file_names: list[str] = [file_name] if isinstance(file_name, str) else list(file_name)
            # Build a color mask when requested so that keypoints are only
            # detected on the colored icon pixels, not in hollow background
            # regions (mirrors the same fix in SIFTEngine).
            hue_ranges = HUE_RANGES.get(cfg.get("color", "")) if cfg.get("color_mask") else None
            min_matches = cfg.get("min_matches", _DEFAULT_MIN_MATCHES)

            variants: list[dict] = []
            for fname in file_names:
                path = os.path.join(assets_dir, fname)
                if not os.path.exists(path):
                    print(f"[ORBEngine] Warning: '{fname}' not found, skipping for '{label}'.")
                    continue
                result = _load_template_grey(path, hue_ranges)
                if result is None:
                    print(f"[ORBEngine] Error: failed to load '{fname}'.")
                    continue
                img, mask = result

                kp, des = self._orb.detectAndCompute(img, mask)
                if des is None:
                    print(f"[ORBEngine] Warning: no keypoints found in '{fname}', skipping for '{label}'.")
                    continue
                variants.append({
                    "image": img,
                    "w": img.shape[1],
                    "h": img.shape[0],
                    "kp": kp,
                    "des": des,
                })

            if not variants:
                continue

            self._templates[label] = {
                "variants": variants,
                "min_matches": min_matches,
                "roi": cfg.get("roi"),
            }
            print(f"[ORBEngine] Loaded '{label}' ({len(variants)} variant(s), min_matches={min_matches})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results: list[Detection] = []

        for label, data in self._templates.items():
            roi_frame, off_x, off_y = apply_roi(frame, data.get("roi"))
            live_kp, live_des = self._orb.detectAndCompute(roi_frame, None)
            if live_des is None or len(live_des) < 2:
                continue

            for variant in data["variants"]:
                matches = self._matcher.knnMatch(variant["des"], live_des, k=2)

                # Lowe's ratio test — guard against pairs with fewer than 2 neighbours
                good = [
                    m for pair in matches
                    if len(pair) == 2
                    for m, n in [pair]
                    if m.distance < 0.75 * n.distance
                ]

                if len(good) < data["min_matches"]:
                    continue

                src_pts = np.float32([variant["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is None:
                    continue

                h, w = variant["h"], variant["w"]
                corners = np.float32([
                    [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]
                ]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(corners, M)

                xs = [int(p[0][0]) + off_x for p in dst]
                ys = [int(p[0][1]) + off_y for p in dst]

                # Normalise: min_matches → 0.5, 2×min_matches → 1.0
                confidence = min(len(good) / (2.0 * data["min_matches"]), 1.0)

                results.append(Detection(
                    label=label,
                    x=min(xs),
                    y=min(ys),
                    w=max(xs) - min(xs),
                    h=max(ys) - min(ys),
                    confidence=confidence,
                ))

        return results
