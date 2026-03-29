from __future__ import annotations

import os

import cv2
import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import register

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
            path = os.path.join(assets_dir, file_name)
            if not os.path.exists(path):
                print(f"[ORBEngine] Warning: '{file_name}' not found, skipping '{label}'.")
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[ORBEngine] Error: failed to load '{file_name}'.")
                continue
            kp, des = self._orb.detectAndCompute(img, None)
            if des is None:
                print(f"[ORBEngine] Warning: no keypoints found in '{cfg['file']}', skipping '{label}'.")
                continue
            min_matches = cfg.get("min_matches", _DEFAULT_MIN_MATCHES)
            self._templates[label] = {
                "image": img,
                "w": img.shape[1],
                "h": img.shape[0],
                "kp": kp,
                "des": des,
                "min_matches": min_matches,
            }
            print(f"[ORBEngine] Loaded '{label}' (min_matches={min_matches})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        live_kp, live_des = self._orb.detectAndCompute(frame, None)
        if live_des is None or len(live_des) < 2:
            return []

        results: list[Detection] = []
        for label, data in self._templates.items():
            matches = self._matcher.knnMatch(data["des"], live_des, k=2)

            # Lowe's ratio test — guard against pairs with fewer than 2 neighbours
            good = [
                m for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]

            if len(good) < data["min_matches"]:
                continue

            src_pts = np.float32([data["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is None:
                continue

            h, w = data["h"], data["w"]
            corners = np.float32([
                [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]
            ]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(corners, M)

            xs = [int(p[0][0]) for p in dst]
            ys = [int(p[0][1]) for p in dst]

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
