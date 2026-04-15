import os

import cv2
import numpy as np

from ..engine import HUE_RANGES, Detection, VisionEngine, apply_roi
from ..registry import register
from ._utils import _load_template_grey

_DEFAULT_MIN_MATCHES = 12


@register("SIFT")
class SIFTEngine(VisionEngine):
    """
    SIFT feature matching with FLANN + Lowe's ratio test.
    Scale and rotation invariant; slower than PIXEL but more robust.

    Confidence is normalised: min_matches → 0.5, 2×min_matches → 1.0.
    """

    def __init__(self) -> None:
        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5},
            {"checks": 50},
        )
        self._templates: dict = {}

    @property
    def name(self) -> str:
        return "SIFT"

    def load(self, targets: dict, assets_dir: str) -> None:
        self._templates.clear()
        for label, cfg in targets.items():
            file_name = cfg.get("file")
            if file_name is None:
                print(f"[SIFTEngine] Skipping '{label}' (no template file — not supported by SIFT).")
                continue

            # Normalise: a single filename string is treated as a one-item list.
            file_names: list[str] = [file_name] if isinstance(file_name, str) else list(file_name)
            # Build a color mask when requested so that keypoints are only
            # detected on the colored icon pixels, not in hollow background
            # regions (e.g. the interior of the JUMP_CUE pound-sign shape).
            hue_ranges = HUE_RANGES.get(cfg.get("color", "")) if cfg.get("color_mask") else None
            min_matches = cfg.get("min_matches", _DEFAULT_MIN_MATCHES)

            variants: list[dict] = []
            for fname in file_names:
                path = os.path.join(assets_dir, fname)
                if not os.path.exists(path):
                    print(f"[SIFTEngine] Warning: '{fname}' not found, skipping for '{label}'.")
                    continue
                result = _load_template_grey(path, hue_ranges)
                if result is None:
                    print(f"[SIFTEngine] Error: failed to load '{fname}'.")
                    continue
                img, mask = result

                kp, des = self._sift.detectAndCompute(img, mask)
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
            print(f"[SIFTEngine] Loaded '{label}' ({len(variants)} variant(s), min_matches={min_matches})")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results: list[Detection] = []

        for label, data in self._templates.items():
            roi_frame, off_x, off_y = apply_roi(frame, data.get("roi"))
            live_kp, live_des = self._sift.detectAndCompute(roi_frame, None)
            if live_des is None or len(live_des) < 2:
                continue

            for variant in data["variants"]:
                if variant["des"] is None:
                    continue

                matches = self._flann.knnMatch(variant["des"], live_des, k=2)
                good = [m for m, n in matches if m.distance < 0.7 * n.distance]

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

                # Normalise: at min_matches → 0.5, at 2×min_matches → 1.0
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
