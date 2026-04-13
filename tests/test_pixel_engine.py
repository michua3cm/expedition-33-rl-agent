"""
Unit tests for vision/engines/pixel.py — PixelEngine.

Template file I/O is mocked.  HSV saturation detection uses real numpy
arrays so cv2 colour-space math is exercised without hitting the filesystem.
"""

from unittest.mock import patch

import numpy as np

from vision.engines.pixel import PixelEngine


def _grey_bgr(h: int = 50, w: int = 50) -> np.ndarray:
    """Return a flat-grey BGR frame (saturation ≈ 0 in HSV)."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _colorful_bgr(h: int = 50, w: int = 50) -> np.ndarray:
    """Return a pure-green BGR frame (saturation = 255 in HSV)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (0, 255, 0)  # BGR green
    return frame


# ---------------------------------------------------------------------------
# load() — skip / reject behaviours
# ---------------------------------------------------------------------------

class TestPixelEngineLoad:
    def test_skips_target_with_no_file_and_no_hsv_sat_max(self):
        engine = PixelEngine()
        engine.load({"LABEL": {"file": None}}, "/fake")
        assert "LABEL" not in engine._targets

    def test_loads_hsv_sat_target_with_no_file(self):
        engine = PixelEngine()
        engine.load({"GRADIENT_INCOMING": {"file": None, "hsv_sat_max": 40}}, "/fake")
        assert "GRADIENT_INCOMING" in engine._targets
        assert engine._targets["GRADIENT_INCOMING"]["kind"] == "hsv_sat"
        assert engine._targets["GRADIENT_INCOMING"]["hsv_sat_max"] == 40.0

    def test_skips_missing_template_file(self):
        engine = PixelEngine()
        with patch("vision.engines.pixel.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": "missing.png"}}, "/fake")
        assert "DODGE" not in engine._targets

    def test_loads_grey_template_when_file_exists(self):
        engine = PixelEngine()
        fake_img = np.zeros((20, 30), dtype=np.uint8)
        with (
            patch("vision.engines.pixel.os.path.exists", return_value=True),
            patch("vision.engines.pixel.cv2.imread", return_value=fake_img),
        ):
            engine.load({"DODGE": {"file": "t.png", "threshold": 0.7}}, "/fake")
        assert "DODGE" in engine._targets
        assert engine._targets["DODGE"]["kind"] == "grey"
        assert engine._targets["DODGE"]["threshold"] == 0.7
        assert len(engine._targets["DODGE"]["templates"]) == 1

    def test_loads_multiple_files_for_single_label(self):
        engine = PixelEngine()
        fake_img = np.zeros((20, 30), dtype=np.uint8)
        with (
            patch("vision.engines.pixel.os.path.exists", return_value=True),
            patch("vision.engines.pixel.cv2.imread", return_value=fake_img),
        ):
            engine.load(
                {"DODGE": {"file": ["t1.png", "t2.png", "t3.png"], "threshold": 0.7}},
                "/fake",
            )
        assert "DODGE" in engine._targets
        assert len(engine._targets["DODGE"]["templates"]) == 3

    def test_skips_missing_files_within_list(self):
        engine = PixelEngine()
        fake_img = np.zeros((10, 10), dtype=np.uint8)

        def _exists(path: str) -> bool:
            return "present.png" in path

        with (
            patch("vision.engines.pixel.os.path.exists", side_effect=_exists),
            patch("vision.engines.pixel.cv2.imread", return_value=fake_img),
        ):
            engine.load(
                {"DODGE": {"file": ["present.png", "missing.png"]}},
                "/fake",
            )
        assert "DODGE" in engine._targets
        assert len(engine._targets["DODGE"]["templates"]) == 1

    def test_skips_label_when_all_files_missing(self):
        engine = PixelEngine()
        with patch("vision.engines.pixel.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": ["a.png", "b.png"]}}, "/fake")
        assert "DODGE" not in engine._targets

    def test_loads_multiple_targets(self):
        engine = PixelEngine()
        fake_img = np.zeros((10, 10), dtype=np.uint8)
        with (
            patch("vision.engines.pixel.os.path.exists", return_value=True),
            patch("vision.engines.pixel.cv2.imread", return_value=fake_img),
        ):
            engine.load(
                {
                    "A": {"file": "a.png"},
                    "B": {"file": "b.png"},
                },
                "/fake",
            )
        assert "A" in engine._targets
        assert "B" in engine._targets


# ---------------------------------------------------------------------------
# needs_color
# ---------------------------------------------------------------------------

class TestPixelEngineNeedsColor:
    def test_false_when_no_targets_loaded(self):
        engine = PixelEngine()
        assert engine.needs_color is False

    def test_false_for_grey_template_only(self):
        engine = PixelEngine()
        fake_img = np.zeros((10, 10), dtype=np.uint8)
        with (
            patch("vision.engines.pixel.os.path.exists", return_value=True),
            patch("vision.engines.pixel.cv2.imread", return_value=fake_img),
        ):
            engine.load({"DODGE": {"file": "t.png"}}, "/fake")
        assert engine.needs_color is False

    def test_true_for_hsv_sat_target(self):
        engine = PixelEngine()
        engine.load({"GRADIENT_INCOMING": {"file": None, "hsv_sat_max": 40}}, "/fake")
        assert engine.needs_color is True


# ---------------------------------------------------------------------------
# detect() — HSV saturation path (no filesystem, real cv2)
# ---------------------------------------------------------------------------

class TestPixelEngineDetectHsvSat:
    def _engine_with_hsv_sat(self, sat_max: float = 40.0) -> PixelEngine:
        engine = PixelEngine()
        engine.load(
            {"GRADIENT_INCOMING": {"file": None, "hsv_sat_max": sat_max}},
            "/fake",
        )
        return engine

    def test_fires_on_grey_frame(self):
        # Grey frame → saturation ≈ 0 → well below threshold → detection fires
        engine = self._engine_with_hsv_sat(40.0)
        detections = engine.detect(_grey_bgr())
        assert len(detections) == 1
        assert detections[0].label == "GRADIENT_INCOMING"

    def test_does_not_fire_on_colorful_frame(self):
        # Colourful frame → saturation = 255 → well above threshold → no detection
        engine = self._engine_with_hsv_sat(40.0)
        detections = engine.detect(_colorful_bgr())
        assert detections == []

    def test_confidence_is_one_for_zero_saturation(self):
        # mean_sat = 0 → conf = 1.0 - (0 / sat_max) = 1.0
        engine = self._engine_with_hsv_sat(40.0)
        detections = engine.detect(_grey_bgr())
        assert detections[0].confidence == 1.0

    def test_detection_covers_full_frame(self):
        engine = self._engine_with_hsv_sat(40.0)
        frame = _grey_bgr(h=60, w=80)
        det = engine.detect(frame)[0]
        assert det.x == 0
        assert det.y == 0
        assert det.w == 80
        assert det.h == 60

    def test_empty_templates_returns_empty_list(self):
        engine = PixelEngine()  # no load() called
        assert engine.detect(_grey_bgr()) == []
