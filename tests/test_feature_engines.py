"""
Unit tests for vision/engines/sift.py and vision/engines/orb.py.

Both engines share the same load() guard logic (skip file=None, skip missing
files) and the same detect() early-exit when no templates are loaded.
Actual cv2 keypoint extraction from real images is not tested here — that
requires live template files and is covered by integration/manual testing.
"""

from unittest.mock import patch

import numpy as np

from vision.engines.orb import ORBEngine
from vision.engines.sift import SIFTEngine


def _blank_frame(h: int = 50, w: int = 50) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# SIFTEngine
# ---------------------------------------------------------------------------

class TestSIFTEngineLoad:
    def test_skips_target_with_no_file(self):
        engine = SIFTEngine()
        engine.load({"GRADIENT_INCOMING": {"file": None}}, "/fake")
        assert "GRADIENT_INCOMING" not in engine._templates

    def test_skips_missing_template_file(self):
        engine = SIFTEngine()
        with patch("vision.engines.sift.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": "missing.png"}}, "/fake")
        assert "DODGE" not in engine._templates

    def test_clears_templates_on_reload(self):
        engine = SIFTEngine()
        # Inject a sentinel to verify _templates.clear() is called on next load
        engine._templates["SENTINEL"] = {}
        with patch("vision.engines.sift.os.path.exists", return_value=False):
            engine.load({"B": {"file": "b.png"}}, "/fake")
        assert "SENTINEL" not in engine._templates

    def test_skips_multiple_none_file_targets(self):
        engine = SIFTEngine()
        engine.load(
            {
                "GRADIENT_INCOMING": {"file": None},
                "ANOTHER": {"file": None},
            },
            "/fake",
        )
        assert len(engine._templates) == 0


class TestSIFTEngineLoad:
    def test_loads_multiple_files_for_single_label(self):
        # SIFT load does not gate on des being None, so a blank image is enough
        # to trigger variant creation without needing to mock detectAndCompute.
        engine = SIFTEngine()
        fake_img = np.zeros((20, 20), dtype=np.uint8)
        with (
            patch("vision.engines.sift.os.path.exists", return_value=True),
            patch("vision.engines.sift.cv2.imread", return_value=fake_img),
        ):
            engine.load(
                {"DODGE": {"file": ["t1.png", "t2.png"]}},
                "/fake",
            )
        assert "DODGE" in engine._templates
        assert len(engine._templates["DODGE"]["variants"]) == 2

    def test_skips_label_when_all_files_missing(self):
        engine = SIFTEngine()
        with patch("vision.engines.sift.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": ["a.png", "b.png"]}}, "/fake")
        assert "DODGE" not in engine._templates


class TestSIFTEngineDetect:
    def test_returns_empty_list_when_no_templates_loaded(self):
        engine = SIFTEngine()
        assert engine.detect(_blank_frame()) == []

    def test_returns_empty_list_on_blank_frame_with_no_keypoints(self):
        # Blank frame has no texture → detectAndCompute returns no descriptors
        # → live_des is None → skipped → empty result
        engine = SIFTEngine()
        # Load a dummy template descriptor to enter the matching loop
        engine._templates["DODGE"] = {
            "variants": [
                {
                    "des": np.zeros((10, 128), dtype=np.float32),
                    "kp": [],
                    "w": 10,
                    "h": 10,
                }
            ],
            "min_matches": 5,
            "roi": None,
        }
        assert engine.detect(_blank_frame()) == []


# ---------------------------------------------------------------------------
# ORBEngine
# ---------------------------------------------------------------------------

class TestORBEngineLoad:
    def test_skips_target_with_no_file(self):
        engine = ORBEngine()
        engine.load({"GRADIENT_INCOMING": {"file": None}}, "/fake")
        assert "GRADIENT_INCOMING" not in engine._templates

    def test_skips_missing_template_file(self):
        engine = ORBEngine()
        with patch("vision.engines.orb.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": "missing.png"}}, "/fake")
        assert "DODGE" not in engine._templates

    def test_clears_templates_on_reload(self):
        engine = ORBEngine()
        engine._templates["SENTINEL"] = {}
        with patch("vision.engines.orb.os.path.exists", return_value=False):
            engine.load({"B": {"file": "b.png"}}, "/fake")
        assert "SENTINEL" not in engine._templates

    def test_skips_multiple_none_file_targets(self):
        engine = ORBEngine()
        engine.load(
            {
                "GRADIENT_INCOMING": {"file": None},
                "ANOTHER": {"file": None},
            },
            "/fake",
        )
        assert len(engine._templates) == 0


class TestORBEngineLoad:
    def test_loads_multiple_files_for_single_label(self):
        # ORB skips variants with des=None, so we must mock ORB_create before
        # instantiation to inject a detectAndCompute that returns real des.
        fake_img = np.zeros((20, 20), dtype=np.uint8)
        fake_des = np.zeros((4, 32), dtype=np.uint8)
        with (
            patch("vision.engines.orb.os.path.exists", return_value=True),
            patch("vision.engines.orb.cv2.imread", return_value=fake_img),
            patch("vision.engines.orb.cv2.ORB_create") as mock_orb_create,
            patch("vision.engines.orb.cv2.BFMatcher"),
        ):
            mock_orb_create.return_value.detectAndCompute.return_value = ([], fake_des)
            engine = ORBEngine()
            engine.load(
                {"DODGE": {"file": ["t1.png", "t2.png"]}},
                "/fake",
            )
        assert "DODGE" in engine._templates
        assert len(engine._templates["DODGE"]["variants"]) == 2

    def test_skips_label_when_all_files_missing(self):
        engine = ORBEngine()
        with patch("vision.engines.orb.os.path.exists", return_value=False):
            engine.load({"DODGE": {"file": ["a.png", "b.png"]}}, "/fake")
        assert "DODGE" not in engine._templates


class TestORBEngineDetect:
    def test_returns_empty_list_when_no_templates_loaded(self):
        engine = ORBEngine()
        assert engine.detect(_blank_frame()) == []

    def test_returns_empty_list_on_blank_frame_with_no_keypoints(self):
        # Blank frame has no texture → detectAndCompute returns no descriptors
        # → live_des is None → skipped → empty result
        engine = ORBEngine()
        engine._templates["DODGE"] = {
            "variants": [
                {
                    "des": np.zeros((10, 32), dtype=np.uint8),
                    "kp": [],
                    "w": 10,
                    "h": 10,
                }
            ],
            "min_matches": 5,
            "roi": None,
        }
        assert engine.detect(_blank_frame()) == []
