"""
Unit tests for vision/engines/composite.py — CompositeEngine.

Sub-engines are replaced with MagicMock objects so no template files,
cv2 inference, or model weights are needed.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from vision.engine import Detection
from vision.engines.composite import CompositeEngine


def _mock_engine(needs_color: bool = False, detections: list | None = None) -> MagicMock:
    eng = MagicMock()
    eng.needs_color = needs_color
    eng.detect.return_value = detections or []
    return eng


def _det(label: str) -> Detection:
    return Detection(label=label, x=0, y=0, w=10, h=10, confidence=0.8)


# ---------------------------------------------------------------------------
# load() — routing by engine key
# ---------------------------------------------------------------------------

class TestCompositeEngineLoad:
    def test_defaults_to_pixel_when_no_engine_key(self):
        engine = CompositeEngine()
        mock_pixel = _mock_engine()
        with patch("vision.engines.composite.create", return_value=mock_pixel):
            engine.load({"DODGE": {"file": "t.png"}}, "/fake")
        assert "PIXEL" in engine._sub_engines

    def test_routes_target_to_specified_engine(self):
        engine = CompositeEngine()
        mock_sift = _mock_engine()
        with patch("vision.engines.composite.create", return_value=mock_sift):
            engine.load({"JUMP_CUE": {"file": "t.png", "engine": "SIFT"}}, "/fake")
        assert "SIFT" in engine._sub_engines

    def test_groups_multiple_targets_by_engine(self):
        engine = CompositeEngine()
        created_engines: dict[str, MagicMock] = {}

        def _create(name: str) -> MagicMock:
            m = _mock_engine()
            created_engines[name] = m
            return m

        targets = {
            "DODGE":    {"file": "a.png"},                  # → PIXEL (default)
            "JUMP_CUE": {"file": "b.png", "engine": "SIFT"},  # → SIFT
        }
        with patch("vision.engines.composite.create", side_effect=_create):
            engine.load(targets, "/fake")

        # Each engine must have been loaded with only its own targets
        assert created_engines["PIXEL"].load.call_args[0][0] == {"DODGE": targets["DODGE"]}
        assert created_engines["SIFT"].load.call_args[0][0] == {"JUMP_CUE": targets["JUMP_CUE"]}

    def test_clears_previous_sub_engines_on_reload(self):
        engine = CompositeEngine()
        mock_eng = _mock_engine()
        with patch("vision.engines.composite.create", return_value=mock_eng):
            engine.load({"A": {"file": "a.png"}}, "/fake")
            engine.load({"B": {"file": "b.png"}}, "/fake")
        # Only one PIXEL sub-engine from the latest load
        assert len(engine._sub_engines) == 1


# ---------------------------------------------------------------------------
# needs_color
# ---------------------------------------------------------------------------

class TestCompositeEngineNeedsColor:
    def test_false_when_no_sub_engines(self):
        engine = CompositeEngine()
        assert engine.needs_color is False

    def test_false_when_all_sub_engines_greyscale(self):
        engine = CompositeEngine()
        with patch("vision.engines.composite.create", return_value=_mock_engine(needs_color=False)):
            engine.load({"A": {"file": "a.png"}}, "/fake")
        assert engine.needs_color is False

    def test_true_when_any_sub_engine_needs_color(self):
        engine = CompositeEngine()
        engines_iter = iter([_mock_engine(needs_color=True), _mock_engine(needs_color=False)])
        with patch("vision.engines.composite.create", side_effect=engines_iter):
            engine.load(
                {
                    "A": {"file": "a.png", "engine": "PIXEL"},
                    "B": {"file": "b.png", "engine": "SIFT"},
                },
                "/fake",
            )
        assert engine.needs_color is True


# ---------------------------------------------------------------------------
# detect() — result merging and frame routing
# ---------------------------------------------------------------------------

class TestCompositeEngineDetect:
    def test_returns_empty_when_no_sub_engines(self):
        engine = CompositeEngine()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert engine.detect(frame) == []

    def test_merges_detections_from_all_sub_engines(self):
        engine = CompositeEngine()
        mock_p = _mock_engine(detections=[_det("DODGE")])
        mock_s = _mock_engine(detections=[_det("JUMP_CUE")])
        engines_iter = iter([mock_p, mock_s])
        with patch("vision.engines.composite.create", side_effect=engines_iter):
            engine.load(
                {
                    "DODGE":    {"file": "a.png", "engine": "PIXEL"},
                    "JUMP_CUE": {"file": "b.png", "engine": "SIFT"},
                },
                "/fake",
            )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = engine.detect(frame)
        labels = {d.label for d in result}
        assert labels == {"DODGE", "JUMP_CUE"}

    def test_passes_bgr_frame_to_color_engine(self):
        """When composite needs_color=True, sub-engines that need color get BGR."""
        engine = CompositeEngine()
        mock_color = _mock_engine(needs_color=True, detections=[])
        with patch("vision.engines.composite.create", return_value=mock_color):
            engine.load({"A": {"file": "a.png"}}, "/fake")

        bgr_frame = np.zeros((50, 50, 3), dtype=np.uint8)
        engine.detect(bgr_frame)
        call_frame = mock_color.detect.call_args[0][0]
        assert call_frame is bgr_frame

    def test_passes_grey_frame_to_grey_engine_when_composite_is_grey(self):
        """When composite needs_color=False, the caller already passed grey."""
        engine = CompositeEngine()
        mock_grey = _mock_engine(needs_color=False, detections=[])
        with patch("vision.engines.composite.create", return_value=mock_grey):
            engine.load({"A": {"file": "a.png"}}, "/fake")

        grey_frame = np.zeros((50, 50), dtype=np.uint8)
        engine.detect(grey_frame)
        call_frame = mock_grey.detect.call_args[0][0]
        assert call_frame is grey_frame
