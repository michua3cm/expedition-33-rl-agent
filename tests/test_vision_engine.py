"""
Unit tests for vision/engine.py — Detection, GameState, apply_roi, _iou, nms.
"""

import numpy as np

from vision.engine import Detection, GameState, _iou, apply_roi, nms


class TestApplyRoi:
    def test_none_roi_returns_original_frame_with_zero_offsets(self):
        # Arrange
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        # Act
        result, off_x, off_y = apply_roi(frame, None)

        # Assert
        assert result is frame
        assert off_x == 0
        assert off_y == 0

    def test_crops_width_and_height_correctly(self):
        # Arrange
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        # Act — take right half, full height
        cropped, off_x, off_y = apply_roi(frame, (0.5, 0.0, 0.5, 1.0))

        # Assert
        assert cropped.shape == (100, 100, 3)   # 200 * 0.5 = 100 wide
        assert off_x == 100                     # 200 * 0.5
        assert off_y == 0

    def test_offsets_match_top_left_of_roi(self):
        # Arrange
        frame = np.zeros((200, 400, 3), dtype=np.uint8)

        # Act
        _, off_x, off_y = apply_roi(frame, (0.25, 0.50, 0.50, 0.25))

        # Assert
        assert off_x == 100   # 400 * 0.25
        assert off_y == 100   # 200 * 0.50

    def test_zero_size_roi_clamps_to_minimum_1x1(self):
        # Arrange
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act — degenerate roi with zero w/h fractions
        cropped, _, _ = apply_roi(frame, (0.0, 0.0, 0.0, 0.0))

        # Assert — must not produce a 0-size array
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] >= 1

    def test_works_with_greyscale_frame(self):
        # Arrange
        frame = np.zeros((100, 200), dtype=np.uint8)

        # Act
        cropped, off_x, off_y = apply_roi(frame, (0.0, 0.0, 0.5, 0.5))

        # Assert
        assert cropped.shape == (50, 100)
        assert off_x == 0
        assert off_y == 0

    def test_cropped_region_contains_correct_pixels(self):
        # Arrange — paint a known pixel at (120, 60) in a 100×200 frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[60, 120] = (255, 0, 0)   # x=120, y=60 in full frame

        # Act — ROI covering x 100-200, y 0-100 (right half, full height)
        cropped, off_x, off_y = apply_roi(frame, (0.5, 0.0, 0.5, 1.0))

        # Assert — pixel should appear at (120-100, 60) = (20, 60) in crop
        assert off_x == 100
        assert np.array_equal(cropped[60, 20], [255, 0, 0])


class TestDetection:
    def test_detection_stores_all_fields(self):
        # Arrange & Act
        det = Detection(label="DODGE", x=10, y=20, w=30, h=40, confidence=0.85)

        # Assert
        assert det.label == "DODGE"
        assert det.x == 10
        assert det.y == 20
        assert det.w == 30
        assert det.h == 40
        assert det.confidence == 0.85

    def test_detection_equality(self):
        # Arrange
        d1 = Detection(label="PERFECT", x=0, y=0, w=10, h=10, confidence=0.9)
        d2 = Detection(label="PERFECT", x=0, y=0, w=10, h=10, confidence=0.9)

        # Assert — dataclasses compare by value
        assert d1 == d2

    def test_detection_inequality_on_label(self):
        # Arrange
        d1 = Detection(label="DODGE", x=0, y=0, w=10, h=10, confidence=0.8)
        d2 = Detection(label="PARRY", x=0, y=0, w=10, h=10, confidence=0.8)

        # Assert
        assert d1 != d2

    def test_detection_confidence_at_boundaries(self):
        # Arrange & Act — confidence = 0.0 and 1.0 are valid
        low  = Detection(label="X", x=0, y=0, w=1, h=1, confidence=0.0)
        high = Detection(label="X", x=0, y=0, w=1, h=1, confidence=1.0)

        # Assert
        assert low.confidence == 0.0
        assert high.confidence == 1.0


class TestGameState:
    def test_game_state_stores_all_fields(self):
        # Arrange
        dets = [Detection("DODGE", 0, 0, 10, 10, 0.8)]
        frame = np.zeros((64, 64), dtype=np.uint8)

        # Act
        state = GameState(
            detections=dets,
            timestamp=1234567890.0,
            engine_name="PIXEL",
            frame=frame,
        )

        # Assert
        assert state.detections == dets
        assert state.timestamp == 1234567890.0
        assert state.engine_name == "PIXEL"
        assert state.frame is frame

    def test_game_state_frame_defaults_to_none(self):
        # Arrange & Act
        state = GameState(detections=[], timestamp=0.0, engine_name="PIXEL")

        # Assert
        assert state.frame is None

    def test_game_state_empty_detections(self):
        # Arrange & Act
        state = GameState(detections=[], timestamp=0.0, engine_name="ORB")

        # Assert
        assert state.detections == []
        assert len(state.detections) == 0

    def test_game_state_multiple_detections(self):
        # Arrange
        dets = [
            Detection("PERFECT", 0, 0, 10, 10, 0.9),
            Detection("DODGE",   5, 5, 10, 10, 0.7),
        ]

        # Act
        state = GameState(detections=dets, timestamp=1.0, engine_name="SIFT")

        # Assert
        assert len(state.detections) == 2
        assert state.detections[0].label == "PERFECT"
        assert state.detections[1].label == "DODGE"


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------

class TestIou:
    def _det(self, x, y, w, h) -> Detection:
        return Detection(label="X", x=x, y=y, w=w, h=h, confidence=1.0)

    def test_no_overlap_returns_zero(self):
        a = self._det(0, 0, 10, 10)
        b = self._det(20, 0, 10, 10)
        assert _iou(a, b) == 0.0

    def test_identical_boxes_return_one(self):
        a = self._det(0, 0, 10, 10)
        assert _iou(a, a) == 1.0

    def test_half_horizontal_overlap(self):
        # Boxes overlap by 5×10 = 50; union = 100+100-50 = 150
        a = self._det(0, 0, 10, 10)
        b = self._det(5, 0, 10, 10)
        assert abs(_iou(a, b) - 50 / 150) < 1e-9

    def test_touching_edges_returns_zero(self):
        # Shares an edge but no interior area
        a = self._det(0, 0, 10, 10)
        b = self._det(10, 0, 10, 10)
        assert _iou(a, b) == 0.0

    def test_contained_box(self):
        # Inner 10×10 fully inside 20×20 — intersection=100, union=400+100-100=400
        outer = self._det(0, 0, 20, 20)
        inner = self._det(5, 5, 10, 10)
        assert abs(_iou(outer, inner) - 100 / 400) < 1e-9

    def test_symmetry(self):
        a = self._det(0, 0, 10, 10)
        b = self._det(3, 3, 10, 10)
        assert _iou(a, b) == _iou(b, a)


# ---------------------------------------------------------------------------
# nms
# ---------------------------------------------------------------------------

class TestNms:
    def _det(self, label, x, y, w, h, conf) -> Detection:
        return Detection(label=label, x=x, y=y, w=w, h=h, confidence=conf)

    def test_empty_list_returns_empty(self):
        assert nms([]) == []

    def test_single_detection_returned_unchanged(self):
        d = self._det("A", 0, 0, 10, 10, 0.9)
        assert nms([d]) == [d]

    def test_non_overlapping_same_label_both_kept(self):
        a = self._det("A", 0, 0, 10, 10, 0.9)
        b = self._det("A", 100, 0, 10, 10, 0.8)
        assert len(nms([a, b])) == 2

    def test_overlapping_same_label_lower_conf_suppressed(self):
        # b is almost identical to a → high IoU → lower-conf b suppressed
        high = self._det("A", 0, 0, 10, 10, 0.9)
        low  = self._det("A", 1, 1, 10, 10, 0.5)
        result = nms([high, low])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_labels_not_suppressed(self):
        # Same position, different labels — must both survive
        a = self._det("A", 0, 0, 10, 10, 0.9)
        b = self._det("B", 0, 0, 10, 10, 0.8)
        assert len(nms([a, b])) == 2

    def test_keeps_highest_conf_when_inputs_reversed(self):
        # Input order should not matter — highest conf always wins
        low  = self._det("A", 0, 0, 10, 10, 0.5)
        high = self._det("A", 0, 0, 10, 10, 0.9)
        result = nms([low, high])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_custom_iou_threshold_controls_suppression(self):
        # IoU between a and b ≈ 0.33 — kept at threshold 0.5, suppressed at 0.2
        a = self._det("A", 0, 0, 10, 10, 0.9)
        b = self._det("A", 5, 0, 10, 10, 0.7)
        assert len(nms([a, b], iou_threshold=0.5)) == 2   # IoU < 0.5 → both kept
        assert len(nms([a, b], iou_threshold=0.2)) == 1   # IoU > 0.2 → b suppressed

    def test_multiple_labels_handled_independently(self):
        # Two DODGE boxes (overlapping) + one PERFECT box
        d1 = self._det("DODGE",   0, 0, 10, 10, 0.9)
        d2 = self._det("DODGE",   1, 1, 10, 10, 0.6)
        p  = self._det("PERFECT", 0, 0, 10, 10, 0.8)
        result = nms([d1, d2, p])
        labels = [r.label for r in result]
        assert labels.count("DODGE") == 1    # one suppressed
        assert labels.count("PERFECT") == 1  # untouched
