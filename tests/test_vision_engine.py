"""
Unit tests for vision/engine.py — Detection and GameState dataclasses.
"""

import numpy as np

from vision.engine import Detection, GameState


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
