"""
Unit tests for calibration/roi_overlay.py — coordinate conversion and
draw call dispatch logic.
"""

from unittest.mock import MagicMock

from calibration.roi_overlay import draw_roi_overlays, roi_to_pixels


class TestRoiToPixels:
    def test_converts_fractions_to_pixels(self):
        # Arrange & Act
        x, y, w, h = roi_to_pixels((0.25, 0.50, 0.50, 0.25), frame_w=400, frame_h=200)

        # Assert
        assert x == 100   # 400 * 0.25
        assert y == 100   # 200 * 0.50
        assert w == 200   # 400 * 0.50
        assert h == 50    # 200 * 0.25

    def test_adds_monitor_offsets(self):
        # Arrange & Act
        x, y, w, h = roi_to_pixels(
            (0.0, 0.0, 1.0, 1.0), frame_w=800, frame_h=600, off_x=30, off_y=15
        )

        # Assert
        assert x == 30    # 0 + off_x
        assert y == 15    # 0 + off_y
        assert w == 800
        assert h == 600

    def test_zero_fraction_clamps_width_and_height_to_one(self):
        # Arrange & Act — degenerate ROI with zero w/h
        x, y, w, h = roi_to_pixels((0.0, 0.0, 0.0, 0.0), frame_w=100, frame_h=100)

        # Assert — must never produce zero-size region
        assert w >= 1
        assert h >= 1

    def test_full_frame_roi_covers_entire_frame(self):
        # Arrange & Act
        x, y, w, h = roi_to_pixels((0.0, 0.0, 1.0, 1.0), frame_w=1920, frame_h=1080)

        # Assert
        assert x == 0
        assert y == 0
        assert w == 1920
        assert h == 1080


class TestDrawRoiOverlays:
    def _make_overlay(self) -> MagicMock:
        overlay = MagicMock()
        overlay.draw_roi_rect = MagicMock()
        return overlay

    def test_skips_targets_without_roi(self):
        # Arrange
        targets = {
            "DODGE": {"file": "t.png", "color": "yellow"},           # no roi
            "GRADIENT_INCOMING": {"file": None, "color": "purple"},   # no roi
        }
        overlay = self._make_overlay()

        # Act
        draw_roi_overlays(overlay, targets, frame_w=800, frame_h=600, off_x=0, off_y=0)

        # Assert — no draw calls made
        overlay.draw_roi_rect.assert_not_called()

    def test_draws_rect_for_target_with_roi(self):
        # Arrange
        targets = {
            "TURN_ALLY": {
                "color": "gold",
                "roi": (0.0, 0.0, 0.20, 0.35),
            }
        }
        overlay = self._make_overlay()

        # Act
        draw_roi_overlays(overlay, targets, frame_w=1000, frame_h=1000, off_x=0, off_y=0)

        # Assert
        overlay.draw_roi_rect.assert_called_once_with(0, 0, 200, 350, "gold", "TURN_ALLY")

    def test_draws_one_rect_per_target_with_roi(self):
        # Arrange
        targets = {
            "A": {"color": "red",  "roi": (0.0, 0.0, 0.5, 0.5)},
            "B": {"color": "blue", "roi": (0.5, 0.5, 0.5, 0.5)},
            "C": {"color": "lime"},  # no roi — skipped
        }
        overlay = self._make_overlay()

        # Act
        draw_roi_overlays(overlay, targets, frame_w=100, frame_h=100, off_x=0, off_y=0)

        # Assert — exactly 2 calls (C skipped)
        assert overlay.draw_roi_rect.call_count == 2

    def test_applies_monitor_offsets_to_coordinates(self):
        # Arrange
        targets = {"X": {"color": "white", "roi": (0.0, 0.0, 1.0, 1.0)}}
        overlay = self._make_overlay()

        # Act
        draw_roi_overlays(overlay, targets, frame_w=800, frame_h=600, off_x=50, off_y=20)

        # Assert — off_x/off_y added to x, y
        overlay.draw_roi_rect.assert_called_once_with(50, 20, 800, 600, "white", "X")

    def test_uses_white_as_default_color_when_missing(self):
        # Arrange
        targets = {"X": {"roi": (0.0, 0.0, 1.0, 1.0)}}   # no "color" key
        overlay = self._make_overlay()

        # Act
        draw_roi_overlays(overlay, targets, frame_w=100, frame_h=100, off_x=0, off_y=0)

        # Assert
        _, _, _, _, color, _ = overlay.draw_roi_rect.call_args[0]
        assert color == "white"
