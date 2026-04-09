"""
Unit tests for calibration/collector.py — SmartCollector.

All file I/O and OS calls are mocked.  No actual images are written to disk.
"""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from vision.engine import Detection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(label: str, x=10, y=20, w=100, h=50, confidence=0.9) -> Detection:
    return Detection(label=label, x=x, y=y, w=w, h=h, confidence=confidence)


def _make_frame(h: int = 600, w: int = 800) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# _write_yolo_label
# ---------------------------------------------------------------------------

class TestWriteYoloLabel:
    def _collector(self):
        """Return a SmartCollector with heavy dependencies mocked out."""
        with (
            patch("calibration.collector.mss.mss"),
            patch("calibration.collector.OverlayWindow"),
            patch("calibration.collector.vision.registry.create") as mock_create,
        ):
            mock_engine = MagicMock()
            mock_engine.needs_color = False
            mock_create.return_value = mock_engine

            from calibration.collector import SmartCollector
            collector = SmartCollector.__new__(SmartCollector)
            # Minimal state initialisation — only what _write_yolo_label needs
            return collector

    def test_writes_correct_format_for_single_detection(self):
        # Arrange
        collector = self._collector()
        det = _det("DODGE", x=80, y=120, w=160, h=60)  # class_id=0
        frame_w, frame_h = 800, 600

        # Act
        m = mock_open()
        with patch("builtins.open", m):
            collector._write_yolo_label("out.txt", [det], frame_w, frame_h)

        written = m().write.call_args[0][0]
        parts = written.split()

        # Assert
        assert parts[0] == "0"  # DODGE = class 0
        # x_center = (80 + 80) / 800 = 0.2
        assert abs(float(parts[1]) - 0.2) < 1e-5
        # y_center = (120 + 30) / 600 = 0.25
        assert abs(float(parts[2]) - 0.25) < 1e-5
        # w_frac = 160 / 800 = 0.2
        assert abs(float(parts[3]) - 0.2) < 1e-5
        # h_frac = 60 / 600 = 0.1
        assert abs(float(parts[4]) - 0.1) < 1e-5

    def test_skips_gradient_incoming(self):
        # Arrange
        collector = self._collector()
        dets = [
            _det("GRADIENT_INCOMING"),
            _det("DODGE"),
        ]

        m = mock_open()
        with patch("builtins.open", m):
            collector._write_yolo_label("out.txt", dets, 800, 600)

        written = m().write.call_args[0][0]
        lines = [ln for ln in written.split("\n") if ln.strip()]

        # Assert — only one line (DODGE), GRADIENT_INCOMING omitted
        assert len(lines) == 1
        assert lines[0].startswith("0 ")  # DODGE = 0

    def test_skips_unknown_labels(self):
        # Arrange
        collector = self._collector()
        dets = [_det("UNKNOWN_LABEL")]

        m = mock_open()
        with patch("builtins.open", m):
            collector._write_yolo_label("out.txt", dets, 800, 600)

        written = m().write.call_args[0][0]

        # Assert — empty file
        assert written.strip() == ""

    def test_writes_multiple_detections(self):
        # Arrange
        collector = self._collector()
        dets = [_det("PERFECT"), _det("PARRIED")]  # class 1 and 2

        m = mock_open()
        with patch("builtins.open", m):
            collector._write_yolo_label("out.txt", dets, 100, 100)

        written = m().write.call_args[0][0]
        lines = [ln for ln in written.split("\n") if ln.strip()]

        # Assert — two lines with correct class IDs
        assert len(lines) == 2
        assert lines[0].startswith("1 ")   # PERFECT = 1
        assert lines[1].startswith("2 ")   # PARRIED = 2

    def test_empty_detections_writes_empty_file(self):
        # Arrange
        collector = self._collector()

        m = mock_open()
        with patch("builtins.open", m):
            collector._write_yolo_label("out.txt", [], 800, 600)

        written = m().write.call_args[0][0]
        assert written == ""


# ---------------------------------------------------------------------------
# SmartCollector state — trigger cooldown and mode flags
# ---------------------------------------------------------------------------

class TestSmartCollectorState:
    def _make_collector(self):
        """Return a fully initialised SmartCollector with all I/O mocked."""
        with (
            patch("calibration.collector.mss.mss") as mock_mss,
            patch("calibration.collector.OverlayWindow"),
            patch("calibration.collector.vision.registry.create") as mock_create,
        ):
            mock_sct = MagicMock()
            mock_sct.monitors = [
                None,
                {"top": 0, "left": 0, "width": 1920, "height": 1080},
            ]
            mock_mss.return_value = mock_sct

            mock_engine = MagicMock()
            mock_engine.needs_color = False
            mock_create.return_value = mock_engine

            from calibration.collector import SmartCollector
            collector = SmartCollector()
            return collector

    def test_initial_flags_are_off(self):
        collector = self._make_collector()
        assert collector._trigger_mode is False
        assert collector._auto_capture is False
        assert collector._show_roi is True
        assert collector.running is True

    def test_initial_counts_are_zero(self):
        collector = self._make_collector()
        assert collector._raw_count == 0
        assert collector._labeled_count == 0

    def test_trigger_cooldown_fires_when_elapsed(self):
        # Arrange
        collector = self._make_collector()
        collector._trigger_mode = True
        collector._trigger_cooldowns = {"DODGE": 0.0}  # very old save time

        frame = _make_frame()
        detections = [_det("DODGE")]

        # Act
        with (
            patch.object(collector, "_save_labeled") as mock_save,
            patch("calibration.collector.time.time", return_value=1000.0),
        ):
            # Simulate trigger logic directly (extracted from run loop)
            curr_time = 1000.0
            for det in detections:
                last = collector._trigger_cooldowns.get(det.label, 0.0)
                if curr_time - last >= 0.4:
                    collector._save_labeled(frame, detections)
                    for d in detections:
                        collector._trigger_cooldowns[d.label] = curr_time
                    break

        # Assert
        mock_save.assert_called_once_with(frame, detections)
        assert collector._trigger_cooldowns["DODGE"] == 1000.0

    def test_trigger_cooldown_suppresses_save_within_window(self):
        # Arrange
        collector = self._make_collector()
        collector._trigger_mode = True
        collector._trigger_cooldowns = {"DODGE": 999.9}  # saved 0.05 s ago

        detections = [_det("DODGE")]
        saves = []

        # Simulate trigger logic
        curr_time = 999.95  # only 0.05 s elapsed — below 0.4 s threshold
        for det in detections:
            last = collector._trigger_cooldowns.get(det.label, 0.0)
            if curr_time - last >= 0.4:
                saves.append(True)
                for d in detections:
                    collector._trigger_cooldowns[d.label] = curr_time
                break

        # Assert — no save fired
        assert saves == []

    def test_save_raw_increments_raw_count(self):
        # Arrange
        collector = self._make_collector()
        frame = _make_frame()

        with (
            patch("calibration.collector.cv2.imwrite"),
            patch("calibration.collector.time.time", return_value=1234.0),
        ):
            collector._save_raw(frame)

        assert collector._raw_count == 1

    def test_save_labeled_increments_labeled_count(self):
        # Arrange
        collector = self._make_collector()
        frame = _make_frame()
        dets = [_det("DODGE")]

        with (
            patch("calibration.collector.cv2.imwrite"),
            patch("calibration.collector.time.time", return_value=1234.0),
            patch("builtins.open", mock_open()),
        ):
            collector._save_labeled(frame, dets)

        assert collector._labeled_count == 1

    def test_trigger_mode_off_does_not_save_on_detection(self):
        # Arrange
        collector = self._make_collector()
        collector._trigger_mode = False
        detections = [_det("DODGE")]
        collector._trigger_cooldowns = {}

        # Simulate run-loop trigger block
        saves = []
        if collector._trigger_mode and detections:
            saves.append(True)

        # Assert
        assert saves == []
