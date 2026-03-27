"""
Unit tests for calibration/logger.py — CalibrationLogger.

File I/O is fully mocked — no CSV files are written to disk.
"""

import pytest
from unittest.mock import patch, mock_open, MagicMock, call

from calibration.logger import CalibrationLogger


class TestRecordingState:
    def test_initial_state_is_not_recording(self):
        # Arrange & Act
        logger = CalibrationLogger()

        # Assert
        assert logger.get_record_status() is False

    def test_start_recording_sets_flag_true(self):
        # Arrange
        logger = CalibrationLogger()

        # Act
        logger.start_recording()

        # Assert
        assert logger.get_record_status() is True

    def test_start_recording_clears_existing_points(self):
        # Arrange
        logger = CalibrationLogger()
        logger.points = [(1, 2, 3, 4, "X")]

        # Act
        logger.start_recording()

        # Assert
        assert logger.points == []

    def test_stop_recording_sets_flag_false(self):
        # Arrange
        logger = CalibrationLogger()
        logger.start_recording()

        # Act
        with patch.object(logger, "save_to_csv"):
            logger.stop_recording()

        # Assert
        assert logger.get_record_status() is False

    def test_stop_recording_calls_save_to_csv(self):
        # Arrange
        logger = CalibrationLogger()
        logger.start_recording()

        # Act
        with patch.object(logger, "save_to_csv") as mock_save:
            logger.stop_recording()

        # Assert
        mock_save.assert_called_once()


class TestAddPoint:
    def test_add_point_appends_when_recording(self):
        # Arrange
        logger = CalibrationLogger()
        logger.start_recording()

        # Act
        logger.add_point(10, 20, 30, 40, "DODGE")

        # Assert
        assert len(logger.points) == 1
        assert logger.points[0] == (10, 20, 30, 40, "DODGE")

    def test_add_point_ignores_when_not_recording(self):
        # Arrange
        logger = CalibrationLogger()

        # Act
        logger.add_point(10, 20, 30, 40, "DODGE")

        # Assert
        assert logger.points == []

    def test_add_point_accumulates_multiple_points(self):
        # Arrange
        logger = CalibrationLogger()
        logger.start_recording()

        # Act
        logger.add_point(1, 2, 3, 4, "A")
        logger.add_point(5, 6, 7, 8, "B")
        logger.add_point(9, 0, 1, 2, "C")

        # Assert
        assert len(logger.points) == 3


class TestSaveToCsv:
    def test_save_to_csv_writes_header_and_rows(self, tmp_path):
        # Arrange
        logger = CalibrationLogger()
        logger.points = [(10, 20, 30, 40, "DODGE"), (1, 2, 3, 4, "PERFECT")]

        with patch("calibration.logger.LOG_DIR", str(tmp_path)), \
             patch("calibration.logger.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "20260101_120000"

            # Act
            logger.save_to_csv()

        # Assert — file exists and has correct content
        import csv
        saved_file = tmp_path / "calibration_20260101_120000.csv"
        assert saved_file.exists()
        with open(saved_file, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["x", "y", "w", "h", "type"]
        assert rows[1] == ["10", "20", "30", "40", "DODGE"]
        assert rows[2] == ["1", "2", "3", "4", "PERFECT"]

    def test_save_to_csv_does_nothing_when_no_points(self, tmp_path):
        # Arrange
        logger = CalibrationLogger()
        logger.points = []

        with patch("calibration.logger.LOG_DIR", str(tmp_path)):
            # Act
            logger.save_to_csv()

        # Assert — no CSV file was created
        assert list(tmp_path.iterdir()) == []

    def test_save_to_csv_handles_write_error_gracefully(self, tmp_path):
        # Arrange
        logger = CalibrationLogger()
        logger.points = [(1, 2, 3, 4, "X")]

        with patch("calibration.logger.LOG_DIR", str(tmp_path)), \
             patch("calibration.logger.datetime") as mock_dt, \
             patch("builtins.open", side_effect=OSError("disk full")):
            mock_dt.now.return_value.strftime.return_value = "20260101_120000"

            # Act & Assert — must not raise
            logger.save_to_csv()
