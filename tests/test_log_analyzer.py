"""
Unit tests for calibration/analysis/core.py — LogAnalyzer.

mss and pandas I/O are fully mocked — no real screen capture or CSV files.
"""

from unittest.mock import patch

import pandas as pd

from calibration.analysis.core import LogAnalyzer


def _make_analyzer(screen_w: int = 1920, screen_h: int = 1080, padding: int = 10) -> LogAnalyzer:
    """Return a LogAnalyzer with _get_screen_resolution mocked out."""
    with patch.object(LogAnalyzer, "_get_screen_resolution"):
        analyzer = LogAnalyzer(padding=padding)
    analyzer.screen_w = screen_w
    analyzer.screen_h = screen_h
    return analyzer


# ---------------------------------------------------------------------------
# calculate_roi
# ---------------------------------------------------------------------------

class TestCalculateRoi:
    def test_none_input_returns_none(self):
        analyzer = _make_analyzer()
        assert analyzer.calculate_roi(None) is None

    def test_empty_dataframe_returns_none(self):
        analyzer = _make_analyzer()
        assert analyzer.calculate_roi(pd.DataFrame()) is None

    def test_correct_roi_without_padding(self):
        # Single detection at (100,200,50,80) — right edge at 150, bottom at 280
        analyzer = _make_analyzer(padding=0)
        df = pd.DataFrame({"x": [100], "y": [200], "w": [50], "h": [80]})
        roi = analyzer.calculate_roi(df)
        assert roi["left"]   == 100
        assert roi["top"]    == 200
        assert roi["width"]  == 50    # 150 - 100
        assert roi["height"] == 80    # 280 - 200

    def test_padding_expands_roi(self):
        analyzer = _make_analyzer(padding=20)
        df = pd.DataFrame({"x": [100], "y": [100], "w": [50], "h": [50]})
        roi = analyzer.calculate_roi(df)
        assert roi["left"]  == 80    # 100 - 20
        assert roi["top"]   == 80    # 100 - 20
        assert roi["width"] == 90    # (150+20) - (100-20) = 170 - 80
        assert roi["height"] == 90

    def test_clamped_to_zero_on_left_and_top(self):
        # Detection near origin — padding would go negative
        analyzer = _make_analyzer(padding=50)
        df = pd.DataFrame({"x": [10], "y": [10], "w": [20], "h": [20]})
        roi = analyzer.calculate_roi(df)
        assert roi["left"] == 0   # clamped: max(0, 10-50)
        assert roi["top"]  == 0

    def test_clamped_to_screen_bounds_on_right_and_bottom(self):
        # Detection near bottom-right — padding would exceed screen
        analyzer = _make_analyzer(screen_w=1920, screen_h=1080, padding=50)
        df = pd.DataFrame({"x": [1900], "y": [1060], "w": [30], "h": [30]})
        roi = analyzer.calculate_roi(df)
        assert roi["left"] + roi["width"]  <= 1920
        assert roi["top"]  + roi["height"] <= 1080

    def test_multiple_detections_cover_union(self):
        analyzer = _make_analyzer(padding=0)
        df = pd.DataFrame({
            "x": [0,  200],
            "y": [0,  300],
            "w": [50,  50],
            "h": [50,  50],
        })
        roi = analyzer.calculate_roi(df)
        # min_x=0, max_x=250, min_y=0, max_y=350
        assert roi["left"]   == 0
        assert roi["top"]    == 0
        assert roi["width"]  == 250
        assert roi["height"] == 350

    def test_monitor_index_included_in_result(self):
        from calibration.config import MONITOR_INDEX
        analyzer = _make_analyzer()
        df = pd.DataFrame({"x": [0], "y": [0], "w": [10], "h": [10]})
        roi = analyzer.calculate_roi(df)
        assert roi["mon"] == MONITOR_INDEX


# ---------------------------------------------------------------------------
# load_and_merge_logs
# ---------------------------------------------------------------------------

class TestLoadAndMergeLogs:
    def test_returns_none_when_no_csv_files(self):
        analyzer = _make_analyzer()
        with patch("calibration.analysis.core.glob.glob", return_value=[]):
            result = analyzer.load_and_merge_logs()
        assert result is None

    def test_returns_none_when_all_files_are_empty(self):
        analyzer = _make_analyzer()
        with (
            patch("calibration.analysis.core.glob.glob", return_value=["a.csv"]),
            patch("calibration.analysis.core.pd.read_csv", return_value=pd.DataFrame()),
        ):
            result = analyzer.load_and_merge_logs()
        assert result is None

    def test_returns_dataframe_for_valid_csv(self):
        analyzer = _make_analyzer()
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "w": [5, 6], "h": [7, 8]})
        with (
            patch("calibration.analysis.core.glob.glob", return_value=["a.csv"]),
            patch("calibration.analysis.core.pd.read_csv", return_value=df),
        ):
            result = analyzer.load_and_merge_logs()
        assert result is not None
        assert len(result) == 2

    def test_merges_multiple_csv_files(self):
        analyzer = _make_analyzer()
        df1 = pd.DataFrame({"x": [1], "y": [1], "w": [1], "h": [1]})
        df2 = pd.DataFrame({"x": [2], "y": [2], "w": [2], "h": [2]})
        with (
            patch("calibration.analysis.core.glob.glob", return_value=["a.csv", "b.csv"]),
            patch("calibration.analysis.core.pd.read_csv", side_effect=[df1, df2]),
        ):
            result = analyzer.load_and_merge_logs()
        assert result is not None
        assert len(result) == 2

    def test_skips_bad_files_without_raising(self):
        analyzer = _make_analyzer()
        good_df = pd.DataFrame({"x": [1], "y": [1], "w": [1], "h": [1]})
        with (
            patch("calibration.analysis.core.glob.glob", return_value=["bad.csv", "good.csv"]),
            patch(
                "calibration.analysis.core.pd.read_csv",
                side_effect=[Exception("bad file"), good_df],
            ),
        ):
            result = analyzer.load_and_merge_logs()
        assert result is not None
        assert len(result) == 1
