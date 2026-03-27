"""
Unit tests for tools/vision_benchmark.py.

All I/O, screen capture, and vision engine calls are fully mocked.
No live monitor access, no real image files, and no real engine
inference is performed in any test.
"""

from __future__ import annotations

import csv
import io
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from tools.vision_benchmark import (
    EngineResult,
    LiveStressResult,
    _load_images,
    _print_live_report,
    _print_report,
    _save_csv,
    run_engine_benchmark,
    run_live_stress_test,
)
from vision.engine import Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(label: str, confidence: float) -> Detection:
    return Detection(label=label, x=0, y=0, w=10, h=10, confidence=confidence)


def _make_grey_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# EngineResult — record() and finalise()
# ---------------------------------------------------------------------------

class TestEngineResultRecord:
    def test_record_accumulates_latency(self):
        # Arrange
        result = EngineResult(engine_name="PIXEL", frame_count=2)

        # Act
        result.record(0.01, [])
        result.record(0.02, [])

        # Assert
        assert result.latencies == [0.01, 0.02]

    def test_record_counts_detections(self):
        # Arrange
        result = EngineResult(engine_name="PIXEL", frame_count=3)
        detections = [_make_detection("DODGE", 0.8), _make_detection("PERFECT", 0.9)]

        # Act
        result.record(0.01, detections)
        result.record(0.01, [_make_detection("DODGE", 0.7)])

        # Assert
        assert result.detection_counts["DODGE"] == 2
        assert result.detection_counts["PERFECT"] == 1

    def test_record_sums_and_tracks_max_confidence(self):
        # Arrange
        result = EngineResult(engine_name="PIXEL", frame_count=2)

        # Act
        result.record(0.01, [_make_detection("DODGE", 0.6)])
        result.record(0.01, [_make_detection("DODGE", 0.9)])

        # Assert
        assert pytest.approx(result.confidence_sums["DODGE"], abs=1e-6) == 1.5
        assert pytest.approx(result.confidence_maxes["DODGE"], abs=1e-6) == 0.9


class TestEngineResultFinalise:
    def test_finalise_computes_fps_and_latency_stats(self):
        # Arrange — 4 frames each taking 10 ms (0.010 s)
        result = EngineResult(engine_name="PIXEL", frame_count=4)
        for _ in range(4):
            result.record(0.010, [])

        # Act
        result.finalise()

        # Assert
        assert pytest.approx(result.fps, abs=0.1) == 100.0
        assert pytest.approx(result.mean_ms, abs=0.01) == 10.0
        assert pytest.approx(result.median_ms, abs=0.01) == 10.0
        assert pytest.approx(result.p95_ms, abs=0.5) == 10.0
        assert pytest.approx(result.max_ms, abs=0.01) == 10.0

    def test_finalise_is_noop_when_no_latencies(self):
        # Arrange
        result = EngineResult(engine_name="PIXEL", frame_count=0)

        # Act
        result.finalise()

        # Assert — all metrics remain at their zero defaults
        assert result.fps == 0.0
        assert result.mean_ms == 0.0

    def test_finalise_uses_varied_latencies_for_p95(self):
        # Arrange — 100 frames: 99 at 10 ms, 1 at 100 ms
        result = EngineResult(engine_name="PIXEL", frame_count=100)
        for _ in range(99):
            result.record(0.010, [])
        result.record(0.100, [])

        # Act
        result.finalise()

        # Assert — p95 should be below the spike, max should capture it
        assert result.p95_ms < 50.0
        assert pytest.approx(result.max_ms, abs=0.1) == 100.0


# ---------------------------------------------------------------------------
# LiveStressResult — finalise() and recommended_hz
# ---------------------------------------------------------------------------

class TestLiveStressResultFinalise:
    def _make_result_with_fps(self, target_fps: float) -> LiveStressResult:
        """Build a LiveStressResult whose sustained FPS equals target_fps."""
        latency = 1.0 / target_fps
        result = LiveStressResult(engine_name="PIXEL", duration_s=1.0, frame_count=0)
        for _ in range(100):
            result.latencies.append(latency)
        result.finalise()
        return result

    def test_finalise_recommends_60hz_when_fps_above_72(self):
        # Arrange & Act
        result = self._make_result_with_fps(80.0)  # 80 >= 60 * 1.2 = 72

        # Assert
        assert result.recommended_hz == 60

    def test_finalise_recommends_30hz_when_fps_between_36_and_72(self):
        # Arrange & Act
        result = self._make_result_with_fps(50.0)  # 50 >= 36, < 72

        # Assert
        assert result.recommended_hz == 30

    def test_finalise_recommends_20hz_when_fps_between_24_and_36(self):
        # Arrange & Act
        result = self._make_result_with_fps(30.0)  # 30 >= 24, < 36

        # Assert
        assert result.recommended_hz == 20

    def test_finalise_recommends_0hz_when_fps_below_24(self):
        # Arrange & Act
        result = self._make_result_with_fps(15.0)  # 15 < 24

        # Assert
        assert result.recommended_hz == 0

    def test_finalise_is_noop_when_no_latencies(self):
        # Arrange
        result = LiveStressResult(engine_name="PIXEL", duration_s=5.0, frame_count=0)

        # Act
        result.finalise()

        # Assert
        assert result.fps == 0.0
        assert result.recommended_hz == 0


# ---------------------------------------------------------------------------
# _load_images()
# ---------------------------------------------------------------------------

class TestLoadImages:
    def test_loads_png_and_jpg_files(self, tmp_path):
        # Arrange — create dummy image files
        (tmp_path / "frame_01.png").write_bytes(b"")
        (tmp_path / "frame_02.jpg").write_bytes(b"")
        fake_frame = _make_grey_frame()

        with patch("tools.vision_benchmark.cv2.imread", return_value=fake_frame):
            # Act
            frames = _load_images(str(tmp_path), limit=None)

        # Assert
        assert len(frames) == 2
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_respects_limit_parameter(self, tmp_path):
        # Arrange — 5 image files, limit=3
        for i in range(5):
            (tmp_path / f"frame_{i:02d}.png").write_bytes(b"")
        fake_frame = _make_grey_frame()

        with patch("tools.vision_benchmark.cv2.imread", return_value=fake_frame):
            # Act
            frames = _load_images(str(tmp_path), limit=3)

        # Assert
        assert len(frames) == 3

    def test_raises_when_directory_has_no_images(self, tmp_path):
        # Arrange — only non-image files
        (tmp_path / "notes.txt").write_text("nothing here")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="No PNG/JPG images found"):
            _load_images(str(tmp_path), limit=None)

    def test_skips_unreadable_images(self, tmp_path):
        # Arrange — two files; imread returns None for one (corrupt)
        (tmp_path / "good.png").write_bytes(b"")
        (tmp_path / "bad.png").write_bytes(b"")
        fake_frame = _make_grey_frame()
        imread_returns = [fake_frame, None]

        with patch("tools.vision_benchmark.cv2.imread", side_effect=imread_returns):
            # Act
            frames = _load_images(str(tmp_path), limit=None)

        # Assert — only the readable frame is included
        assert len(frames) == 1

    def test_ignores_non_image_extensions(self, tmp_path):
        # Arrange
        (tmp_path / "frame.png").write_bytes(b"")
        (tmp_path / "readme.txt").write_text("ignore me")
        fake_frame = _make_grey_frame()

        with patch("tools.vision_benchmark.cv2.imread", return_value=fake_frame):
            # Act
            frames = _load_images(str(tmp_path), limit=None)

        # Assert — .txt not picked up
        assert len(frames) == 1


# ---------------------------------------------------------------------------
# run_engine_benchmark()
# ---------------------------------------------------------------------------

class TestRunEngineBenchmark:
    def _make_mock_engine(self, detections: list[Detection] | None = None):
        eng = MagicMock()
        eng.detect.return_value = detections or []
        return eng

    def test_returns_result_with_correct_frame_count(self):
        # Arrange
        frames = [_make_grey_frame() for _ in range(5)]
        mock_engine = self._make_mock_engine()

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine):
            # Act
            result = run_engine_benchmark("PIXEL", frames, warmup=0)

        # Assert
        assert result.frame_count == 5
        assert result.engine_name == "PIXEL"

    def test_calls_load_once_and_detect_for_all_frames(self):
        # Arrange
        frames = [_make_grey_frame() for _ in range(3)]
        mock_engine = self._make_mock_engine()

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine):
            # Act
            run_engine_benchmark("PIXEL", frames, warmup=0)

        # Assert
        mock_engine.load.assert_called_once()
        assert mock_engine.detect.call_count == 3  # warmup=0, timed=3

    def test_warmup_frames_are_detected_but_not_timed(self):
        # Arrange — 5 frames total, warmup=2 → 2 warmup + 5 timed = 7 total detect calls
        frames = [_make_grey_frame() for _ in range(5)]
        mock_engine = self._make_mock_engine()

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine):
            # Act
            run_engine_benchmark("PIXEL", frames, warmup=2)

        # Assert — detect called for warmup (2) + all frames (5) = 7
        assert mock_engine.detect.call_count == 7

    def test_detections_are_recorded_in_result(self):
        # Arrange
        frames = [_make_grey_frame() for _ in range(2)]
        detections = [_make_detection("DODGE", 0.85)]
        mock_engine = self._make_mock_engine(detections=detections)

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine):
            # Act
            result = run_engine_benchmark("PIXEL", frames, warmup=0)

        # Assert
        assert result.detection_counts.get("DODGE") == 2
        assert pytest.approx(result.confidence_sums["DODGE"], abs=1e-6) == 1.70

    def test_result_is_finalised(self):
        # Arrange
        frames = [_make_grey_frame() for _ in range(4)]
        mock_engine = self._make_mock_engine()

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine):
            # Act
            result = run_engine_benchmark("PIXEL", frames, warmup=0)

        # Assert — fps is computed (non-zero after real perf_counter timing)
        assert result.fps > 0.0


# ---------------------------------------------------------------------------
# run_live_stress_test()
# ---------------------------------------------------------------------------

class TestRunLiveStressTest:
    def _build_mock_sct(self):
        """Build a mock mss context that returns a fake BGRA frame on grab()."""
        fake_bgra = np.zeros((64, 64, 4), dtype=np.uint8)
        mock_sct = MagicMock()
        mock_sct.__enter__.return_value = mock_sct
        mock_sct.__exit__.return_value = False
        mock_sct.monitors = [None, {"top": 0, "left": 0, "width": 64, "height": 64}]
        mock_sct.grab.return_value = fake_bgra
        return mock_sct

    def test_returns_live_stress_result_with_frames_captured(self):
        # Arrange
        mock_engine = MagicMock()
        mock_engine.detect.return_value = []
        mock_sct = self._build_mock_sct()
        fake_grey = _make_grey_frame()

        # perf_counter: deadline_set=0.0, loop_check=0.0 (enter), t0=0.0,
        #               post-detect=0.005, loop_check=1.0 (exit)
        perf_times = [0.0, 0.0, 0.0, 0.005, 1.0]

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine), \
             patch("tools.vision_benchmark.mss.mss", return_value=mock_sct), \
             patch("tools.vision_benchmark.cv2.cvtColor", return_value=fake_grey), \
             patch("tools.vision_benchmark.time.perf_counter", side_effect=perf_times):

            # Act
            result = run_live_stress_test("PIXEL", duration_s=0.5, warmup=0)

        # Assert
        assert result.frame_count == 1
        assert result.engine_name == "PIXEL"
        assert len(result.latencies) == 1
        assert pytest.approx(result.latencies[0], abs=1e-6) == 0.005

    def test_warmup_frames_are_excluded_from_timing(self):
        # Arrange — 2 warmup frames, then 1 timed frame
        mock_engine = MagicMock()
        mock_engine.detect.return_value = []
        mock_sct = self._build_mock_sct()
        fake_grey = _make_grey_frame()

        # perf_counter: deadline_set, loop_check (enter), t0, post, loop_check (exit)
        perf_times = [0.0, 0.0, 0.0, 0.005, 1.0]

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine), \
             patch("tools.vision_benchmark.mss.mss", return_value=mock_sct), \
             patch("tools.vision_benchmark.cv2.cvtColor", return_value=fake_grey), \
             patch("tools.vision_benchmark.time.perf_counter", side_effect=perf_times):

            # Act
            result = run_live_stress_test("PIXEL", duration_s=0.5, warmup=2)

        # Assert — 2 warmup grabs + 1 timed grab = 3 total grab calls
        assert mock_sct.grab.call_count == 3
        assert result.frame_count == 1

    def test_result_is_finalised_after_run(self):
        # Arrange
        mock_engine = MagicMock()
        mock_engine.detect.return_value = []
        mock_sct = self._build_mock_sct()
        fake_grey = _make_grey_frame()
        perf_times = [0.0, 0.0, 0.0, 0.010, 1.0]

        with patch("tools.vision_benchmark.vision.registry.create", return_value=mock_engine), \
             patch("tools.vision_benchmark.mss.mss", return_value=mock_sct), \
             patch("tools.vision_benchmark.cv2.cvtColor", return_value=fake_grey), \
             patch("tools.vision_benchmark.time.perf_counter", side_effect=perf_times):

            # Act
            result = run_live_stress_test("PIXEL", duration_s=0.5, warmup=0)

        # Assert — fps is derived from 10 ms latency = 100 FPS
        assert pytest.approx(result.fps, abs=0.1) == 100.0


# ---------------------------------------------------------------------------
# _save_csv()
# ---------------------------------------------------------------------------

class TestSaveCsv:
    def _make_result(self, engine: str, counts: dict[str, int], sums: dict[str, float]) -> EngineResult:
        r = EngineResult(engine_name=engine, frame_count=10)
        r.detection_counts = counts
        r.confidence_sums = sums
        r.fps = 100.0
        r.mean_ms = 10.0
        r.median_ms = 10.0
        r.p95_ms = 12.0
        r.max_ms = 15.0
        return r

    def test_writes_header_and_data_row(self, tmp_path):
        # Arrange
        csv_path = str(tmp_path / "results.csv")
        result = self._make_result("PIXEL", {"DODGE": 5}, {"DODGE": 4.0})

        # Act
        _save_csv([result], csv_path)

        # Assert
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "engine"
        assert "det_DODGE" in rows[0]
        assert rows[1][0] == "PIXEL"

    def test_creates_parent_directory_if_missing(self, tmp_path):
        # Arrange
        csv_path = str(tmp_path / "subdir" / "out.csv")
        result = self._make_result("ORB", {}, {})

        # Act
        _save_csv([result], csv_path)

        # Assert
        assert (tmp_path / "subdir" / "out.csv").exists()

    def test_empty_confidence_cell_when_label_not_detected(self, tmp_path):
        # Arrange — result has a label with 0 detections
        csv_path = str(tmp_path / "out.csv")
        r1 = self._make_result("PIXEL", {"DODGE": 3}, {"DODGE": 2.4})
        r2 = self._make_result("ORB",   {"DODGE": 0}, {})

        # Act
        _save_csv([r1, r2], csv_path)

        # Assert — ORB confidence cell for DODGE should be empty string
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        conf_col = rows[0].index("conf_DODGE")
        assert rows[2][conf_col] == ""

    def test_multiple_engines_produce_multiple_rows(self, tmp_path):
        # Arrange
        csv_path = str(tmp_path / "multi.csv")
        r1 = self._make_result("PIXEL", {}, {})
        r2 = self._make_result("SIFT",  {}, {})

        # Act
        _save_csv([r1, r2], csv_path)

        # Assert — 1 header + 2 data rows
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# _print_report() and _print_live_report()
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_print_report_contains_engine_name(self, capsys):
        # Arrange
        r = EngineResult(engine_name="PIXEL", frame_count=10, fps=100.0, mean_ms=10.0,
                         median_ms=10.0, p95_ms=11.0, max_ms=15.0)
        r.detection_counts = {"DODGE": 3}
        r.confidence_sums  = {"DODGE": 2.4}

        # Act
        _print_report([r])

        # Assert
        captured = capsys.readouterr().out
        assert "PIXEL" in captured
        assert "VISION ENGINE BENCHMARK REPORT" in captured
        assert "DODGE" in captured

    def test_print_live_report_shows_recommendation(self, capsys):
        # Arrange — 80 FPS → recommends 60 Hz
        r = LiveStressResult(engine_name="PIXEL", duration_s=10.0, frame_count=800,
                             fps=80.0, mean_ms=12.5, median_ms=12.0,
                             p95_ms=15.0, max_ms=20.0, recommended_hz=60)

        # Act
        _print_live_report(r)

        # Assert
        captured = capsys.readouterr().out
        assert "60 Hz" in captured
        assert "PIXEL" in captured
        assert "LIVE CAPTURE STRESS TEST RESULTS" in captured

    def test_print_live_report_shows_below_threshold_message(self, capsys):
        # Arrange — 15 FPS → recommended_hz = 0
        r = LiveStressResult(engine_name="SIFT", duration_s=10.0, frame_count=150,
                             fps=15.0, mean_ms=66.0, median_ms=65.0,
                             p95_ms=80.0, max_ms=100.0, recommended_hz=0)

        # Act
        _print_live_report(r)

        # Assert
        captured = capsys.readouterr().out
        assert "Below 20 Hz threshold" in captured
