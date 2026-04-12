"""
Unit tests for tools/dataset_status.py.

All filesystem access is mocked — no real label files are read or written.
"""

from __future__ import annotations

from unittest.mock import mock_open, patch

from tools.dataset_status import _count_labels, _print_table, run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label_line(class_id: int, x: float = 0.5, y: float = 0.5,
                     w: float = 0.1, h: float = 0.1) -> str:
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# _count_labels
# ---------------------------------------------------------------------------

class TestCountLabels:
    def test_returns_empty_dicts_when_dir_does_not_exist(self):
        with patch("tools.dataset_status.os.path.isdir", return_value=False):
            inst, imgs = _count_labels(["nonexistent/"])

        assert inst == {}
        assert imgs == {}

    def test_counts_single_instance_in_one_file(self):
        # DODGE = class_id 0
        label_content = _make_label_line(0)

        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir", return_value=["frame_001.txt"]),
            patch("builtins.open", mock_open(read_data=label_content)),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        assert inst["DODGE"] == 1
        assert imgs["DODGE"] == 1

    def test_counts_multiple_instances_of_same_class_in_one_file(self):
        # Two DODGE detections in the same file
        label_content = _make_label_line(0) + "\n" + _make_label_line(0)

        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir", return_value=["frame_001.txt"]),
            patch("builtins.open", mock_open(read_data=label_content)),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        # Two instances but still only one image
        assert inst["DODGE"] == 2
        assert imgs["DODGE"] == 1

    def test_counts_across_multiple_files(self):
        # Two files, each with one PERFECT detection (class_id 1)
        label_content = _make_label_line(1)

        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir",
                  return_value=["frame_001.txt", "frame_002.txt"]),
            patch("builtins.open", mock_open(read_data=label_content)),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        assert inst["PERFECT"] == 2
        assert imgs["PERFECT"] == 2

    def test_merges_counts_across_multiple_dirs(self):
        # One DODGE in train/, one DODGE in val/
        label_content = _make_label_line(0)

        def _isdir(path: str) -> bool:
            return path in ("labels/train/", "labels/val/")

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.txt"]),
            patch("builtins.open", mock_open(read_data=label_content)),
        ):
            inst, imgs = _count_labels(["labels/train/", "labels/val/"])

        assert inst["DODGE"] == 2
        assert imgs["DODGE"] == 2

    def test_ignores_non_txt_files(self):
        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir",
                  return_value=["frame_001.png", "frame_001.jpg", "notes.md"]),
            patch("builtins.open", mock_open(read_data="")),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        assert inst == {}
        assert imgs == {}

    def test_ignores_empty_label_files(self):
        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir", return_value=["frame_001.txt"]),
            patch("builtins.open", mock_open(read_data="")),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        assert inst == {}
        assert imgs == {}

    def test_ignores_out_of_range_class_ids(self):
        # class_id 999 is beyond YOLO_CLASSES length
        label_content = "999 0.5 0.5 0.1 0.1"

        with (
            patch("tools.dataset_status.os.path.isdir", return_value=True),
            patch("tools.dataset_status.os.listdir", return_value=["frame_001.txt"]),
            patch("builtins.open", mock_open(read_data=label_content)),
        ):
            inst, imgs = _count_labels(["labels/labeled/"])

        assert inst == {}
        assert imgs == {}


# ---------------------------------------------------------------------------
# _print_table — status label logic
# ---------------------------------------------------------------------------

class TestPrintTable:
    def _captured_output(self, instances: dict, images: dict,
                         target: int = 50) -> str:
        """Run _print_table and return all printed text as a single string."""
        lines: list[str] = []
        with patch("builtins.print", side_effect=lambda *a: lines.append(str(a[0]))):
            _print_table("Test Title", instances, images, target)
        return "\n".join(lines)

    def test_missing_class_shows_missing_status(self):
        output = self._captured_output({}, {})
        assert "MISSING" in output

    def test_class_below_target_shows_low_status(self):
        output = self._captured_output({"DODGE": 10}, {"DODGE": 8}, target=50)
        assert "LOW" in output
        assert "40 more" in output

    def test_class_at_target_shows_ok_status(self):
        # Build a full instances dict with all classes at target
        from calibration.config import YOLO_CLASSES
        instances = {name: 50 for name in YOLO_CLASSES}
        images    = {name: 40 for name in YOLO_CLASSES}
        output = self._captured_output(instances, images, target=50)
        assert "OK" in output
        assert "MISSING" not in output
        assert "LOW" not in output

    def test_all_ok_prints_ready_message(self):
        from calibration.config import YOLO_CLASSES
        instances = {name: 50 for name in YOLO_CLASSES}
        images    = {name: 40 for name in YOLO_CLASSES}
        output = self._captured_output(instances, images, target=50)
        assert "Ready to train" in output

    def test_need_count_is_correct(self):
        # 18 instances, target 50 → need 32 more
        output = self._captured_output({"DODGE": 18}, {"DODGE": 14}, target=50)
        assert "32 more" in output

    def test_custom_target_is_respected(self):
        # 10 instances, target 30 → need 20 more
        output = self._captured_output({"DODGE": 10}, {"DODGE": 8}, target=30)
        assert "20 more" in output


# ---------------------------------------------------------------------------
# run — top-level routing
# ---------------------------------------------------------------------------

class TestRun:
    def test_prints_no_data_message_when_no_label_dirs_exist(self, capsys):
        with patch("tools.dataset_status.os.path.isdir", return_value=False):
            run()

        captured = capsys.readouterr().out
        assert "No label files found" in captured

    def test_prints_no_data_message_when_dirs_empty(self, capsys):
        def _isdir(path: str) -> bool:
            return True

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.png"]),
        ):
            run()

        captured = capsys.readouterr().out
        assert "No label files found" in captured

    def test_reports_pre_autolabel_source_when_labeled_dir_has_data(self, capsys):
        from tools.dataset_status import YOLO_LABELED_LABELS_DIR

        def _isdir(path: str) -> bool:
            return path == YOLO_LABELED_LABELS_DIR

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.txt"]),
            patch("tools.dataset_status._count_labels", return_value=({}, {})),
            patch("tools.dataset_status._print_table") as mock_table,
        ):
            run()

        assert mock_table.call_count == 1
        assert "Pre-autolabel" in mock_table.call_args[0][0]

    def test_reports_post_autolabel_source_when_train_dir_has_data(self, capsys):
        from tools.dataset_status import TRAIN_LABELS

        def _isdir(path: str) -> bool:
            return path == TRAIN_LABELS

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.txt"]),
            patch("tools.dataset_status._count_labels", return_value=({}, {})),
            patch("tools.dataset_status._print_table") as mock_table,
        ):
            run()

        assert mock_table.call_count == 1
        assert "Post-autolabel" in mock_table.call_args[0][0]

    def test_reports_both_sources_when_both_have_data(self, capsys):
        from tools.dataset_status import TRAIN_LABELS, YOLO_LABELED_LABELS_DIR

        def _isdir(path: str) -> bool:
            return path in (YOLO_LABELED_LABELS_DIR, TRAIN_LABELS)

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.txt"]),
            patch("tools.dataset_status._count_labels", return_value=({}, {})),
            patch("tools.dataset_status._print_table") as mock_table,
        ):
            run()

        assert mock_table.call_count == 2

    def test_custom_target_passed_to_print_table(self):
        from tools.dataset_status import YOLO_LABELED_LABELS_DIR

        def _isdir(path: str) -> bool:
            return path == YOLO_LABELED_LABELS_DIR

        with (
            patch("tools.dataset_status.os.path.isdir", side_effect=_isdir),
            patch("tools.dataset_status.os.listdir", return_value=["frame.txt"]),
            patch("tools.dataset_status._count_labels", return_value=({}, {})),
            patch("tools.dataset_status._print_table") as mock_table,
        ):
            run(target=30)

        # target=30 must be forwarded to _print_table
        assert mock_table.call_args[0][3] == 30
