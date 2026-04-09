"""
Unit tests for tools/auto_label.py — _detection_to_yolo and _write_dataset_yaml.

File I/O is fully mocked — no images or YAML files are written to disk.
"""

from unittest.mock import mock_open, patch

from tools.auto_label import CLASS_ID, CLASS_NAMES, _detection_to_yolo, _write_dataset_yaml
from vision.engine import Detection


def _det(label: str, x: int, y: int, w: int, h: int) -> Detection:
    return Detection(label=label, x=x, y=y, w=w, h=h, confidence=0.9)


# ---------------------------------------------------------------------------
# _detection_to_yolo
# ---------------------------------------------------------------------------

class TestDetectionToYolo:
    def test_output_has_five_space_separated_fields(self):
        det = _det("DODGE", x=0, y=0, w=10, h=10)
        line = _detection_to_yolo(det, img_w=100, img_h=100)
        assert len(line.split()) == 5

    def test_class_id_is_correct(self):
        for label in CLASS_NAMES:
            det = _det(label, x=0, y=0, w=10, h=10)
            line = _detection_to_yolo(det, img_w=100, img_h=100)
            assert int(line.split()[0]) == CLASS_ID[label]

    def test_x_center_normalized(self):
        # x=40, w=20 → x_center = (40 + 10) / 200 = 0.25
        det = _det("DODGE", x=40, y=0, w=20, h=10)
        line = _detection_to_yolo(det, img_w=200, img_h=100)
        assert abs(float(line.split()[1]) - 0.25) < 1e-5

    def test_y_center_normalized(self):
        # y=30, h=40 → y_center = (30 + 20) / 100 = 0.5
        det = _det("DODGE", x=0, y=30, w=10, h=40)
        line = _detection_to_yolo(det, img_w=100, img_h=100)
        assert abs(float(line.split()[2]) - 0.5) < 1e-5

    def test_width_normalized(self):
        # w=80, img_w=200 → 0.4
        det = _det("DODGE", x=0, y=0, w=80, h=10)
        line = _detection_to_yolo(det, img_w=200, img_h=100)
        assert abs(float(line.split()[3]) - 0.4) < 1e-5

    def test_height_normalized(self):
        # h=50, img_h=200 → 0.25
        det = _det("DODGE", x=0, y=0, w=10, h=50)
        line = _detection_to_yolo(det, img_w=100, img_h=200)
        assert abs(float(line.split()[4]) - 0.25) < 1e-5

    def test_full_frame_detection_gives_center_half_half(self):
        # x=0, y=0, w=img_w, h=img_h → x_center=0.5, y_center=0.5, w=1.0, h=1.0
        det = _det("PERFECT", x=0, y=0, w=100, h=100)
        line = _detection_to_yolo(det, img_w=100, img_h=100)
        parts = line.split()
        assert abs(float(parts[1]) - 0.5) < 1e-5
        assert abs(float(parts[2]) - 0.5) < 1e-5
        assert abs(float(parts[3]) - 1.0) < 1e-5
        assert abs(float(parts[4]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# _write_dataset_yaml
# ---------------------------------------------------------------------------

class TestWriteDatasetYaml:
    def test_yaml_contains_all_class_names(self):
        m = mock_open()
        with patch("builtins.open", m):
            _write_dataset_yaml()

        written = m().write.call_args[0][0]
        for name in CLASS_NAMES:
            assert name in written

    def test_yaml_contains_nc_field(self):
        m = mock_open()
        with patch("builtins.open", m):
            _write_dataset_yaml()

        written = m().write.call_args[0][0]
        assert f"nc: {len(CLASS_NAMES)}" in written

    def test_yaml_contains_train_and_val_keys(self):
        m = mock_open()
        with patch("builtins.open", m):
            _write_dataset_yaml()

        written = m().write.call_args[0][0]
        assert "train:" in written
        assert "val:" in written

    def test_class_ids_are_sequential_from_zero(self):
        m = mock_open()
        with patch("builtins.open", m):
            _write_dataset_yaml()

        written = m().write.call_args[0][0]
        for i in range(len(CLASS_NAMES)):
            assert f"  {i}:" in written
