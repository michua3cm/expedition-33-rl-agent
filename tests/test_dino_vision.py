"""
Unit tests for vision/engines/dino.py (DINOv2VisionEngine).

Heavy deps (torch, cv2) are mocked via sys.modules injection — tests run in
CI with only the dev group installed.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared stub factory
# ---------------------------------------------------------------------------

def _dummy_bgr(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_stubs(emb_dim: int = 384):
    """
    Return (stubs, mock_torch, mock_model, mock_cv2).

    The fake DINOv2 model always returns the same unit vector so that cosine
    similarity with itself equals 1.0 (above any reasonable threshold).
    """
    unit_emb = np.ones(emb_dim, dtype=np.float32) / np.sqrt(emb_dim)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.side_effect = lambda s: s
    mock_torch.from_numpy.return_value.to.return_value = MagicMock()

    raw_features = MagicMock()
    normalised = MagicMock()
    raw_features.__truediv__ = MagicMock(return_value=normalised)
    normalised.squeeze.return_value.cpu.return_value.numpy.return_value = unit_emb.copy()

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.return_value = raw_features          # model(tensor) → raw_features

    mock_torch.hub.load.return_value = mock_model

    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = _dummy_bgr()
    mock_cv2.resize.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_cv2.cvtColor.return_value = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_cv2.IMREAD_COLOR = 1
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.INTER_CUBIC = 2

    stubs = {
        "torch": mock_torch,
        "cv2": mock_cv2,
    }
    return stubs, mock_torch, mock_model, mock_cv2


def _fresh_engine(stubs):
    sys.modules.pop("vision.engines.dino", None)
    from vision.engines.dino import DINOv2VisionEngine
    return DINOv2VisionEngine


def _load_engine(eng, tmp_path, files=None, label="JUMP_CUE"):
    if files is None:
        (tmp_path / "ref.png").touch()
        file_cfg = {"file": "ref.png"}
    else:
        for f in files:
            (tmp_path / f).touch()
        file_cfg = {"files": files}
    targets = {label: {**file_cfg, "roi": (0.15, 0.15, 0.70, 0.50)}}
    eng.load(targets, str(tmp_path))


# ===========================================================================
# Properties
# ===========================================================================

class TestDINOv2VisionEngineProperties:
    def test_name_is_dino(self):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            assert Cls().name == "DINO"

    def test_needs_color_is_true(self):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            assert Cls().needs_color is True


# ===========================================================================
# load()
# ===========================================================================

class TestDINOv2VisionEngineLoad:
    def test_dinov2_vits14_model_loaded(self, tmp_path):
        from unittest.mock import patch
        stubs, mock_torch, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            _load_engine(eng, tmp_path)
        mock_torch.hub.load.assert_called_once_with(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )

    def test_single_file_key_creates_one_embedding(self, tmp_path):
        from unittest.mock import patch
        stubs, _, mock_model, _ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            _load_engine(eng, tmp_path, label="JUMP_CUE")
        assert len(eng._ref_embeddings["JUMP_CUE"]) == 1

    def test_files_key_creates_multiple_embeddings(self, tmp_path):
        from unittest.mock import patch
        stubs, _, mock_model, _ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            _load_engine(eng, tmp_path, files=["r1.png", "r2.png", "r3.png"])
        assert len(eng._ref_embeddings["JUMP_CUE"]) == 3

    def test_encode_called_once_per_reference(self, tmp_path):
        from unittest.mock import patch
        stubs, _, mock_model, _ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            _load_engine(eng, tmp_path, files=["a.png", "b.png"])
        assert mock_model.call_count == 2

    def test_target_without_roi_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            (tmp_path / "t.png").touch()
            eng.load({"X": {"file": "t.png"}}, str(tmp_path))
        assert eng._ref_embeddings == {}

    def test_target_without_file_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            eng.load({"X": {"roi": (0.0, 0.0, 1.0, 1.0)}}, str(tmp_path))
        assert eng._ref_embeddings == {}

    def test_unreadable_template_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, _, _, mock_cv2 = _make_stubs()
        mock_cv2.imread.return_value = None
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)()
            (tmp_path / "bad.png").touch()
            eng.load({"X": {"file": "bad.png", "roi": (0.0, 0.0, 1.0, 1.0)}}, str(tmp_path))
        assert eng._ref_embeddings == {}


# ===========================================================================
# detect()
# ===========================================================================

class TestDINOv2VisionEngineDetect:
    @pytest.fixture()
    def loaded_engine(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)(threshold=0.75)
            _load_engine(eng, tmp_path, files=["r1.png", "r2.png"])
        return eng

    def test_detection_returned_above_threshold(self, loaded_engine):
        # same unit vector → dot product 1.0 > 0.75
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert len(dets) == 1
        assert dets[0].label == "JUMP_CUE"

    def test_no_detection_below_threshold(self, tmp_path):
        from unittest.mock import MagicMock, patch
        stubs, _, mock_model, _ = _make_stubs()

        # All references return e1; crop returns e2; e1·e2 = 0 < threshold
        e1 = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        e2 = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        call_count = 0

        def _side(tensor):
            nonlocal call_count
            call_count += 1
            # First two calls are template loads (r1, r2), rest are detections
            emb = e1 if call_count <= 2 else e2
            raw = MagicMock()
            norm = MagicMock()
            norm.squeeze.return_value.cpu.return_value.numpy.return_value = emb
            raw.__truediv__ = MagicMock(return_value=norm)
            return raw

        mock_model.side_effect = _side
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)(threshold=0.75)
            _load_engine(eng, tmp_path, files=["r1.png", "r2.png"])
        dets = eng.detect(_dummy_bgr(480, 640))
        assert dets == []

    def test_uses_max_similarity_across_references(self, tmp_path):
        """Fires when the best reference matches, even if others don't."""
        from unittest.mock import MagicMock, patch
        stubs, _, mock_model, _ = _make_stubs()

        # ref1 = e1, ref2 = e2, crop = e1  → max(e1·e1, e2·e1) = max(1, 0) = 1 → fires
        e1 = np.array([1.0] + [0.0] * 383, dtype=np.float32)
        e2 = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)
        call_count = 0

        def _side(tensor):
            nonlocal call_count
            call_count += 1
            emb = {1: e1, 2: e2, 3: e1}.get(call_count, e1)
            raw = MagicMock()
            norm = MagicMock()
            norm.squeeze.return_value.cpu.return_value.numpy.return_value = emb
            raw.__truediv__ = MagicMock(return_value=norm)
            return raw

        mock_model.side_effect = _side
        with patch.dict(sys.modules, stubs):
            eng = _fresh_engine(stubs)(threshold=0.75)
            _load_engine(eng, tmp_path, files=["r1.png", "r2.png"])
        dets = eng.detect(_dummy_bgr(480, 640))
        assert len(dets) == 1

    def test_coordinates_are_full_frame(self, loaded_engine):
        # roi=(0.15, 0.15, 0.70, 0.50) on 640×480
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert len(dets) == 1
        det = dets[0]
        assert det.x == int(0.15 * 640)
        assert det.y == int(0.15 * 480)
        assert det.w == int(0.70 * 640)
        assert det.h == int(0.50 * 480)

    def test_confidence_equals_max_similarity(self, loaded_engine):
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert dets[0].confidence == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Registry integration
# ===========================================================================

class TestDINOv2Registration:
    def test_registered_as_dino(self):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("vision.engines.dino", None)
            import vision.registry as registry
            from vision.engines.dino import DINOv2VisionEngine  # noqa: F401
            eng = registry.create("DINO")
        assert eng.name == "DINO"
