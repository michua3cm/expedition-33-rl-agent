"""
Unit tests for vision/engines/clip.py (CLIPVisionEngine).

Heavy deps (open_clip, torch, PIL, cv2) are mocked via sys.modules injection —
tests run in CI with only the dev group installed.
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


def _make_stubs():
    """
    Return (stubs, mock_open_clip, mock_model).

    The fake CLIP model's encode_image() always returns a unit vector so that
    cosine similarity with itself equals 1.0 (above any reasonable threshold).
    """
    unit_emb = np.ones(512, dtype=np.float32) / np.sqrt(512)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.side_effect = lambda s: s

    raw_features = MagicMock()
    normalised = MagicMock()
    raw_features.__truediv__ = MagicMock(return_value=normalised)
    normalised.squeeze.return_value.cpu.return_value.numpy.return_value = unit_emb.copy()

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.encode_image.return_value = raw_features

    mock_preprocess = MagicMock()
    mock_preprocess.return_value.unsqueeze.return_value.to.return_value = MagicMock()

    mock_open_clip = MagicMock()
    mock_open_clip.create_model_and_transforms.return_value = (
        mock_model, MagicMock(), mock_preprocess
    )

    mock_cv2 = MagicMock()
    mock_cv2.imread.return_value = _dummy_bgr()
    mock_cv2.cvtColor.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_cv2.IMREAD_COLOR = 1
    mock_cv2.COLOR_BGR2RGB = 4

    stubs = {
        "torch": mock_torch,
        "open_clip": mock_open_clip,
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
        "cv2": mock_cv2,
    }
    return stubs, mock_open_clip, mock_model, mock_cv2


def _fresh_engine(stubs):
    """Pop and reimport CLIPVisionEngine inside the stubs context."""
    sys.modules.pop("vision.engines.clip", None)
    from vision.engines.clip import CLIPVisionEngine
    return CLIPVisionEngine


def _load_engine(eng, tmp_path, label="PERFECT"):
    (tmp_path / "template.png").touch()   # os.path.exists check only
    targets = {label: {"file": "template.png", "roi": (0.1, 0.3, 0.8, 0.4)}}
    eng.load(targets, str(tmp_path))


# ===========================================================================
# Properties
# ===========================================================================

class TestCLIPVisionEngineProperties:
    def test_name_is_clip(self):
        stubs, *_ = _make_stubs()
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            assert Cls().name == "CLIP"

    def test_needs_color_is_true(self):
        stubs, *_ = _make_stubs()
        with __import__("unittest.mock", fromlist=["patch"]).patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            assert Cls().needs_color is True


# ===========================================================================
# load()
# ===========================================================================

class TestCLIPVisionEngineLoad:
    def test_vit_b32_model_created(self, tmp_path):
        from unittest.mock import patch
        stubs, mock_open_clip, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls()
            _load_engine(eng, tmp_path)
        mock_open_clip.create_model_and_transforms.assert_called_once_with(
            "ViT-B-32", pretrained="openai"
        )

    def test_encode_image_called_for_template(self, tmp_path):
        from unittest.mock import patch
        stubs, _, mock_model, _ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls()
            _load_engine(eng, tmp_path)
        mock_model.encode_image.assert_called_once()

    def test_missing_template_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, _, _, mock_cv2 = _make_stubs()
        mock_cv2.imread.return_value = None   # simulate unreadable file
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls()
            (tmp_path / "bad.png").touch()
            eng.load({"X": {"file": "bad.png", "roi": (0.0, 0.0, 1.0, 1.0)}}, str(tmp_path))
        assert eng._ref_embeddings == {}

    def test_target_without_roi_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls()
            (tmp_path / "t.png").touch()
            eng.load({"NOROI": {"file": "t.png"}}, str(tmp_path))
        assert eng._ref_embeddings == {}

    def test_target_without_file_skipped(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls()
            eng.load({"NOFILE": {"roi": (0.0, 0.0, 1.0, 1.0)}}, str(tmp_path))
        assert eng._ref_embeddings == {}


# ===========================================================================
# detect()
# ===========================================================================

class TestCLIPVisionEngineDetect:
    @pytest.fixture()
    def loaded_engine(self, tmp_path):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls(threshold=0.75)
            _load_engine(eng, tmp_path)
        return eng

    def test_detection_returned_above_threshold(self, loaded_engine):
        # unit vector dot unit vector = 1.0 > 0.75 threshold
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert len(dets) == 1
        assert dets[0].label == "PERFECT"

    def test_no_detection_below_threshold(self, tmp_path):
        from unittest.mock import MagicMock, patch

        stubs, _, mock_model, _ = _make_stubs()

        # First call (template load): emb = e1.  Second call (crop): emb = e2.
        # e1 · e2 = 0 < 0.75
        call_count = 0
        e1 = np.array([1.0] + [0.0] * 511, dtype=np.float32)
        e2 = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)

        def _side(t):
            nonlocal call_count
            call_count += 1
            emb = e1 if call_count == 1 else e2
            raw = MagicMock()
            norm = MagicMock()
            norm.squeeze.return_value.cpu.return_value.numpy.return_value = emb
            raw.__truediv__ = MagicMock(return_value=norm)
            return raw

        mock_model.encode_image.side_effect = _side

        with patch.dict(sys.modules, stubs):
            Cls = _fresh_engine(stubs)
            eng = Cls(threshold=0.75)
            _load_engine(eng, tmp_path)
        dets = eng.detect(_dummy_bgr(480, 640))
        assert dets == []

    def test_coordinates_are_full_frame(self, loaded_engine):
        # roi=(0.1, 0.3, 0.8, 0.4) on 640×480
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert len(dets) == 1
        det = dets[0]
        assert det.x == int(0.1 * 640)
        assert det.y == int(0.3 * 480)
        assert det.w == int(0.8 * 640)
        assert det.h == int(0.4 * 480)

    def test_confidence_equals_cosine_similarity(self, loaded_engine):
        dets = loaded_engine.detect(_dummy_bgr(480, 640))
        assert dets[0].confidence == pytest.approx(1.0, abs=1e-4)


# ===========================================================================
# Registry integration
# ===========================================================================

class TestCLIPRegistration:
    def test_registered_as_clip(self):
        from unittest.mock import patch
        stubs, *_ = _make_stubs()
        with patch.dict(sys.modules, stubs):
            sys.modules.pop("vision.engines.clip", None)
            # Re-register by importing the engine class directly
            import vision.registry as registry
            from vision.engines.clip import CLIPVisionEngine  # noqa: F401
            eng = registry.create("CLIP")
        assert eng.name == "CLIP"
