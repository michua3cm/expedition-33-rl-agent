"""
Unit tests for environment/clip_env.py.

open_clip, torch, and PIL are mocked via sys.modules injection — no GPU or
clip group install required to run in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

OBS_DIM = 512


# ---------------------------------------------------------------------------
# Stub factory
# ---------------------------------------------------------------------------

def _make_clip_stubs():
    """
    Return (stubs, mock_model, mock_preprocess, mock_open_clip).

    Sets up the full mock chain so that:
      encode_image(tensor) / norm(...)  →  squeeze → cpu → numpy
      → np.zeros(512, float32)
    """
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.device.side_effect = str  # torch.device("cpu") -> "cpu"

    np_features = np.zeros(OBS_DIM, dtype=np.float32)
    mock_normalized = MagicMock()
    mock_normalized.squeeze.return_value.cpu.return_value.numpy.return_value = np_features

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_features = MagicMock()
    mock_features.__truediv__ = MagicMock(return_value=mock_normalized)
    mock_model.encode_image.return_value = mock_features

    mock_preprocess = MagicMock()

    mock_open_clip = MagicMock()
    mock_open_clip.create_model_and_transforms.return_value = (
        mock_model, MagicMock(), mock_preprocess
    )

    mock_pil = MagicMock()

    stubs = {
        "torch": mock_torch,
        "open_clip": mock_open_clip,
        "PIL": mock_pil,
        "PIL.Image": mock_pil.Image,
    }
    return stubs, mock_model, mock_preprocess, mock_open_clip


# ===========================================================================
# TestCLIPExpedition33Env
# ===========================================================================

class TestCLIPExpedition33Env:
    @pytest.fixture(autouse=True)
    def _setup(self, mocker):
        stubs, mock_model, mock_preprocess, mock_open_clip = _make_clip_stubs()
        self.mock_model = mock_model
        self.mock_preprocess = mock_preprocess
        self.mock_open_clip = mock_open_clip

        mock_instance = MagicMock()
        mock_instance.monitor = {"width": 100, "height": 100}
        mocker.patch("environment.gym_env.GameInstance", return_value=mock_instance)

        with patch.dict(sys.modules, stubs):
            sys.modules.pop("environment.clip_env", None)
            yield

    def test_observation_space_shape_and_dtype(self):
        from environment.clip_env import CLIPExpedition33Env

        env = CLIPExpedition33Env()

        assert env.observation_space.shape == (OBS_DIM,)
        assert env.observation_space.dtype == np.float32

    def test_action_space_unchanged(self):
        from environment.clip_env import CLIPExpedition33Env

        env = CLIPExpedition33Env()

        assert env.action_space.n == 7

    def test_clip_initialized_with_vit_b_32(self):
        from environment.clip_env import CLIPExpedition33Env

        CLIPExpedition33Env()

        self.mock_open_clip.create_model_and_transforms.assert_called_once_with(
            "ViT-B-32", pretrained="openai"
        )

    def test_build_obs_returns_zeros_when_frame_is_none(self):
        from environment.clip_env import CLIPExpedition33Env

        env = CLIPExpedition33Env()
        state = MagicMock()
        state.frame = None

        obs = env._build_obs(state)

        assert obs.shape == (OBS_DIM,)
        np.testing.assert_array_equal(obs, np.zeros(OBS_DIM, dtype=np.float32))

    def test_build_obs_calls_encode_image_and_returns_correct_shape(self):
        from environment.clip_env import CLIPExpedition33Env

        env = CLIPExpedition33Env()
        state = MagicMock()
        state.frame = np.zeros((100, 100, 3), dtype=np.uint8)

        obs = env._build_obs(state)

        self.mock_model.encode_image.assert_called_once()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_build_obs_l2_normalizes_features(self):
        # norm() must be called on the raw features before returning.
        from environment.clip_env import CLIPExpedition33Env

        env = CLIPExpedition33Env()
        state = MagicMock()
        state.frame = np.zeros((100, 100, 3), dtype=np.uint8)

        env._build_obs(state)

        self.mock_model.encode_image.return_value.norm.assert_called()
