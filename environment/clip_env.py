from __future__ import annotations

import numpy as np
from gymnasium import spaces

from environment.gym_env import Expedition33Env

OBS_DIM = 512  # CLIP ViT-B/32 image embedding dimension


class CLIPExpedition33Env(Expedition33Env):
    """
    Expedition33Env with a CLIP ViT-B/32 image embedding as the observation.

    Replaces the 30-dim detection vector with a 512-dim L2-normalised image
    embedding.  The underlying detection engine still runs for the reward
    signal; only the observation space changes.

    The same policy architecture works on a robot camera without retraining
    the perception module — swap the game screen for a camera frame and the
    CLIP embedding is identical in meaning.

    Install deps: uv sync --group cuda --group clip
    """

    def __init__(
        self,
        engine: str = "PIXEL",
        roi=None,
        step_delay: float = 0.15,
    ) -> None:
        # include_frame=True so GameState carries the raw BGR frame for CLIP.
        super().__init__(
            engine=engine,
            roi=roi,
            step_delay=step_delay,
            include_frame=True,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self._init_clip()

    def _init_clip(self) -> None:
        import open_clip
        import torch

        self._torch = torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_device = torch.device(device_str)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._clip_model = model.to(self._clip_device).eval()
        self._clip_preprocess = preprocess

    def _build_obs(self, game_state) -> np.ndarray:
        frame = game_state.frame
        if frame is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self._clip_preprocess(img).unsqueeze(0).to(self._clip_device)

        with self._torch.no_grad():
            features = self._clip_model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy().astype(np.float32)
