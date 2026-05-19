"""
Diffusion Policy trainer for Clair Obscur: Expedition 33.

Architecture: 1D Convolutional UNet with FiLM conditioning (Chi et al. 2023).
- obs_horizon=2:   conditions on the last 2 observations
- pred_horizon=16: predicts 16 future actions per inference call
- action_horizon=8: executes the first 8 predictions before replanning

Training objective: predict noise (ε-parametrisation, DDPM).
Inference: DDIM with 10 steps for fast sampling.

Action space: discrete integers [0, num_actions-1] normalised to [-1, 1]
for the diffusion process; rounded back to int at prediction time.

Usage
-----
    dp = DiffusionPolicy(obs_dim=9, num_actions=7)
    dp.train_from_demos("data/demos/", epochs=200)
    action = dp.predict(obs_deque)   # collections.deque of obs arrays
    dp.save("data/models/dp_20260519.pt")
"""

import math
import os
from collections import deque
from datetime import datetime

import numpy as np

# All torch imports are deferred to first use so that this module can be
# imported in CI without torch installed (tests mock sys.modules).

OBS_HORIZON = 2
PRED_HORIZON = 16
ACTION_HORIZON = 8
NUM_DIFFUSION_STEPS = 100
DDIM_STEPS = 10


# ── Architecture ──────────────────────────────────────────────────────────────


def _make_noise_net(action_dim: int, cond_dim: int, hidden: int = 256):
    """Return a ConditionalUNet1D instance."""
    return ConditionalUNet1D(action_dim=action_dim, cond_dim=cond_dim, hidden=hidden)


class _FiLMResBlock:
    """Residual block with FiLM (Feature-wise Linear Modulation) conditioning."""


class ConditionalUNet1D:
    """
    1D Conv UNet that denoises action sequences conditioned on observations
    and a diffusion timestep.

    Input :  noisy_actions (B, T, action_dim)
    Output:  predicted noise (B, T, action_dim)
    """


def _build_unet(action_dim: int, cond_dim: int, hidden: int = 256):
    """Build and return the full noise prediction network as an nn.Module."""
    import torch.nn as nn

    class _FiLMBlock(nn.Module):
        def __init__(self, dim, c_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1),
                nn.GroupNorm(min(8, dim), dim),
                nn.Mish(),
            )
            self.film = nn.Linear(c_dim, dim * 2)
            self.skip = nn.Identity()

        def forward(self, x, c):
            s, b = self.film(c).chunk(2, dim=-1)
            h = self.net(x)
            return h * (1 + s[..., None]) + b[..., None] + self.skip(x)

    class _UNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [hidden, hidden * 2, hidden * 4]

            self.inp = nn.Conv1d(action_dim, hidden, 1)

            self.down = nn.ModuleList()
            in_d = hidden
            for out_d in dims:
                self.down.append(nn.ModuleList([
                    _FiLMBlock(in_d, cond_dim),
                    _FiLMBlock(in_d, cond_dim),
                    nn.Conv1d(in_d, out_d, 3, stride=2, padding=1),
                ]))
                in_d = out_d

            self.mid = nn.ModuleList([
                _FiLMBlock(in_d, cond_dim),
                _FiLMBlock(in_d, cond_dim),
            ])

            self.up = nn.ModuleList()
            for out_d in reversed(dims):
                self.up.append(nn.ModuleList([
                    nn.ConvTranspose1d(in_d, out_d, 4, stride=2, padding=1),
                    _FiLMBlock(out_d * 2, cond_dim),
                    _FiLMBlock(out_d * 2, cond_dim),
                    nn.Conv1d(out_d * 2, out_d, 1),
                ]))
                in_d = out_d

            self.out = nn.Sequential(
                nn.GroupNorm(min(8, in_d), in_d),
                nn.Mish(),
                nn.Conv1d(in_d, action_dim, 1),
            )

        def forward(self, x, c):
            # x: (B, T, action_dim) → (B, action_dim, T)
            x = x.permute(0, 2, 1)
            x = self.inp(x)
            skips = []
            for b1, b2, down in self.down:
                x = b1(x, c)
                x = b2(x, c)
                skips.append(x)
                x = down(x)
            for blk in self.mid:
                x = blk(x, c)
            for up, b1, b2, proj in self.up:
                x = up(x)
                sk = skips.pop()
                if x.shape[-1] != sk.shape[-1]:
                    x = x[..., : sk.shape[-1]]
                x = torch.cat([x, sk], dim=1)
                x = b1(x, c)
                x = b2(x, c)
                x = proj(x)
            return self.out(x).permute(0, 2, 1)  # (B, T, action_dim)

    import torch  # noqa: PLC0415
    return _UNet()


def _sinusoidal_embedding(t, dim: int):
    """Sinusoidal timestep embedding: (B,) → (B, dim)."""
    import torch

    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
    )
    emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


# ── DDPM / DDIM noise schedule ────────────────────────────────────────────────


class _DDPMSchedule:
    """Linear DDPM noise schedule with DDIM sampling."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        import torch

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_ab = alpha_bars.sqrt()
        self.sqrt_one_minus_ab = (1 - alpha_bars).sqrt()

    def q_sample(self, x0, t, noise=None):
        """Forward noising: sample x_t from x_0 and noise at timestep t."""
        import torch

        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_ab[t].to(x0.device)
        r = self.sqrt_one_minus_ab[t].to(x0.device)
        return s[:, None, None] * x0 + r[:, None, None] * noise, noise

    def ddim_sample(self, model, shape, cond, steps: int = 10, device="cpu"):
        """Reverse DDIM sampling: denoise from pure noise."""
        import torch

        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps + 1, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            for i in range(steps):
                t_cur = ts[i].expand(shape[0]).to(device)
                t_nxt = ts[i + 1].expand(shape[0]).to(device)
                t_emb = _sinusoidal_embedding(t_cur, 128).to(device)
                cond_vec = torch.cat([cond, t_emb], dim=-1)
                eps = model(x, cond_vec)

                ab_cur = self.alpha_bars[t_cur[0]].to(device)
                ab_nxt = self.alpha_bars[t_nxt[0]].to(device) if t_nxt[0] >= 0 else torch.tensor(1.0)
                x0_pred = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
                x0_pred = x0_pred.clamp(-1, 1)
                x = ab_nxt.sqrt() * x0_pred + (1 - ab_nxt).sqrt() * eps
        return x


# ── DiffusionPolicy ───────────────────────────────────────────────────────────


class DiffusionPolicy:
    """
    Diffusion Policy (Chi et al. 2023) for discrete-action Expedition 33.

    Args:
        obs_dim:       Observation dimensionality (9 for UE4SS Phase 1).
        num_actions:   Number of discrete actions (default: 7).
        obs_horizon:   Past observations fed as context (default: 2).
        pred_horizon:  Future actions predicted per call (default: 16).
        action_horizon: Actions executed before replanning (default: 8).
        hidden:        UNet hidden channel width (default: 256).
        lr:            Adam learning rate (default: 1e-4).
        device:        Torch device string (default: 'auto').
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int = 7,
        obs_horizon: int = OBS_HORIZON,
        pred_horizon: int = PRED_HORIZON,
        action_horizon: int = ACTION_HORIZON,
        hidden: int = 256,
        lr: float = 1e-4,
        device: str = "auto",
    ):
        import torch

        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # cond_dim = flattened obs context + sinusoidal timestep embedding (dim=128)
        cond_dim = obs_horizon * obs_dim + 128
        self._net = _build_unet(action_dim=1, cond_dim=cond_dim, hidden=hidden).to(device)
        self._sched = _DDPMSchedule()
        self._opt = torch.optim.AdamW(self._net.parameters(), lr=lr)

        # Rolling observation context for online prediction
        self._obs_deque: deque = deque(maxlen=obs_horizon)
        self._pending_actions: list[int] = []

    # ── Training ──────────────────────────────────────────────────────────────

    def train_from_demos(
        self,
        demos_dir: str,
        epochs: int = 200,
        batch_size: int = 64,
        checkpoint_dir: str = "data/models",
    ) -> str:
        """
        Train from .npz demo files and return the checkpoint path.

        Args:
            demos_dir:      Directory containing .npz demo files.
            epochs:         Number of full passes over the dataset.
            batch_size:     Training batch size.
            checkpoint_dir: Directory where the .pt checkpoint is saved.

        Returns:
            Path to the saved checkpoint file.
        """
        import torch
        from torch.utils.data import DataLoader

        from il.dataset import DemoDataset

        dataset = DemoDataset(
            demos_dir,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
            num_actions=self.num_actions,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        self._net.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for obs_seq, act_seq in loader:
                obs_seq = obs_seq.to(self.device)  # (B, obs_horizon, obs_dim)
                act_seq = act_seq.to(self.device)  # (B, pred_horizon, 1)

                B = obs_seq.shape[0]
                t = torch.randint(0, self._sched.T, (B,), device=self.device)
                noisy, noise = self._sched.q_sample(act_seq, t)

                t_emb = _sinusoidal_embedding(t, 128)
                obs_flat = obs_seq.flatten(1)  # (B, obs_horizon * obs_dim)
                cond = torch.cat([obs_flat, t_emb], dim=-1)

                eps_pred = self._net(noisy, cond)
                loss = torch.nn.functional.mse_loss(eps_pred, noise)

                self._opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                self._opt.step()
                total_loss += loss.item()

            avg = total_loss / max(len(loader), 1)
            if epoch % 10 == 0 or epoch == 1:
                print(f"[DiffusionPolicy] epoch {epoch}/{epochs}  loss={avg:.4f}")

        return self.save(checkpoint_dir)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> int:
        """
        Return the next action given the latest observation.

        Maintains an internal rolling obs window and a queue of pending actions.
        Replans (runs DDIM) when the queue is empty.

        Args:
            obs: Latest observation vector, shape (obs_dim,).

        Returns:
            Discrete action index in [0, num_actions-1].
        """
        import torch

        self._obs_deque.append(obs)

        if not self._pending_actions:
            # Re-plan: run DDIM to get next action_horizon actions
            while len(self._obs_deque) < self.obs_horizon:
                self._obs_deque.appendleft(np.zeros(self.obs_dim, dtype=np.float32))

            obs_np = np.stack(list(self._obs_deque), axis=0)  # (obs_horizon, obs_dim)
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)  # (1, oh, od)
            obs_flat = obs_t.flatten(1)  # (1, obs_horizon * obs_dim)

            actions_norm = self._sched.ddim_sample(
                self._net,
                shape=(1, self.pred_horizon, 1),
                cond=obs_flat,
                steps=DDIM_STEPS,
                device=self.device,
            )  # (1, pred_horizon, 1)

            actions_norm = actions_norm.squeeze().cpu().numpy()  # (pred_horizon,)
            # Denormalise: [-1, 1] → [0, num_actions-1]
            actions_float = (actions_norm + 1) / 2 * (self.num_actions - 1)
            self._pending_actions = list(
                np.clip(np.round(actions_float), 0, self.num_actions - 1).astype(int)[
                    : self.action_horizon
                ]
            )

        return int(self._pending_actions.pop(0))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str = "data/models") -> str:
        """Save model weights to a .pt checkpoint.  Returns the file path."""
        import torch

        os.makedirs(checkpoint_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(checkpoint_dir, f"dp_{ts}.pt")
        torch.save(
            {
                "net": self._net.state_dict(),
                "obs_dim": self.obs_dim,
                "num_actions": self.num_actions,
                "obs_horizon": self.obs_horizon,
                "pred_horizon": self.pred_horizon,
                "action_horizon": self.action_horizon,
            },
            path,
        )
        print(f"[DiffusionPolicy] Saved → {path}")
        return path

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "DiffusionPolicy":
        """Load a checkpoint saved by DiffusionPolicy.save()."""
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        dp = cls(
            obs_dim=ckpt["obs_dim"],
            num_actions=ckpt["num_actions"],
            obs_horizon=ckpt["obs_horizon"],
            pred_horizon=ckpt["pred_horizon"],
            action_horizon=ckpt["action_horizon"],
            device=device,
        )
        dp._net.load_state_dict(ckpt["net"])
        dp._net.eval()
        return dp
