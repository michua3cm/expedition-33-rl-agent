"""
UE4SS-backed Gymnasium environment for Clair Obscur: Expedition 33.

Reads game state from the UE4SS Lua mod (mods/StateReader) via a JSON temp
file instead of running a vision pipeline.  Phase 1 supports 9-dim obs built
entirely from runtime values — no screenshots needed.

Observation space — Box(9,) float32, values roughly in [-1, 1]:
  [0] player_hp_ratio       player_hp / player_hp_max, clamped [0, 1]
  [1] enemy_hp_ratio        enemy_hp  / enemy_hp_max,  clamped [0, 1]
  [2] player_ap_norm        player_ap / 9.0,           clamped [0, 1]
  [3] enemy_break_ratio     enemy_break / enemy_break_max, clamped [0, 1]
  [4] in_battle             1.0 if a battle is active, else 0.0
  [5] is_offensive_phase    1.0 if it is the player's action turn
  [6] player_hp_delta       Δ player_hp_ratio since the previous step
  [7] enemy_hp_delta        Δ enemy_hp_ratio  since the previous step
  [8] player_ap_delta       Δ player_ap_norm  since the previous step

Action space — Discrete(7): see environment/actions.py
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import actions as A
from .instance import GameInstance
from .ue4ss_reader import GameState, StateReader

OBS_DIM = 9


class UE4SSExpedition33Env(gym.Env):
    """
    Phase-1 Gymnasium environment backed by UE4SS game state.

    Args:
        reader:      StateReader instance.  Created with defaults when None.
        game:        GameInstance for action dispatch.  Pass None for headless
                     unit tests — actions will be silently skipped.
        step_delay:  Seconds to wait after executing an action before reading
                     the next state (allows animations to settle).
        max_steps:   Truncation limit per episode.  0 = unlimited.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        reader: StateReader | None = None,
        game: GameInstance | None = None,
        step_delay: float = 0.15,
        max_steps: int = 0,
    ):
        super().__init__()
        self._reader = reader or StateReader()
        self._game = game
        self.step_delay = step_delay
        self.max_steps = max_steps

        # Dims 0–5 are "current"; 6–8 are deltas computed on each step.
        # low/high are approximate — deltas can briefly exceed ±1 in edge cases.
        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -1.0, dtype=np.float32),
            high=np.full(OBS_DIM,  1.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(A.NUM_ACTIONS)

        self._prev: np.ndarray = np.zeros(3, dtype=np.float32)  # [hp, ehp, ap]
        self._step_count: int = 0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0
        state = self._reader.read()
        obs = self._build_obs(state, reset=True)
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._execute_action(action)
        if self._game is not None:
            time.sleep(self.step_delay)

        state = self._reader.read()
        obs = self._build_obs(state)
        reward = self._compute_reward(obs, state)

        self._step_count += 1
        terminated = bool(not state["in_battle"] and self._step_count > 1)
        truncated = self.max_steps > 0 and self._step_count >= self.max_steps

        info: dict = {
            "in_battle": state["in_battle"],
            "is_offensive_phase": state["is_offensive_phase"],
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        pass  # live game window is the render

    def close(self) -> None:
        if self._game is not None:
            self._game.sct.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_action(self, action: int) -> None:
        if self._game is None or action == A.NOOP:
            return
        if action == A.PARRY:
            self._game.parry()
        elif action == A.DODGE:
            self._game.dodge()
        elif action == A.JUMP:
            self._game.jump()
        elif action == A.GRADIENT_PARRY:
            self._game.gradient_parry()
        elif action == A.ATTACK:
            self._game.attack()
        elif action == A.JUMP_ATTACK:
            self._game.jump_attack()
        else:
            raise ValueError(f"Unknown action index: {action}")

    def _build_obs(self, state: GameState, reset: bool = False) -> np.ndarray:
        p_hp  = _safe_ratio(state["player_hp"],  state["player_hp_max"])
        e_hp  = _safe_ratio(state["enemy_hp"],   state["enemy_hp_max"])
        p_ap  = float(state["player_ap"]) / 9.0
        e_brk = _safe_ratio(state["enemy_break"], state["enemy_break_max"])
        in_b  = 1.0 if state["in_battle"] else 0.0
        is_off = 1.0 if state["is_offensive_phase"] else 0.0

        current = np.array([p_hp, e_hp, p_ap, e_brk, in_b, is_off], dtype=np.float32)
        curr3 = current[:3]  # hp, ehp, ap — the three delta dims

        deltas = np.zeros(3, dtype=np.float32) if reset else (curr3 - self._prev)
        self._prev = curr3.copy()

        return np.concatenate([current, deltas])

    def _compute_reward(self, obs: np.ndarray, state: GameState) -> float:
        reward = -0.1  # step penalty keeps the agent from stalling

        enemy_hp_delta = float(obs[7])
        player_hp_delta = float(obs[6])

        if enemy_hp_delta < 0:   # dealt damage
            reward += abs(enemy_hp_delta) * 20.0
        if player_hp_delta < 0:  # received damage
            reward -= abs(player_hp_delta) * 10.0
        if not state["in_battle"] and self._step_count > 1:  # won the battle
            reward += 100.0

        return reward


# ── Module-level helper ───────────────────────────────────────────────────────


def _safe_ratio(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(np.clip(value / maximum, 0.0, 1.0))
