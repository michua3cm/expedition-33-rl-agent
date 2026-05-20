# Next Steps & Roadmap

## What Was Built (Session 2)

Three parallel development tracks were implemented and pushed as separate branches.

### Track A — `feat/ue4ss-env`
- `environment/ue4ss_reader.py` — reads game state from UE4SS Lua mod JSON output
- `environment/ue4ss_env.py` — 9-dim Gymnasium env (no screen capture needed)
- `mods/StateReader/Scripts/main.lua` — UE4SS Lua mod template (CONFIG table needs filling)
- `calibration/ue4ss_config.py` — stub for Blueprint class/property names (fill after Live Viewer)
- `tools/demo_recorder.py` — extended with `UE4SSDemoRecorder` and `--env {vision,ue4ss}` flag
- `main.py` — added `demo` subcommand
- `tests/test_ue4ss_env.py` — 23 tests, all passing

### Track B — `feat/diffusion-policy`
- `il/diffusion_policy.py` — Diffusion Policy trainer (DDPM noise, DDIM 10-step inference, FiLM-conditioned 1D Conv UNet)
- `il/dataset.py` — `DemoDataset` with sliding obs window + action chunking; `load_transitions()` kept for GAIL backward compat
- `main.py` — added `dp-train` subcommand
- `tests/test_diffusion_policy.py` — 10 tests (5 pass, 5 skip without torch)

### Track C — `reinforcement-learning`
- `rl/train_sac.py` — SAC trainer (SB3), optional Diffusion Policy warm-start, robotics-compatible
- `rl/policy.py` — added `load_sac_checkpoint()`
- `main.py` — added `sac-train` subcommand
- `tests/test_sac.py` — 10 tests, all passing

### Known Constraint
`sac-train` imports `UE4SSExpedition33Env` from `environment/ue4ss_env.py`, which only exists on
`feat/ue4ss-env`. SAC cannot run until `feat/ue4ss-env` is merged into `dev` first.

---

## Branch Review & Merge Order

Each branch can be pulled and tested independently (unit tests only). For live execution,
follow this order — each phase unblocks the next.

### Phase 1 — Validate & Merge `feat/ue4ss-env` ← Start here

**User actions required (game must be running):**
- [ ] Install UE4SS (Nexus Mods mod 630 for Expedition 33) into
      `...\Sandfall\Binaries\Win64\` alongside `SandFall-Win64-Shipping.exe`
- [ ] Start a battle → open UE4SS console → run Live Property Viewer
- [ ] Note the exact Blueprint class name for the player character
      and property names for: CurrentHP, MaxHP, CurrentAP, BreakMeter, IsPlayerTurn
- [ ] Fill those names into `calibration/ue4ss_config.py`
- [ ] Fill the `CONFIG` table in `mods/StateReader/Scripts/main.lua`
- [ ] Reload UE4SS mods in-game; verify `%TEMP%\expedition33_state.json` is written each frame
- [ ] Run `uv run main.py demo --env ue4ss --session test_01` and confirm `.npz` is saved

**Code review:**
- [ ] Run `uv run pytest tests/test_ue4ss_env.py`
- [ ] Review `environment/ue4ss_env.py` reward shaping — adjust weights if needed
- [ ] Merge `feat/ue4ss-env` → `dev`

### Phase 2 — Validate & Merge `feat/diffusion-policy`

Requires `.npz` demo files from Phase 1 (or any existing demos).

- [ ] Run `uv run pytest tests/test_diffusion_policy.py`
- [ ] Smoke test: `uv run main.py dp-train --demos-dir data/demos --obs-dim 9 --epochs 5`
- [ ] Verify loss decreases and a checkpoint is saved under `data/models/`
- [ ] Merge `feat/diffusion-policy` → `dev`

### Phase 3 — Validate & Merge `reinforcement-learning`

PPO can be tested independently. SAC requires Phase 1 merged first.

- [ ] Run `uv run pytest tests/test_rl.py tests/test_sac.py`
- [ ] Test PPO: `uv run main.py rl-train --timesteps 1000`
- [ ] After Phase 1 merge: `uv run main.py sac-train --timesteps 1000`
- [ ] Merge `reinforcement-learning` → `dev`

### Phase 4 — Full Pipeline Run (all three merged into `dev`)

- [ ] Record real demos with UE4SS (`demo --env ue4ss`, offensive phase only for now)
- [ ] Train Diffusion Policy on real demos → save checkpoint
- [ ] Fine-tune with PPO → compare with SAC on sample efficiency
- [ ] Delete branch `claude/init-project-setup-OpPAU` (marked for deletion, not yet done)

---

## Future Phases

### Phase 5 — Vision Timing Windows (14-dim obs)
Upgrade from 9-dim to 14-dim observation space by adding the five vision-detected timing dims:
`parry_window`, `dodge_window`, `jump_window`, `gradient_attack`, `counter_attack_window`.

These are not memory-readable — they require a visual frame classifier.

- Complete `feat/dino-engine` (DINOv2, 768-dim) — highest robotics transfer value
- Merge `feat/clip-obs` (CLIP, 512-dim) — already built, language-conditioned
- Add timing window classifiers on top of DINOv2/CLIP features
- Update `UE4SSExpedition33Env` to accept a vision engine and build 14-dim obs

### Phase 6 — Action Masking
Add phase-aware action masking so the agent cannot select offensive actions during the
defensive phase and vice versa.

- Replace `PPO` with `sb3-contrib.MaskablePPO`
- Add `action_masks()` method to `UE4SSExpedition33Env` using `is_offensive_phase`
- Offensive phase (1): mask actions 5–10; defensive phase (0): mask actions 1–4
- NOOP (0) always unmasked

### Phase 7 — Transformer IL Upgrade
Swap the 1D Conv UNet in `il/diffusion_policy.py` for a Transformer encoder
(ACT-style action chunking) for better long-horizon sequence modelling.

- Implement `TransformerDiffusionPolicy` in `il/diffusion_policy.py`
- Cross-attention between obs tokens and noisy action tokens
- Compatible with the same `DemoDataset` and `.npz` format — no data changes needed

### Phase 8 — Robotics Migration
Transfer the trained policies to a real robot (mobile manipulation or humanoid).

Key design decisions already in place that make this a swap not a rewrite:
- Obs dim flows through `env.observation_space.shape` — no hardcoded dims
- SAC trainer works unchanged for continuous joint control (remove argmax decode)
- `il/diffusion_policy.py` encoder is swappable (MLP → DINOv2 ViT)
- Demo format matches robomimic/LeRobot HDF5 layout

Migration checklist:
- [ ] Replace `environment/gym_env.py` with a robot env adapter (IsaacGym / MuJoCo / LeRobot)
- [ ] Replace `tools/demo_recorder.py` with `robot_recorder.py` (ROS topics or LeRobot teleoperation)
- [ ] Swap MLP observation encoder for `DINOv2ViT` in `il/diffusion_policy.py`
- [ ] Remove argmax decode from SAC action output for continuous joints
- [ ] `rl/train_sac.py` — unchanged
