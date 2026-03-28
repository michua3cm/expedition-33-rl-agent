# Expedition 33 RL Agent

> **Project Goal:** Develop a Reinforcement Learning agent capable of playing _Clair Obscur: Expedition 33_ autonomously using computer vision, Imitation Learning, and Reinforcement Learning — then transfer the experience to a Robotics project.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Vision Engine Reference](#vision-engine-reference)
- [Configuration](#configuration)
- [Documentation](#documentation)

---

## Overview

The agent interacts with the game purely through visual inputs (screen capture) and keyboard/mouse simulation. No game memory access, no code injection. The vision system is designed to be swappable so different algorithms can be benchmarked fairly and replaced without touching any other part of the codebase.

**Roadmap:**

| Phase | Description | Status |
|---|---|---|
| 1 | Vision System & Calibration | Complete |
| 2 | YOLO Training Pipeline | In Progress |
| 3 | Environment Wrapper (Gym) | In Progress |
| 4 | Imitation Learning (IL) | Planned |
| 5 | Reinforcement Learning (RL) | Planned |

---

## Key Features

- **Swappable Vision Engines:** PIXEL, SIFT, ORB, and YOLO all implement the same `VisionEngine` interface. Switch with one flag (`-e pixel`, `-e sift`, `-e orb`, `-e yolo`). Adding a new engine requires one file.
- **YOLO Auto-Label Pipeline:** PIXEL bootstraps YOLO training labels automatically — no manual labeling required for the existing game elements.
- **Gymnasium Environment:** `Expedition33Env` wraps the game as a standard `gym.Env` with a 30-dim observation space and 7-action discrete action space.
- **Async State Buffer:** `StateBuffer` runs screen capture and vision inference on a background thread at configurable Hz so the policy loop is never stalled waiting for a frame.
- **Human Demo Recorder:** Records human gameplay as `(observation, action, timestamp)` trajectories saved as compressed `.npz` files for imitation learning.
- **Vision Benchmark Tool:** Profiles all vision engines against saved screenshots (FPS, latency percentiles, detection rates) and runs a live capture stress test to recommend a safe poll Hz for your hardware.
- **Non-Intrusive Capture:** `mss` screen capture at >60 FPS without hooking into the game process.
- **Transparent Debug Overlay:** Win32-based HUD draws real-time bounding boxes and confidence scores over the game.
- **Data Logging Pipeline:** Records detected game events to CSV for ROI analysis and RL environment setup.

---

## Architecture

```text
expedition-33-rl-agent/
├── vision/          # Swappable vision engines (PIXEL, SIFT, ORB, YOLO)
├── environment/     # Gymnasium RL environment, action space, async capture
├── calibration/     # Data collection, CSV logging, config
├── tools/           # Offline pipeline tools (YOLO training, demo recorder, benchmark)
├── tests/           # Unit tests (pytest)
├── assets/          # Template images for PIXEL/SIFT/ORB engines
└── data/            # Runtime outputs (logs, screenshots, demos, YOLO dataset)
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the full annotated file tree.

---

## Installation

This project uses `uv` for package and environment management.

**Install `uv` if you don't have it:**

- **Git Bash (Windows) / macOS / Linux:**
  ```bash
  curl -sSL https://astral.sh/uv/install.sh | bash
  ```
- **PowerShell (Windows):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

> Restart your terminal after installing to enable the `uv` command.

**Setup:**

```bash
git clone <your-repo-url>
cd expedition-33-rl-agent
uv sync                  # installs all runtime dependencies
uv sync --group dev      # also installs pytest + pytest-mock
```

---

## Quick Start

> **Administrator privileges required** — `win32api` hotkeys and the topmost overlay need elevated permissions. Run your terminal or IDE as Administrator.
>
> Launch **_Clair Obscur: Expedition 33_** in **Windowed** or **Borderless Window** mode before running any command.

| Command | What it does |
|---|---|
| `uv run main.py record` | Run vision system live, record detections to CSV |
| `uv run main.py analyze` | Calculate optimal capture ROI from recorded logs |
| `uv run main.py collect` | Collect screenshots for YOLO training |
| `uv run main.py autolabel` | Auto-label screenshots with PIXEL engine |
| `uv run main.py train` | Train YOLOv8 on the labeled dataset |
| `uv run python -m tools.demo_recorder` | Record human gameplay demonstrations |
| `uv run python -m tools.vision_benchmark` | Benchmark vision engines on saved screenshots |
| `uv run python -m tools.vision_benchmark --live` | Live capture stress test with Hz recommendation |
| `uv run pytest tests/` | Run the full unit test suite |

See [docs/USAGE.md](docs/USAGE.md) for the full CLI reference — all options, flags, hotkeys, and output formats.

---

## Vision Engine Reference

| Engine | Algorithm | Resolution Robust | Needs Training | Best For |
|---|---|---|---|---|
| `pixel` | Template matching (TM_CCOEFF_NORMED) | No | No | Fast baseline, fixed resolution |
| `sift` | SIFT + FLANN + Lowe's ratio test | Partial | No | Scale-invariant matching |
| `orb` | ORB + BFMatcher (Hamming) + Lowe's ratio | Partial | No | Faster than SIFT, robotics-ready (ORB-SLAM) |
| `yolo` | YOLOv8 fine-tuned on gameplay | Yes | Yes (auto) | Production use, resolution-robust |

All engines implement the same `VisionEngine` interface and return `List[Detection]` with confidence normalised to 0.0–1.0.

**Adding a new engine:**
1. Create `vision/engines/<name>.py` implementing `VisionEngine` with `@register("<NAME>")`
2. Add one import line in `vision/engines/__init__.py`
3. Done — calibration, environment, and CLI pick it up automatically.

---

## Configuration

All targets, thresholds, and paths are defined in `calibration/config.py`.

| Variable | Default | Description |
|---|---|---|
| `ASSETS_DIR` | `assets` | Template image directory for PIXEL/SIFT/ORB engines |
| `LOG_DIR` | `data/logs` | CSV calibration log output directory |
| `SCREENSHOT_DIR` | `data/screenshots` | Debug snapshot directory |
| `YOLO_RAW_DIR` | `data/yolo_dataset/images/raw` | Raw screenshot input for YOLO pipeline |
| `YOLO_MODEL_PATH` | `data/yolo_dataset/train/weights/best.pt` | Trained YOLO model path |
| `DEMO_DIR` | `data/demos` | Human demonstration `.npz` output directory |
| `MONITOR_INDEX` | `1` | Monitor to capture (`0` = all monitors, `1` = primary) |
| `DEFAULT_THRESHOLD` | `0.6` | Default PIXEL engine confidence threshold |
| `DEFAULT_MIN_MATCHES` | `12` | Default SIFT/ORB minimum feature matches |

**Target configuration** (`TARGETS` dict, per-entry fields):
- `threshold` — PIXEL engine confidence cutoff (0.0–1.0). Higher = fewer false positives.
- `min_matches` — SIFT/ORB minimum good feature matches to confirm a detection.
- `color_mode` — *(optional, default `False`)* If `True`, the PIXEL engine loads and matches the template in BGR instead of greyscale. Required for targets whose only distinguishing feature is colour (e.g. `TURN_ALLY` blue border vs `TURN_ENEMY` red border). Has no effect on SIFT, ORB, or YOLO.
- YOLO ignores `threshold` and `min_matches`; it uses the model's own confidence output.

---

## Documentation

| Document | Description |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | Full CLI reference — all commands, options, hotkeys, and output formats |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Full annotated file tree |
| [TESTING.md](TESTING.md) | Test guide — running tests, per-file coverage, mocking conventions |

---

_Project by Michael Tsai_
