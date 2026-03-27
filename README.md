# Expedition 33 RL Agent

> **Project Goal:** Develop a Reinforcement Learning agent capable of playing _Clair Obscur: Expedition 33_ autonomously using computer vision, Imitation Learning, and Reinforcement Learning — then transfer the experience to a Robotics project.

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

## Project Structure

```text
expedition-33-rl-agent/
├── main.py                      # CLI entry point
├── overlay_ui.py                # Win32 transparent overlay
├── pyproject.toml               # Dependencies and project metadata
│
├── assets/                      # Template images for PIXEL/SIFT/ORB engines
│
├── vision/                      # Standalone vision layer (no external deps)
│   ├── engine.py                # VisionEngine ABC, Detection, GameState dataclasses
│   ├── registry.py              # @register decorator + create() factory
│   └── engines/
│       ├── pixel.py             # Template matching (TM_CCOEFF_NORMED)
│       ├── sift.py              # SIFT feature matching (FLANN + Lowe's ratio)
│       ├── orb.py               # ORB + BFMatcher (Hamming distance)
│       └── yolo.py              # YOLOv8 inference engine
│
├── calibration/                 # Data collection and calibration tools
│   ├── app.py                   # Calibration recorder (uses any vision engine)
│   ├── collector.py             # Screenshot collector for YOLO training data
│   ├── config.py                # Targets, thresholds, paths
│   ├── logger.py                # CSV logging
│   └── analysis/                # ROI optimization from recorded logs
│
├── tools/                       # Offline pipeline tools
│   ├── auto_label.py            # PIXEL → YOLO label generator + dataset.yaml
│   ├── train.py                 # YOLOv8 training wrapper
│   ├── demo_recorder.py         # Human gameplay demonstration recorder
│   └── vision_benchmark.py      # Vision engine performance profiler + live stress test
│
├── tests/                       # Unit tests (pytest)
│   └── test_vision_benchmark.py # Tests for vision_benchmark.py
│
├── environment/                 # RL environment (Gymnasium-compatible)
│   ├── actions.py               # Shared action-index constants (7 Phase 1 actions)
│   ├── gym_env.py               # Expedition33Env: gym.Env wrapper
│   ├── state_buffer.py          # Async background capture thread
│   ├── instance.py              # GameInstance: vision + controller bridge
│   └── controls.py              # GameController: DirectInput keyboard/mouse
│
└── data/
    ├── logs/                    # CSV calibration logs
    ├── screenshots/             # Debug snapshots
    ├── demos/                   # Human demonstration .npz files
    └── yolo_dataset/            # YOLO training dataset (auto-generated)
        ├── dataset.yaml
        ├── images/
        │   ├── raw/             # Collected screenshots (input)
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

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
uv sync
```

`uv sync` creates the `.venv` and installs all locked dependencies including `ultralytics` for YOLO and `pynput` for the demo recorder.

**Install dev dependencies (required to run tests):**

```bash
uv sync --group dev
```

## Usage

> **Administrator privileges required.** This program uses global hotkeys (`win32api`) and draws a topmost overlay. Run your terminal or IDE as Administrator.
>
> Launch **_Clair Obscur: Expedition 33_** in **Windowed** or **Borderless Window** mode before running any command.

---

### Calibration Recorder

Run the vision system live over the game to collect ROI data.

```bash
uv run main.py record               # default: PIXEL engine
uv run main.py record -e pixel      # explicit PIXEL
uv run main.py record -e sift       # SIFT engine
uv run main.py record -e orb        # ORB engine
uv run main.py record -e yolo       # YOLO engine (requires trained model)
```

| Key | Function | Description |
|---|---|---|
| **F9** | Start Recording | Begins logging detections to CSV. Saves a debug screenshot. |
| **F10** | Stop & Save | Stops recording and writes the session to `data/logs/`. |
| **F11** | Exit | Closes the overlay and exits safely. |

The overlay shows `○ IDLE` (green) when monitoring but not recording, and `● REC` (red) when actively logging.

---

### ROI Analysis

Calculate the optimal monitor region from recorded logs to maximize FPS.

```bash
uv run main.py analyze
```

---

### YOLO Training Pipeline

Three commands, run in order.

**Step 1 — Collect screenshots while playing:**

```bash
uv run main.py collect
```

| Key | Function |
|---|---|
| **F9** | Save a single screenshot |
| **F10** | Toggle auto-capture (one screenshot per second) |
| **F11** | Exit |

Screenshots are saved to `data/yolo_dataset/images/raw/`.

**Step 2 — Auto-label with PIXEL and generate the dataset:**

```bash
uv run main.py autolabel                        # default: 80/20 train/val split
uv run main.py autolabel --val-split 0.15       # custom split
uv run main.py autolabel --no-negatives         # exclude frames with no detections
```

Runs the PIXEL engine on every raw screenshot, writes YOLO `.txt` label files, splits into `train/` and `val/`, and generates `dataset.yaml`. No manual labeling required.

**Step 3 — Train:**

```bash
uv run main.py train                                # default: yolov8n, 100 epochs
uv run main.py train --epochs 150 --model yolov8s   # larger model, more epochs
```

| Option | Default | Description |
|---|---|---|
| `--epochs` | `100` | Training epochs |
| `--imgsz` | `640` | Input image size |
| `--model` | `yolov8n.pt` | Base model (`n`=fastest → `x`=most accurate) |

The trained model is saved to `data/yolo_dataset/train/weights/best.pt` and picked up automatically by `record -e yolo`.

---

### Human Demo Recorder

Record human gameplay as observation–action trajectories for imitation learning.

```bash
uv run python -m tools.demo_recorder                          # default session name, PIXEL engine, 20 Hz
uv run python -m tools.demo_recorder --session combat_01      # named session
uv run python -m tools.demo_recorder --engine SIFT --hz 30    # SIFT engine at 30 Hz
```

| Option | Default | Description |
|---|---|---|
| `--session` | `demo` | Output filename stem (saved as `data/demos/<session>.npz`) |
| `--engine` | `PIXEL` | Vision engine used for observation capture |
| `--hz` | `20.0` | Capture rate in frames per second |
| `--save-dir` | `data/demos` | Directory to write `.npz` files |

**Key → Action mapping:**

| Key | Action |
|---|---|
| `E` | PARRY |
| `Q` | DODGE |
| `W` | GRADIENT_PARRY |
| `F` | ATTACK |
| `SPACE` | JUMP |
| Left click | JUMP_ATTACK |
| _(no input)_ | NOOP |

Press **Ctrl+C** to stop recording. The trajectory is saved automatically as a compressed `.npz` with arrays:
- `observations` — `float32 (N, 30)`: vision state at each timestep
- `actions` — `int32 (N,)`: action index per timestep
- `timestamps` — `float64 (N,)`: Unix timestamp of each capture

---

### Vision Benchmark

Profile all vision engines offline or run a live capture stress test to determine the safe poll rate for your hardware.

**Offline benchmark** — runs engines against saved screenshots:

```bash
uv run python -m tools.vision_benchmark                             # PIXEL vs SIFT vs ORB
uv run python -m tools.vision_benchmark --engines PIXEL ORB        # specific engines
uv run python -m tools.vision_benchmark --csv results/bench.csv    # save summary to CSV
```

| Option | Default | Description |
|---|---|---|
| `--engines` | `PIXEL SIFT ORB` | Space-separated engine names to benchmark |
| `--img-dir` | `data/screenshots` | Directory of `.png`/`.jpg` screenshots |
| `--limit` | all | Max images per engine |
| `--warmup` | `5` | Warmup frames excluded from timing |
| `--csv` | — | Optional path to save summary as CSV |

Outputs three tables: throughput/latency (FPS, mean/median/p95/max ms), detection rate per label, and mean confidence per label.

**Live capture stress test** — measures full-pipeline FPS (screen capture + inference) and recommends a poll Hz:

```bash
uv run python -m tools.vision_benchmark --live --live-engine PIXEL
uv run python -m tools.vision_benchmark --live --live-engine ORB --live-duration 30
```

| Option | Default | Description |
|---|---|---|
| `--live-engine` | `PIXEL` | Engine to stress test |
| `--live-duration` | `10` | Test duration in seconds |

Prints sustained FPS, latency stats, and a tier recommendation (20 / 30 / 60 Hz) with 1.2× headroom. Use the result to set `--hz` in the demo recorder or `poll_hz` in `StateBuffer`.

---

### Running Tests

```bash
uv run pytest tests/
```

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
3. That's it — calibration, environment, and CLI all pick it up automatically.

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
- YOLO ignores both; it uses the model's own confidence output.

---

_Project by Michael Tsai_
