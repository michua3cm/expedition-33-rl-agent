# Expedition 33 RL Agent

> **Project Goal:** Develop a Reinforcement Learning agent capable of playing _Clair Obscur: Expedition 33_ autonomously using computer vision, Imitation Learning, and Reinforcement Learning — then transfer the experience to a Robotics project.

## Overview

The agent interacts with the game purely through visual inputs (screen capture) and keyboard/mouse simulation. No game memory access, no code injection. The vision system is designed to be swappable so different algorithms can be benchmarked fairly and replaced without touching any other part of the codebase.

**Roadmap:**

| Phase | Description | Status |
|---|---|---|
| 1 | Vision System & Calibration | Complete |
| 2 | YOLO Training Pipeline | In Progress |
| 3 | Environment Wrapper (Gym) | Planned |
| 4 | Imitation Learning (IL) | Planned |
| 5 | Reinforcement Learning (RL) | Planned |

## Key Features

- **Swappable Vision Engines:** PIXEL, SIFT, and YOLO all implement the same `VisionEngine` interface. Switch with one flag (`-e pixel`, `-e sift`, `-e yolo`). Adding a new engine requires one file.
- **YOLO Auto-Label Pipeline:** PIXEL bootstraps YOLO training labels automatically — no manual labeling required for the existing 5 game elements.
- **Async-Ready Architecture:** Vision runs behind a clean `detect(frame) → List[Detection]` contract, ready to be decoupled from the training loop via a `StateBuffer`.
- **Non-Intrusive Capture:** `mss` screen capture at >60 FPS without hooking into the game process.
- **Transparent Debug Overlay:** Win32-based HUD draws real-time bounding boxes and confidence scores over the game.
- **Data Logging Pipeline:** Records detected game events to CSV for ROI analysis and future RL environment setup.

## Project Structure

```text
expedition-33-rl-agent/
├── main.py                      # CLI entry point
├── overlay_ui.py                # Win32 transparent overlay
├── pyproject.toml               # Dependencies and project metadata
│
├── assets/                      # Template images for PIXEL/SIFT engines
│
├── vision/                      # Standalone vision layer (no external deps)
│   ├── engine.py                # VisionEngine ABC, Detection, GameState dataclasses
│   ├── registry.py              # @register decorator + create() factory
│   └── engines/
│       ├── pixel.py             # Template matching (TM_CCOEFF_NORMED)
│       ├── sift.py              # SIFT feature matching (FLANN + Lowe's ratio)
│       └── yolo.py              # YOLOv8 inference engine
│
├── calibration/                 # Data collection and calibration tools
│   ├── app.py                   # Calibration recorder (uses any vision engine)
│   ├── collector.py             # Screenshot collector for YOLO training data
│   ├── config.py                # Targets, thresholds, paths
│   ├── loader.py                # Template asset loader
│   ├── logger.py                # CSV logging
│   └── analysis/                # ROI optimization from recorded logs
│
├── tools/                       # Offline pipeline tools
│   ├── auto_label.py            # PIXEL → YOLO label generator + dataset.yaml
│   └── train.py                 # YOLOv8 training wrapper
│
├── environment/                 # RL environment bridge (in progress)
│   ├── instance.py              # GameInstance: vision + controller bridge
│   └── controls.py              # GameController: DirectInput keyboard/mouse
│
└── data/
    ├── logs/                    # CSV calibration logs
    ├── screenshots/             # Debug snapshots
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

`uv sync` creates the `.venv` and installs all locked dependencies including `ultralytics` for YOLO.

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

## Vision Engine Reference

| Engine | Algorithm | Resolution Robust | Needs Training | Best For |
|---|---|---|---|---|
| `pixel` | Template matching (TM_CCOEFF_NORMED) | No | No | Fast baseline, fixed resolution |
| `sift` | SIFT + FLANN + Lowe's ratio test | Partial | No | Scale-invariant matching |
| `yolo` | YOLOv8 fine-tuned on gameplay | Yes | Yes (auto) | Production use, resolution-robust |

All engines implement the same `VisionEngine` interface and return `List[Detection]` with confidence normalised to 0.0–1.0.

**Adding a new engine** (e.g. ORB):
1. Create `vision/engines/orb.py` implementing `VisionEngine` with `@register("ORB")`
2. Add one import line in `vision/engines/__init__.py`
3. That's it — calibration, environment, and CLI all pick it up automatically.

---

## Configuration

All targets and thresholds are defined in `calibration/config.py`.

```python
TARGETS = {
    "PERFECT": {"file": "template_perfect.png", "color": "lime",    "threshold": 0.65, "min_matches": 10},
    "DODGE":   {"file": "template_dodge.png",   "color": "yellow",  "threshold": 0.65, "min_matches": 10},
    "PARRIED": {"file": "template_parried.png", "color": "cyan",    "threshold": 0.65, "min_matches": 10},
    "JUMP":    {"file": "template_jump.png",    "color": "magenta", "threshold": 0.75, "min_matches": 15},
    "MOUSE":   {"file": "template_mouse.png",   "color": "orange",  "threshold": 0.90, "min_matches": 10},
}
```

- `threshold` — used by PIXEL engine (0.0–1.0). Higher = fewer false positives.
- `min_matches` — used by SIFT engine (minimum good feature matches to confirm detection).
- YOLO ignores both; it uses the confidence value from the trained model.

---

_Project by Michael Tsai_
