# CLI Usage Guide

Full reference for all commands in the Expedition 33 RL Agent project.

> **Administrator privileges required** for all `main.py` commands — `win32api` hotkeys and the topmost overlay need elevated permissions. Run your terminal or IDE as Administrator.
>
> Launch **_Clair Obscur: Expedition 33_** in **Windowed** or **Borderless Window** mode before running any game-facing command.

---

## Table of Contents

- [Calibration Recorder](#calibration-recorder)
- [ROI Analysis](#roi-analysis)
- [YOLO Training Pipeline](#yolo-training-pipeline)
- [Human Demo Recorder](#human-demo-recorder)
- [Vision Benchmark](#vision-benchmark)
- [Running Tests](#running-tests)

---

## Calibration Recorder

Run the vision system live over the game to collect detection data and ROI logs.

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

## ROI Analysis

Calculate the optimal monitor capture region from recorded logs to maximise FPS.

```bash
uv run main.py analyze
```

Reads all CSV files in `data/logs/` and prints the tightest bounding box that covers all detections — use the output as the `roi` argument when constructing `GameInstance` or `Expedition33Env`.

---

## YOLO Training Pipeline

Three commands, run in order. No game required for steps 2 and 3.

### Step 1 — Collect screenshots

```bash
uv run main.py collect
```

The collector runs three independent save modes simultaneously:

| Key | Mode | Output |
|---|---|---|
| **F8** | Toggle trigger mode — saves PNG + YOLO label when a detection fires | `images/labeled/` + `labels/labeled/` |
| **F9** | Manual — save one PNG immediately | `images/raw/` |
| **F10** | Toggle auto-capture at 4 FPS | `images/raw/` |
| **Tab** | Toggle ROI boundary overlay | — |
| **F11** | Exit | — |

**Trigger mode** uses a per-target cooldown of 0.4 s to avoid saving duplicate frames of the same static UI element. `GRADIENT_INCOMING` is never written to label files (no bounding box). The overlay shows `●TRIG` / `●AUTO` in red when either active mode is on.

Pre-labeled frames go directly to `images/labeled/` and `labels/labeled/` and can be used for training without a separate autolabel step. Unlabeled raw frames in `images/raw/` must go through the autolabel step first.

### Step 2 — Auto-label with PIXEL

```bash
uv run main.py autolabel                        # default: 80/20 train/val split
uv run main.py autolabel --val-split 0.15       # custom split
uv run main.py autolabel --no-negatives         # exclude frames with no detections
```

Runs the PIXEL engine on every raw screenshot, writes YOLO `.txt` label files, splits into `train/` and `val/`, and generates `dataset.yaml`. No manual labeling required.

### Step 3 — Train

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

## Human Demo Recorder

Record human gameplay as observation–action trajectories for imitation learning.

```bash
uv run python -m tools.demo_recorder                          # defaults
uv run python -m tools.demo_recorder --session combat_01      # named session
uv run python -m tools.demo_recorder --engine SIFT --hz 30    # SIFT at 30 Hz
```

| Option | Default | Description |
|---|---|---|
| `--session` | `demo` | Output filename stem — saved as `data/demos/<session>.npz` |
| `--engine` | `PIXEL` | Vision engine for observation capture |
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

Press **Ctrl+C** to stop. The trajectory is saved automatically as a compressed `.npz`:
- `observations` — `float32 (N, 30)`: vision state at each timestep
- `actions` — `int32 (N,)`: action index per timestep
- `timestamps` — `float64 (N,)`: Unix timestamp of each capture

> **Choosing `--hz`:** Run the live stress test first (`--live` flag below) to find the max sustainable FPS for your engine and hardware, then set `--hz` to 60–70% of that value for safe headroom.

---

## Vision Benchmark

Profile vision engines on saved screenshots or run a live capture stress test.

### Offline benchmark

```bash
uv run python -m tools.vision_benchmark                             # PIXEL vs SIFT vs ORB
uv run python -m tools.vision_benchmark --engines PIXEL ORB        # specific engines
uv run python -m tools.vision_benchmark --csv results/bench.csv    # save to CSV
```

| Option | Default | Description |
|---|---|---|
| `--engines` | `PIXEL SIFT ORB` | Space-separated engine names |
| `--img-dir` | `data/screenshots` | Directory of `.png`/`.jpg` screenshots |
| `--limit` | all | Max images per engine |
| `--warmup` | `5` | Warmup frames excluded from timing |
| `--csv` | — | Optional path to save summary as CSV |

Outputs three tables: throughput/latency (FPS, mean/median/p95/max ms), detection rate per label, and mean confidence per label when detected.

### Live capture stress test

Measures full-pipeline FPS (screen capture + colour conversion + inference) and recommends a safe poll Hz.

```bash
uv run python -m tools.vision_benchmark --live --live-engine PIXEL
uv run python -m tools.vision_benchmark --live --live-engine ORB --live-duration 30
```

| Option | Default | Description |
|---|---|---|
| `--live-engine` | `PIXEL` | Engine to stress test |
| `--live-duration` | `10` | Test duration in seconds |

Prints sustained FPS, latency stats, and a tier recommendation (20 / 30 / 60 Hz) at 1.2× headroom. Example output:

```
  60 Hz  [YES]  ####################  (need 72 FPS, got 84.7)
  30 Hz  [YES]  ####################  (need 36 FPS, got 84.7)
  20 Hz  [YES]  ####################  (need 24 FPS, got 84.7)
```

---

## Running Tests

```bash
uv sync --group dev          # install pytest + pytest-mock (first time only)
uv run pytest tests/         # run the full suite
uv run pytest tests/ -v      # verbose output
uv run pytest tests/ -x      # stop on first failure
```

See [TESTING.md](../TESTING.md) for a full breakdown of coverage per module and mocking conventions.
