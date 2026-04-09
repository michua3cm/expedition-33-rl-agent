# Project Structure

Full annotated file tree for the Expedition 33 RL Agent project.

```text
expedition-33-rl-agent/
├── main.py                      # CLI entry point (record / analyze / collect / autolabel / train)
├── overlay_ui.py                # Win32 transparent overlay (bounding boxes + HUD)
├── pyproject.toml               # Dependencies and project metadata (uv)
│
├── assets/                      # Template images for PIXEL / SIFT / ORB engines
│
├── vision/                      # Standalone vision layer — no game or RL dependencies
│   ├── engine.py                # VisionEngine ABC, Detection and GameState dataclasses
│   ├── registry.py              # @register decorator + create() factory
│   └── engines/
│       ├── pixel.py             # Template matching (grey/BGR) + HSV saturation threshold
│       ├── sift.py              # SIFT feature matching (FLANN + Lowe's ratio test)
│       ├── orb.py               # ORB + BFMatcher (Hamming distance)
│       └── yolo.py              # YOLOv8 inference engine
│
├── calibration/                 # Data collection and calibration tools
│   ├── app.py                   # Calibration recorder (live vision over game)
│   ├── collector.py             # SmartCollector: trigger / manual / auto screenshot capture
│   ├── config.py                # TARGETS, YOLO_CLASSES, thresholds, all path constants
│   ├── logger.py                # CSV logging for calibration sessions
│   ├── roi_overlay.py           # ROI fractional-to-pixel conversion + overlay draw helper
│   └── analysis/                # ROI optimisation from recorded logs
│       ├── core.py              # Bounding-box analysis logic
│       └── entry.py             # CLI entry point for analyze command
│
├── environment/                 # RL environment — Gymnasium-compatible
│   ├── actions.py               # Shared action-index constants (7 Phase 1 actions)
│   ├── gym_env.py               # Expedition33Env: gym.Env wrapper
│   ├── state_buffer.py          # Async background capture thread (StateBuffer)
│   ├── instance.py              # GameInstance: vision + controller bridge
│   └── controls.py              # GameController: DirectInput keyboard/mouse
│
├── tools/                       # Offline pipeline tools (no game required to run)
│   ├── auto_label.py            # PIXEL → YOLO label generator + dataset.yaml writer
│   ├── train.py                 # YOLOv8 training wrapper
│   ├── demo_recorder.py         # Human gameplay demonstration recorder
│   └── vision_benchmark.py      # Vision engine profiler + live capture stress test
│
├── tests/                       # Unit tests — see TESTING.md
│   ├── test_actions.py          # environment/actions.py
│   ├── test_vision_engine.py    # vision/engine.py (Detection, GameState, apply_roi, _iou, nms)
│   ├── test_vision_registry.py  # vision/registry.py
│   ├── test_pixel_engine.py     # vision/engines/pixel.py
│   ├── test_feature_engines.py  # vision/engines/sift.py + orb.py
│   ├── test_composite_engine.py # vision/engines/composite.py
│   ├── test_calibration_logger.py  # calibration/logger.py
│   ├── test_roi_overlay.py      # calibration/roi_overlay.py
│   ├── test_collector.py        # calibration/collector.py
│   ├── test_log_analyzer.py     # calibration/analysis/core.py
│   ├── test_auto_label.py       # tools/auto_label.py
│   ├── test_gym_env.py          # environment/gym_env.py
│   ├── test_state_buffer.py     # environment/state_buffer.py
│   ├── test_demo_recorder.py    # tools/demo_recorder.py
│   └── test_vision_benchmark.py # tools/vision_benchmark.py
│
├── docs/                        # Extended documentation
│   ├── PROJECT_STRUCTURE.md     # This file
│   ├── USAGE.md                 # Full CLI reference
│   └── CONFIGURATION.md         # Config variables and TARGETS reference (planned)
│
└── data/                        # Runtime outputs — gitignored
    ├── logs/                    # CSV calibration logs (from record command)
    ├── screenshots/             # Debug snapshots + YOLO raw input
    ├── demos/                   # Human demonstration .npz files
    └── yolo_dataset/            # YOLO training dataset (auto-generated)
        ├── dataset.yaml         # Class names and split paths
        ├── images/
        │   ├── raw/             # Unlabeled screenshots (collect F9/F10 → autolabel)
        │   ├── labeled/         # Pre-labeled screenshots (collect F8 trigger mode)
        │   ├── train/           # Training split
        │   └── val/             # Validation split
        └── labels/
            ├── labeled/         # YOLO .txt files paired with images/labeled/
            ├── train/           # YOLO .txt label files
            └── val/
```

## Module Responsibilities

| Module | Responsibility | Depends on |
|---|---|---|
| `vision/` | Pure detection — takes a frame, returns detections | numpy, cv2, ultralytics |
| `calibration/` | Data collection, logging, config | vision/, mss |
| `environment/` | RL interface — wraps game as gym.Env | vision/, calibration/config |
| `tools/` | Offline utilities — YOLO pipeline, demo recording, benchmarking | vision/, environment/, calibration/ |
| `tests/` | Unit tests — all external deps mocked | pytest, pytest-mock |

## Known Limitations and Design Notes

### JUMP_CUE — Why PIXEL cannot auto-label it

`JUMP_CUE` is a golden starburst/cross icon that appears mid-combat to signal that only a jump avoids the incoming attack. Unlike static UI elements (e.g. `PERFECT`, `BATTLE_WHEEL`), this icon:

1. **Animates continuously** — it shrinks from a large size, then grows and fades out over ~0.5 s.
2. **Scales with camera distance** — different enemy attacks render the icon at different base sizes.

Template matching (`PIXEL`) requires the in-game icon to be the **exact same pixel dimensions** as the template crop. Because neither condition holds, PIXEL will miss most occurrences and cannot be used to auto-generate YOLO labels for this target.

**Solution:** `JUMP_CUE` sets `"autolabel_engine": "SIFT"` in `calibration/config.py`. SIFT is scale- and rotation-invariant — it extracts keypoints from a single template crop and matches them in a live frame regardless of the icon's current size or rotation. The autolabel pipeline (`tools/auto_label.py`) reads `autolabel_engine` per target and loads the appropriate engine, so all other targets continue to use PIXEL while JUMP_CUE uses SIFT.

**For YOLO training data:** the animation is an advantage — auto-capturing during gameplay naturally produces screenshots of the icon at many sizes within a single session, which improves the trained model's scale robustness.

---

## Key Design Decisions

- **Vision engines are fully interchangeable.** `VisionEngine` is an ABC with `load()`, `detect()`, and `needs_color`. When `needs_color` is `True` the caller passes a BGR frame instead of greyscale; engines handle their own internal conversion. All callers use `registry.create(name)` — no engine-specific imports outside `vision/engines/`.
- **`environment/` has no win32 dependency at the gym level.** `Expedition33Env` and `StateBuffer` only import `GameInstance`, which isolates the Windows-specific `GameController` behind one interface.
- **`tools/` are offline-first.** `demo_recorder.py` and `vision_benchmark.py` can be coded and tested without the game running. They only need the game live when actually recording or stress-testing.
