# Testing Guide

Unit tests for the Expedition 33 RL Agent project. All tests use `pytest` and follow the **Arrange–Act–Assert** pattern. No test makes live network calls, touches the real monitor, or calls real vision engine inference — all external dependencies are fully mocked.

## Setup

```bash
uv sync --group dev
```

This installs `pytest` and `pytest-mock` into the virtual environment.

## Running Tests

```bash
# Run the full test suite
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_gym_env.py

# Run a single test class
uv run pytest tests/test_gym_env.py::TestComputeReward

# Run a single test
uv run pytest tests/test_gym_env.py::TestComputeReward::test_first_detection_of_signal_gives_reward_plus_penalty

# Show verbose output
uv run pytest tests/ -v

# Stop on first failure
uv run pytest tests/ -x
```

## Test Coverage by Module

| Test file | Module under test | Key areas covered |
|---|---|---|
| `test_actions.py` | `environment/actions.py` | Constants, ACTION_NAMES, ACTION_INDEX, round-trip |
| `test_vision_engine.py` | `vision/engine.py` | Detection fields/equality, GameState, apply_roi(), _iou(), nms() |
| `test_vision_registry.py` | `vision/registry.py` | register(), create(), available() |
| `test_pixel_engine.py` | `vision/engines/pixel.py` | load() skip/reject logic, needs_color, hsv_sat detection |
| `test_feature_engines.py` | `vision/engines/sift.py`, `orb.py` | load() file guards, detect() empty/no-keypoint paths |
| `test_composite_engine.py` | `vision/engines/composite.py` | Target routing, needs_color propagation, result merging, frame forwarding |
| `test_calibration_logger.py` | `calibration/logger.py` | Recording state, add_point(), save_to_csv() |
| `test_roi_overlay.py` | `calibration/roi_overlay.py` | roi_to_pixels() conversions, draw_roi_overlays() dispatch |
| `test_collector.py` | `calibration/collector.py` | YOLO label format, GRADIENT skip, trigger cooldown, mode flags, save counters |
| `test_log_analyzer.py` | `calibration/analysis/core.py` | calculate_roi() padding/clamping, load_and_merge_logs() CSV handling |
| `test_auto_label.py` | `tools/auto_label.py` | _detection_to_yolo() normalisation, _write_dataset_yaml() content |
| `test_gym_env.py` | `environment/gym_env.py` | Spaces, reset, step, observation builder, reward, action dispatch |
| `test_state_buffer.py` | `environment/state_buffer.py` | Lifecycle, state access, timeout, error resilience |
| `test_demo_recorder.py` | `tools/demo_recorder.py` | Key/mouse mapping, capture loop, observation builder, save |
| `test_vision_benchmark.py` | `tools/vision_benchmark.py` | EngineResult, LiveStressResult, image loading, benchmark runner, live stress test, CSV export, print output |
| `test_gail.py` | `il/dataset.py`, `il/gail.py` | Demo loading, obs shapes, next_obs shift, Transitions construction, GAIL orchestration, DummyVecEnv wrapping, checkpoint save |
| `test_rl.py` | `rl/policy.py`, `rl/train.py` | GAIL checkpoint warm-start, train() orchestration, warm-start flag, env.close(), checkpoint path |

> **Note:** `test_gail.py` and `test_rl.py` mock all heavy deps via `sys.modules` injection — they run in CI with only the `dev` group installed.

---

## test_actions.py

**Module:** `environment/actions.py`

| Test class | What is verified |
|---|---|
| `TestActionConstants` | All 7 constants are unique integers, zero-indexed, `NUM_ACTIONS == 7` |
| `TestActionNames` | `ACTION_NAMES` covers all constants, values are strings, specific mappings correct |
| `TestActionIndex` | `ACTION_INDEX` is the exact inverse of `ACTION_NAMES`, full round-trip |

---

## test_vision_engine.py

**Module:** `vision/engine.py`

| Test class | What is verified |
|---|---|
| `TestDetection` | Field storage, value equality, label inequality, confidence at 0.0 and 1.0 |
| `TestGameState` | All fields stored, `frame` defaults to `None`, empty and multi-detection lists |
| `TestApplyRoi` | None passthrough, crop dimensions, offsets, 1×1 clamp, greyscale, pixel content |
| `TestIou` | No overlap → 0.0, identical → 1.0, partial overlap, touching edges, contained box, symmetry |
| `TestNms` | Empty list, single detection, non-overlapping both kept, overlapping lower-conf suppressed, different labels not suppressed, input-order independence, custom IoU threshold, multi-label independence |

---

## test_pixel_engine.py

**Module:** `vision/engines/pixel.py`

Template file loading uses `patch("os.path.exists")` and `patch("cv2.imread")`.  HSV saturation detection uses real numpy arrays — no mocking needed.

| Test class | What is verified |
|---|---|
| `TestPixelEngineLoad` | Skips `file=None` with no `hsv_sat_max`, loads `hsv_sat` target, skips missing file, loads grey template, loads multiple targets |
| `TestPixelEngineNeedsColor` | `False` when empty or grey-only, `True` when an `hsv_sat` target is loaded |
| `TestPixelEngineDetectHsvSat` | Fires on grey BGR frame (sat ≈ 0), silent on colourful frame (sat = 255), confidence = 1.0 at zero saturation, detection covers full frame, empty templates → `[]` |

---

## test_feature_engines.py

**Modules:** `vision/engines/sift.py`, `vision/engines/orb.py`

Both engines share identical load-guard logic.  Real cv2 feature extractors are exercised on blank arrays (no template files required).

| Test class | What is verified |
|---|---|
| `TestSIFTEngineLoad` | Skips `file=None` target, skips missing file, clears templates on reload, multiple `None` targets → zero loaded |
| `TestSIFTEngineDetect` | Empty templates → `[]`, blank frame (no keypoints) → `[]` |
| `TestORBEngineLoad` | Same four guards as SIFT |
| `TestORBEngineDetect` | Empty templates → `[]`, blank frame (no keypoints) → `[]` |

---

## test_composite_engine.py

**Module:** `vision/engines/composite.py`

Sub-engines are replaced with `MagicMock` objects — no template files or cv2 inference.

| Test class | What is verified |
|---|---|
| `TestCompositeEngineLoad` | Defaults to PIXEL, routes by `engine` key, groups targets correctly per sub-engine, clears on reload |
| `TestCompositeEngineNeedsColor` | `False` when no sub-engines, `False` when all grey, `True` when any sub-engine needs colour |
| `TestCompositeEngineDetect` | Returns `[]` with no sub-engines, merges detections from all sub-engines, passes BGR frame to colour engine, passes grey frame through unchanged |

---

## test_vision_registry.py

**Module:** `vision/registry.py`

Each test isolates the global `_REGISTRY` via `patch.dict` to avoid cross-test contamination.

| Test class | What is verified |
|---|---|
| `TestRegister` | Stores under uppercase key, returns the decorated class, overwrites duplicates |
| `TestCreate` | Instantiates the correct class, case-insensitive lookup, `ValueError` for unknown engine with available names in message |
| `TestAvailable` | Returns all registered names, empty list when registry is empty |

---

## test_roi_overlay.py

**Module:** `calibration/roi_overlay.py`

| Test class | What is verified |
|---|---|
| `TestRoiToPixels` | Fraction-to-pixel conversion, monitor offset addition, zero w/h clamped to ≥1, full-frame ROI covers entire frame |
| `TestDrawRoiOverlays` | Targets without `roi` skipped, one `draw_roi_rect` call per target with ROI, monitor offsets applied, default color `"white"` when key absent |

---

## test_collector.py

**Module:** `calibration/collector.py`

All file I/O (`cv2.imwrite`, `open`) and platform APIs (`mss`, `OverlayWindow`, `vision.registry`) are mocked — no images are written to disk.

| Test class | What is verified |
|---|---|
| `TestWriteYoloLabel` | Correct YOLO format (`class_id x_center y_center w h`), `GRADIENT_INCOMING` omitted, unknown labels omitted, multi-detection output, empty detection list produces empty file |
| `TestSmartCollectorState` | Initial flags (`_trigger_mode=False`, `_auto_capture=False`, `_show_roi=True`), initial counts zero, trigger fires when cooldown elapsed and updates per-target timestamps, cooldown suppresses save within window, `_save_raw` increments `_raw_count`, `_save_labeled` increments `_labeled_count`, trigger block skipped when `_trigger_mode=False` |

---

## test_log_analyzer.py

**Module:** `calibration/analysis/core.py`

`_get_screen_resolution()` is patched out; `glob.glob` and `pd.read_csv` are mocked so no real CSV files are read.

| Test class | What is verified |
|---|---|
| `TestCalculateRoi` | `None` input → `None`, empty DataFrame → `None`, correct ROI without padding, padding expands all four sides, clamped to `(0, 0)` at top-left, clamped to screen bounds at bottom-right, multiple detections produce union bounding box, `mon` key equals `MONITOR_INDEX` |
| `TestLoadAndMergeLogs` | No CSV files → `None`, all empty files → `None`, single valid CSV → DataFrame, multiple CSVs merged, bad files skipped without raising |

---

## test_auto_label.py

**Module:** `tools/auto_label.py`

`builtins.open` is mocked so no YAML or label files are written to disk.

| Test class | What is verified |
|---|---|
| `TestDetectionToYolo` | Output has five space-separated fields, class ID correct for every label in `CLASS_NAMES`, x/y/w/h normalised correctly, full-frame detection gives 0.5/0.5/1.0/1.0 |
| `TestWriteDatasetYaml` | YAML content contains all class names, `nc:` field matches count, `train:` and `val:` keys present, sequential integer class IDs present |

---

## test_calibration_logger.py

**Module:** `calibration/logger.py`

File I/O uses `tmp_path` and `patch` — no CSV is written to the real `data/logs/` directory.

| Test class | What is verified |
|---|---|
| `TestRecordingState` | Initial state is not recording; `start_recording` sets flag and clears points; `stop_recording` sets flag and calls `save_to_csv` |
| `TestAddPoint` | Appends when recording, ignores when not, accumulates multiple points |
| `TestSaveToCsv` | Writes correct header and rows, no-op when empty, handles `OSError` without raising |

---

## test_gym_env.py

**Module:** `environment/gym_env.py`

`GameInstance` is replaced with a `MagicMock` — no screen capture or DirectInput calls occur.

| Test class | What is verified |
|---|---|
| `TestSpaces` | `observation_space.shape == (30,)`, dtype `float32`, bounds `[0, 1]`, `action_space.n == 7` |
| `TestReset` | Returns `(obs, info)` with correct shape and required keys; clears `_seen_signals` |
| `TestStep` | Returns 5-tuple; `terminated` and `truncated` are always `False`; obs shape correct |
| `TestBuildObs` | All-zero on no detections; correct confidence index per label; normalised bbox centre; duplicate keeps highest confidence; unknown labels ignored |
| `TestComputeReward` | Step penalty only when no signals; first detection grants reward + penalty; repeated signal suppressed; signal resets after disappearing; multiple new signals accumulate |
| `TestExecuteAction` | Parametrised over all 7 actions — each dispatches to the correct `GameInstance` method; unknown index raises `ValueError` |

---

## test_state_buffer.py

**Module:** `environment/state_buffer.py`

Real threads are used to verify concurrency behaviour. Poll rate is set to 200 Hz and timeouts are capped at 1 s so tests run fast.

| Test class | What is verified |
|---|---|
| `TestInitialState` | `latest()` returns `None` before start; `is_running` is `False` |
| `TestLifecycle` | `is_running` is `True` after start, `False` after stop; `stop()` is safe when never started |
| `TestStateAccess` | `latest()` returns `GameState` after first capture; `wait_for_state()` returns `None` on timeout; `latest()` tracks the most recent state across multiple captures |
| `TestErrorResilience` | Transient `RuntimeError` in capture does not crash the thread; subsequent captures succeed |

---

## test_demo_recorder.py

**Module:** `tools/demo_recorder.py`

`pynput` listeners are tested by calling `_on_key_press` and `_on_click` directly. The capture loop is tested by setting `_stop_event` before calling `_capture_loop()`.

| Test class | What is verified |
|---|---|
| `TestKeyMapping` | E/Q/W/F → PARRY/DODGE/GRADIENT_PARRY/ATTACK (case-insensitive, parametrised); SPACE → JUMP; unmapped keys no-op; left click pressed → JUMP_ATTACK; right click and release events → no change |
| `TestCaptureLoop` | Appends `(obs, action, timestamp)` per tick; resets `_pending_action` to NOOP after capture; capture `RuntimeError` does not crash loop |
| `TestBuildObs` | All-zero on empty detections; correct confidence index per label; duplicate keeps highest confidence |
| `TestSave` | `stop()` writes `.npz` with correct `observations`/`actions`/`timestamps` arrays; returns `None` when no frames recorded; `frame_count` reflects buffer length |

---

## test_vision_benchmark.py

**Module:** `tools/vision_benchmark.py`

All image loading, screen capture (`mss`), vision engine calls, and `time.perf_counter` are mocked.

| Test class | What is verified |
|---|---|
| `TestEngineResultRecord` | Accumulates latencies, detection counts, confidence sums and maxes |
| `TestEngineResultFinalise` | FPS and latency stats computed correctly; no-op on empty latencies; p95 captures spike |
| `TestLiveStressResultFinalise` | `recommended_hz` tiers: 60 Hz (≥72 FPS), 30 Hz (≥36), 20 Hz (≥24), 0 (below 24); no-op on empty |
| `TestLoadImages` | Loads PNG/JPG, respects `--limit`, `FileNotFoundError` on empty dir, skips corrupt and non-image files |
| `TestRunEngineBenchmark` | Correct `frame_count`, single `load()` call, warmup detect count, detections recorded, result finalised |
| `TestRunLiveStressTest` | Full `mss`+`cv2`+`time.perf_counter` mock; warmup grabs excluded from latencies; latency and FPS correct |
| `TestSaveCsv` | Header and data rows, auto-creates missing dirs, empty conf cell when label undetected, multi-engine rows |
| `TestPrintReport` / `TestPrintLiveReport` | Output contains engine name, section headers, recommendation text, below-threshold message |

---

## test_rl.py

**Modules:** `rl/policy.py`, `rl/train.py`

All heavy deps (SB3, torch) mocked via `sys.modules` injection — runs in CI with `dev` group only.

| Test class | What is verified |
|---|---|
| `TestLoadGailWeights` | `PPO.load()` called with correct checkpoint path and env; returned model used for training |
| `TestRlTrain` | Returns path ending with `.zip`; `model.save()` called once with `ppo_` in path; `model.learn()` called with the correct `total_timesteps`; `load_gail_weights()` called when `gail_checkpoint` provided; skipped when `gail_checkpoint=None`; `env.close()` called on completion; output directory created when it doesn't exist |

---


## Mocking Conventions

| External dependency | Mock strategy |
|---|---|
| `GameInstance` | `patch("module.GameInstance", return_value=MagicMock())` with `mock.monitor = {"width": 100, "height": 100}` |
| `mss.mss` | `patch("module.mss.mss", return_value=mock_sct)` where `mock_sct` is a context manager returning fake BGRA arrays |
| `cv2.imread` | `patch("module.cv2.imread", return_value=np.zeros(...))` |
| `cv2.cvtColor` | `patch("module.cv2.cvtColor", return_value=np.zeros(...))` |
| `time.perf_counter` | `patch("module.time.perf_counter", side_effect=[...])` with a pre-planned sequence |
| `vision.registry` | `patch.dict(reg._REGISTRY, {}, clear=True)` for isolation; `patch("module.vision.registry.create", return_value=mock_engine)` for callers |
| File I/O | `tmp_path` pytest fixture for real temp files; `patch("builtins.open", ...)` for error injection |
| `pynput` listeners | Call `_on_key_press` / `_on_click` directly; patch `.start()` / `.stop()` on listener objects |
