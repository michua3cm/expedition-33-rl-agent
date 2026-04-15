"""
Vision Engine Benchmark Tool for Clair Obscur: Expedition 33.

Runs one or more vision engines over a set of saved screenshots and
produces a performance + accuracy report so you can make an informed
decision about which engine to use for live RL training.

Also includes a **live capture stress test** that measures real-world
sustained FPS including screen-capture overhead (mss + vision inference
together) and recommends a safe poll rate for your machine.

What it measures
----------------
- **Throughput** — frames per second each engine sustains on your hardware
- **Latency**    — mean, median, p95, and max inference time per frame (ms)
- **Detection rate** — how often each target label is detected across the batch
- **Confidence stats** — mean and max confidence per detected label
- **Live FPS** — real-world capture + inference FPS on your actual monitor

Typical workflow
----------------
1. Collect screenshots with the calibration recorder:
       uv run main.py collect
2. Run the offline benchmark against those screenshots:
       uv run python -m tools.vision_benchmark --engines PIXEL SIFT ORB
3. Run the live stress test for your chosen engine:
       uv run python -m tools.vision_benchmark --live --live-engine PIXEL
4. Read the recommendation and set --hz accordingly in the demo recorder
   or StateBuffer.

CLI usage
---------
    # Benchmark all three template engines on the default screenshot directory
    uv run python -m tools.vision_benchmark

    # Benchmark specific engines
    uv run python -m tools.vision_benchmark --engines PIXEL ORB

    # Use a custom screenshot directory and limit to 200 frames
    uv run python -m tools.vision_benchmark --img-dir data/screenshots --limit 200

    # Save the report to a CSV file
    uv run python -m tools.vision_benchmark --csv results/benchmark.csv

    # Run a 10-second live capture stress test with the PIXEL engine
    uv run python -m tools.vision_benchmark --live --live-engine PIXEL

    # Live stress test for 30 seconds with ORB
    uv run python -m tools.vision_benchmark --live --live-engine ORB --live-duration 30

Options
-------
    --engines         Space-separated engine names to benchmark (default: PIXEL SIFT ORB)
    --img-dir         Directory containing .png/.jpg screenshots (default: data/screenshots)
    --limit           Max images per engine (default: all)
    --warmup          Number of warmup frames before timing starts (default: 5)
    --csv             Optional path to save summary CSV
    --live            Run a live capture stress test instead of the offline benchmark
    --live-engine     Engine to use for the live stress test (default: PIXEL)
    --live-duration   Duration of the live stress test in seconds (default: 10)
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mss
import numpy as np

import vision
from calibration.config import ASSETS_DIR, MONITOR_INDEX, SCREENSHOT_DIR, TARGETS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Aggregated benchmark result for one engine over the full image batch."""
    engine_name: str
    frame_count: int
    # Timing (seconds per frame)
    latencies: list[float] = field(default_factory=list, repr=False)
    # Detection counts and confidence sums per label
    detection_counts: dict[str, int] = field(default_factory=dict)
    confidence_sums:  dict[str, float] = field(default_factory=dict)
    confidence_maxes: dict[str, float] = field(default_factory=dict)

    # --- Derived metrics (populated by finalise()) ---
    fps: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    def record(self, elapsed: float, detections: list[vision.Detection]) -> None:
        self.latencies.append(elapsed)
        for d in detections:
            self.detection_counts[d.label] = self.detection_counts.get(d.label, 0) + 1
            self.confidence_sums[d.label]  = self.confidence_sums.get(d.label, 0.0) + d.confidence
            self.confidence_maxes[d.label] = max(self.confidence_maxes.get(d.label, 0.0), d.confidence)

    def finalise(self) -> None:
        if not self.latencies:
            return
        arr = np.array(self.latencies) * 1000  # → ms
        self.fps      = 1000.0 / arr.mean() if arr.mean() > 0 else 0.0
        self.mean_ms  = float(arr.mean())
        self.median_ms = float(np.median(arr))
        self.p95_ms   = float(np.percentile(arr, 95))
        self.max_ms   = float(arr.max())


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def _load_images(img_dir: str, limit: int | None) -> list[np.ndarray]:
    """Load greyscale frames from a directory (PNG and JPG)."""
    paths = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No PNG/JPG images found in '{img_dir}'")
    if limit:
        paths = paths[:limit]

    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames.append(img)
    return frames


def run_engine_benchmark(
    engine_name: str,
    frames: list[np.ndarray],
    warmup: int,
) -> EngineResult:
    """Run a single engine over the frame batch and return aggregated results."""
    eng = vision.registry.create(engine_name)
    eng.load(TARGETS, ASSETS_DIR)

    result = EngineResult(engine_name=engine_name, frame_count=len(frames))

    # Warmup — not timed
    for frame in frames[:warmup]:
        eng.detect(frame)

    # Timed run
    for frame in frames:
        t0 = time.perf_counter()
        detections = eng.detect(frame)
        elapsed = time.perf_counter() - t0
        result.record(elapsed, detections)

    result.finalise()
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _print_report(results: list[EngineResult]) -> None:
    labels = sorted({
        label
        for r in results
        for label in r.detection_counts
    })

    # --- Throughput / latency table ---
    print("\n" + "=" * 72)
    print("  VISION ENGINE BENCHMARK REPORT")
    print("=" * 72)

    header = f"{'Engine':<10} {'Frames':>7} {'FPS':>7} {'Mean ms':>9} {'Median ms':>10} {'p95 ms':>8} {'Max ms':>8}"
    print(header)
    print("-" * 72)
    for r in results:
        print(
            f"{r.engine_name:<10} {r.frame_count:>7} {r.fps:>7.1f} "
            f"{r.mean_ms:>9.2f} {r.median_ms:>10.2f} "
            f"{r.p95_ms:>8.2f} {r.max_ms:>8.2f}"
        )

    # --- Detection rate table ---
    print("\n" + "=" * 72)
    print("  DETECTION RATES  (detections / total frames)")
    print("=" * 72)

    col_w = 14
    header_labels = "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(f"{'Engine':<10}{header_labels}")
    print("-" * (10 + col_w * len(labels)))

    for r in results:
        row = f"{r.engine_name:<10}"
        for label in labels:
            count = r.detection_counts.get(label, 0)
            rate  = count / r.frame_count if r.frame_count else 0.0
            row  += f"{f'{count} ({rate:.0%})':>{col_w}}"
        print(row)

    # --- Mean confidence table ---
    print("\n" + "=" * 72)
    print("  MEAN CONFIDENCE PER LABEL  (when detected)")
    print("=" * 72)

    print(f"{'Engine':<10}{header_labels}")
    print("-" * (10 + col_w * len(labels)))

    for r in results:
        row = f"{r.engine_name:<10}"
        for label in labels:
            count = r.detection_counts.get(label, 0)
            if count:
                mean_conf = r.confidence_sums[label] / count
                row += f"{mean_conf:>{col_w}.3f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    print("=" * 72 + "\n")


def _save_csv(results: list[EngineResult], csv_path: str) -> None:
    """Save the per-engine summary to a CSV file."""
    labels = sorted({
        label
        for r in results
        for label in r.detection_counts
    })

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        base_cols = ["engine", "frames", "fps", "mean_ms", "median_ms", "p95_ms", "max_ms"]
        det_cols  = [f"det_{lbl}" for lbl in labels]
        rate_cols = [f"rate_{lbl}" for lbl in labels]
        conf_cols = [f"conf_{lbl}" for lbl in labels]
        writer.writerow(base_cols + det_cols + rate_cols + conf_cols)

        for r in results:
            base = [
                r.engine_name, r.frame_count,
                f"{r.fps:.2f}", f"{r.mean_ms:.3f}",
                f"{r.median_ms:.3f}", f"{r.p95_ms:.3f}", f"{r.max_ms:.3f}",
            ]
            dets  = [r.detection_counts.get(lbl, 0) for lbl in labels]
            rates = [
                f"{r.detection_counts.get(lbl, 0) / r.frame_count:.4f}"
                if r.frame_count else "0"
                for lbl in labels
            ]
            confs = [
                f"{r.confidence_sums[lbl] / r.detection_counts[lbl]:.4f}"
                if r.detection_counts.get(lbl, 0) else ""
                for lbl in labels
            ]
            writer.writerow(base + dets + rates + confs)

    print(f"[vision_benchmark] CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# Live capture stress test
# ---------------------------------------------------------------------------

# Recommendation thresholds (Hz)
_SAFE_HZ_TIERS = [60, 30, 20]


@dataclass
class LiveStressResult:
    """Result of a live screen-capture + inference stress test."""
    engine_name: str
    duration_s: float
    frame_count: int
    # Full-pipeline latencies: capture + greyscale conversion + inference
    latencies: list[float] = field(default_factory=list, repr=False)

    fps: float = 0.0
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    recommended_hz: int = 0

    def finalise(self) -> None:
        if not self.latencies:
            return
        arr = np.array(self.latencies) * 1000  # → ms
        self.fps       = 1000.0 / arr.mean() if arr.mean() > 0 else 0.0
        self.mean_ms   = float(arr.mean())
        self.median_ms = float(np.median(arr))
        self.p95_ms    = float(np.percentile(arr, 95))
        self.max_ms    = float(arr.max())
        # Recommend the highest safe tier with ≥20% headroom
        for tier in _SAFE_HZ_TIERS:
            if self.fps >= tier * 1.2:
                self.recommended_hz = tier
                break
        else:
            self.recommended_hz = 0  # below 20 Hz even with headroom


def run_live_stress_test(
    engine_name: str,
    duration_s: float = 10.0,
    warmup: int = 10,
) -> LiveStressResult:
    """
    Capture live frames from the monitor and run the vision engine on each,
    measuring the full pipeline FPS (capture + colour conversion + inference).

    Args:
        engine_name: Vision engine to test (e.g. 'PIXEL').
        duration_s:  How many seconds to run the timed portion.
        warmup:      Number of frames to discard before timing starts.

    Returns:
        LiveStressResult with timing stats and a recommended poll Hz.
    """
    eng = vision.registry.create(engine_name.upper())
    eng.load(TARGETS, ASSETS_DIR)

    result = LiveStressResult(
        engine_name=engine_name.upper(),
        duration_s=duration_s,
        frame_count=0,
    )

    with mss.mss() as sct:
        monitor = sct.monitors[MONITOR_INDEX]

        color_flag = cv2.COLOR_BGRA2BGR if eng.needs_color else cv2.COLOR_BGRA2GRAY

        # Warmup — not timed
        for _ in range(warmup):
            raw = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(raw), color_flag)
            eng.detect(frame)

        # Timed run
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            raw = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(raw), color_flag)
            eng.detect(frame)
            result.latencies.append(time.perf_counter() - t0)

    result.frame_count = len(result.latencies)
    result.finalise()
    return result


def _print_live_report(result: LiveStressResult) -> None:
    print("\n" + "=" * 72)
    print("  LIVE CAPTURE STRESS TEST RESULTS")
    print("=" * 72)
    print(f"  Engine   : {result.engine_name}")
    print(f"  Duration : {result.duration_s:.0f}s")
    print(f"  Frames   : {result.frame_count}")
    print()
    print(f"  Sustained FPS : {result.fps:>7.1f}")
    print(f"  Mean latency  : {result.mean_ms:>7.2f} ms")
    print(f"  Median latency: {result.median_ms:>7.2f} ms")
    print(f"  p95 latency   : {result.p95_ms:>7.2f} ms")
    print(f"  Max latency   : {result.max_ms:>7.2f} ms")
    print()

    if result.recommended_hz > 0:
        print(f"  Recommendation: Safe to run at {result.recommended_hz} Hz")
        print(f"  (sustained {result.fps:.1f} FPS >= {result.recommended_hz} Hz x 1.2 headroom)")
    else:
        print(f"  Recommendation: Below 20 Hz threshold ({result.fps:.1f} FPS sustained).")
        print("  Consider using a lighter engine or reducing the capture ROI.")

    print()
    # Visual tier summary
    for tier in _SAFE_HZ_TIERS:
        needed = tier * 1.2
        status = "YES" if result.fps >= needed else "NO "
        bar    = "#" * int(min(result.fps, needed) / needed * 20)
        print(f"  {tier:>3} Hz  [{status}]  {bar:<20}  (need {needed:.0f} FPS, got {result.fps:.1f})")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark vision engines on saved screenshots."
    )
    p.add_argument(
        "--engines", nargs="+", default=["PIXEL", "SIFT", "ORB"],
        metavar="ENGINE",
        help="Engine(s) to benchmark (default: PIXEL SIFT ORB)",
    )
    p.add_argument(
        "--img-dir", default=SCREENSHOT_DIR,
        help=f"Screenshot directory (default: {SCREENSHOT_DIR})",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max images per engine (default: all)",
    )
    p.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup frames before timing (default: 5)",
    )
    p.add_argument(
        "--csv", default=None,
        help="Optional path to save summary CSV",
    )
    p.add_argument(
        "--live", action="store_true",
        help="Run a live capture stress test instead of the offline benchmark",
    )
    p.add_argument(
        "--live-engine", default="PIXEL",
        help="Engine to use for the live stress test (default: PIXEL)",
    )
    p.add_argument(
        "--live-duration", type=float, default=10.0,
        help="Duration of the live stress test in seconds (default: 10)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.live:
        print(
            f"[vision_benchmark] Live stress test — engine={args.live_engine.upper()}, "
            f"duration={args.live_duration:.0f}s"
        )
        print("[vision_benchmark] Warming up...", flush=True)
        live_result = run_live_stress_test(
            engine_name=args.live_engine,
            duration_s=args.live_duration,
        )
        _print_live_report(live_result)
    else:
        print(f"[vision_benchmark] Loading images from '{args.img_dir}'...")
        frames = _load_images(args.img_dir, args.limit)
        print(f"[vision_benchmark] {len(frames)} frames loaded.")

        results = []
        for engine_name in args.engines:
            print(f"[vision_benchmark] Running {engine_name}...", end=" ", flush=True)
            try:
                result = run_engine_benchmark(engine_name.upper(), frames, args.warmup)
                results.append(result)
                print(f"done — {result.fps:.1f} FPS")
            except Exception as exc:
                print(f"FAILED ({exc})")

        if results:
            _print_report(results)
            if args.csv:
                _save_csv(results, args.csv)
