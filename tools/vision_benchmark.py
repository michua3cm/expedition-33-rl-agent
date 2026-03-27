"""
Vision Engine Benchmark Tool for Clair Obscur: Expedition 33.

Runs one or more vision engines over a set of saved screenshots and
produces a performance + accuracy report so you can make an informed
decision about which engine to use for live RL training.

What it measures
----------------
- **Throughput** — frames per second each engine sustains on your hardware
- **Latency**    — mean, median, p95, and max inference time per frame (ms)
- **Detection rate** — how often each target label is detected across the batch
- **Confidence stats** — mean and max confidence per detected label

Typical workflow
----------------
1. Collect screenshots with the calibration recorder:
       uv run main.py collect
2. Run the benchmark against those screenshots:
       uv run python -m tools.vision_benchmark --engines PIXEL SIFT ORB
3. Read the printed report and pick the engine that fits your FPS budget.

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

Options
-------
    --engines       Space-separated engine names to benchmark (default: PIXEL SIFT ORB)
    --img-dir       Directory containing .png/.jpg screenshots (default: data/screenshots)
    --limit         Max number of images to process per engine (default: all)
    --warmup        Number of warmup frames before timing starts (default: 5)
    --csv           Optional path to save the per-engine summary as CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import vision
from calibration.config import TARGETS, ASSETS_DIR, SCREENSHOT_DIR


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

def _load_images(img_dir: str, limit: Optional[int]) -> list[np.ndarray]:
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
    header_labels = "".join(f"{l:>{col_w}}" for l in labels)
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
        det_cols  = [f"det_{l}" for l in labels]
        rate_cols = [f"rate_{l}" for l in labels]
        conf_cols = [f"conf_{l}" for l in labels]
        writer.writerow(base_cols + det_cols + rate_cols + conf_cols)

        for r in results:
            base = [
                r.engine_name, r.frame_count,
                f"{r.fps:.2f}", f"{r.mean_ms:.3f}",
                f"{r.median_ms:.3f}", f"{r.p95_ms:.3f}", f"{r.max_ms:.3f}",
            ]
            dets  = [r.detection_counts.get(l, 0) for l in labels]
            rates = [
                f"{r.detection_counts.get(l, 0) / r.frame_count:.4f}"
                if r.frame_count else "0"
                for l in labels
            ]
            confs = [
                f"{r.confidence_sums[l] / r.detection_counts[l]:.4f}"
                if r.detection_counts.get(l, 0) else ""
                for l in labels
            ]
            writer.writerow(base + dets + rates + confs)

    print(f"[vision_benchmark] CSV saved → {csv_path}")


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
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

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
