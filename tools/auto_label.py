"""
Auto-labeler: runs vision engines on raw screenshots and generates YOLO label files.

Pipeline:
  1. Reads all .png images from:
       - data/yolo_dataset/images/raw/     (unlabeled; detection is run on these)
       - data/yolo_dataset/images/labeled/ (pre-labeled by trigger mode; .txt copied directly)
  2. Groups TARGETS by their autolabel_engine (default: PIXEL).
  3. For raw images: runs each engine on every image and merges detections.
     For pre-labeled images: reads the existing .txt from labels/labeled/ as-is.
  4. Converts raw detections to YOLO format (normalised x_center, y_center, w, h)
  5. Pools both sources, shuffles, and splits into train/val (default 80/20)
  6. Copies images and labels to train/ and val/ directories
  7. Writes dataset.yaml

Per-target engine override:
  Set ``autolabel_engine`` in a target's config entry to use a different engine
  for that target during labeling.  Example: JUMP_CUE uses SIFT because its
  icon is animated and scale-varying — PIXEL template matching requires exact
  pixel dimensions and cannot handle it.  SIFT is scale- and rotation-invariant
  and detects the icon reliably from a single template crop.

YOLO label format (one line per detection):
  <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
"""

from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict

import cv2

import vision
from calibration.config import (
    ASSETS_DIR,
    TARGETS,
    YOLO_CLASSES,
    YOLO_LABELED_IMAGES_DIR,
    YOLO_LABELED_LABELS_DIR,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join("data", "yolo_dataset")
RAW_DIR      = os.path.join(DATASET_DIR, "images", "raw")
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

# Class IDs — must match YOLO_CLASSES order so trigger-mode labels and
# autolabel-generated labels use the same class_id numbering.
CLASS_NAMES: list[str] = YOLO_CLASSES
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


def _build_engines() -> list[vision.VisionEngine]:
    """
    Load one engine instance per unique autolabel_engine value in TARGETS.
    Each engine is loaded with only the targets assigned to it.
    """
    engine_targets: dict[str, dict] = defaultdict(dict)
    for label, cfg in TARGETS.items():
        engine_name = cfg.get("autolabel_engine", "PIXEL").upper()
        engine_targets[engine_name][label] = cfg

    engines: list[vision.VisionEngine] = []
    for engine_name, targets in engine_targets.items():
        eng = vision.registry.create(engine_name)
        eng.load(targets, ASSETS_DIR)
        engines.append(eng)
        print(f"[AutoLabel] Engine '{engine_name}' handles: {sorted(targets.keys())}")

    return engines


def _detection_to_yolo(det: vision.Detection, img_w: int, img_h: int) -> str:
    """Convert a Detection to a YOLO label line (normalised coords)."""
    x_center = (det.x + det.w / 2) / img_w
    y_center  = (det.y + det.h / 2) / img_h
    w_norm    = det.w / img_w
    h_norm    = det.h / img_h
    cls_id    = CLASS_ID[det.label]
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def _write_dataset_yaml() -> None:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = (
        f"path: {os.path.abspath(DATASET_DIR)}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
        f"{names_block}\n"
    )
    with open(DATASET_YAML, "w") as f:
        f.write(yaml_content)
    print(f"[AutoLabel] Written: {DATASET_YAML}")


def run(val_split: float = 0.2, include_negatives: bool = True) -> None:
    """
    Args:
        val_split:         Fraction of images reserved for validation (default 0.2).
        include_negatives: If True, images with zero detections are included as
                           negative samples (empty label file). Helps reduce false positives.
    """
    # ── Collect images from both sources ─────────────────────────────────────
    raw_files: list[str] = []
    if os.path.isdir(RAW_DIR):
        raw_files = sorted([f for f in os.listdir(RAW_DIR) if f.lower().endswith(".png")])

    prelabeled_files: list[str] = []
    if os.path.isdir(YOLO_LABELED_IMAGES_DIR):
        prelabeled_files = sorted([
            f for f in os.listdir(YOLO_LABELED_IMAGES_DIR)
            if f.lower().endswith(".png")
        ])

    if not raw_files and not prelabeled_files:
        print(f"[AutoLabel] No images found in '{RAW_DIR}' or '{YOLO_LABELED_IMAGES_DIR}'.")
        print("[AutoLabel] Run 'uv run main.py collect' first.")
        return

    print(
        f"[AutoLabel] Found {len(raw_files)} raw + "
        f"{len(prelabeled_files)} pre-labeled = "
        f"{len(raw_files) + len(prelabeled_files)} total images."
    )

    # Engines are only needed for raw images.
    engines = _build_engines() if raw_files else []

    # Prepare output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Tag each item with its source so the processing loop knows what to do.
    # ("raw", filename) → run detection
    # ("labeled", filename) → copy existing .txt from labels/labeled/
    all_items: list[tuple[str, str]] = (
        [("raw", f) for f in raw_files]
        + [("labeled", f) for f in prelabeled_files]
    )
    random.shuffle(all_items)
    split_idx = int(len(all_items) * (1.0 - val_split))
    split_map: dict[str, list[tuple[str, str]]] = {
        "train": all_items[:split_idx],
        "val":   all_items[split_idx:],
    }

    total_labeled = 0
    total_negative = 0

    for split, items in split_map.items():
        for source, filename in items:
            stem = os.path.splitext(filename)[0]
            dst_img   = os.path.join(DATASET_DIR, "images", split, filename)
            dst_label = os.path.join(DATASET_DIR, "labels", split, stem + ".txt")

            if source == "labeled":
                # Pre-labeled by trigger mode: copy the existing .txt directly.
                img_path = os.path.join(YOLO_LABELED_IMAGES_DIR, filename)
                src_label = os.path.join(YOLO_LABELED_LABELS_DIR, stem + ".txt")

                label_lines: list[str] = []
                if os.path.exists(src_label):
                    with open(src_label) as f:
                        label_lines = [ln.strip() for ln in f if ln.strip()]

                if not label_lines and not include_negatives:
                    continue

                shutil.copy2(img_path, dst_img)
                with open(dst_label, "w") as f:
                    f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

            else:
                # Raw image: run detection to generate labels.
                img_path = os.path.join(RAW_DIR, filename)
                bgr_frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if bgr_frame is None:
                    print(f"[AutoLabel] Warning: could not read '{filename}', skipping.")
                    continue

                grey_frame: cv2.typing.MatLike | None = None

                def _grey() -> cv2.typing.MatLike:
                    nonlocal grey_frame
                    if grey_frame is None:
                        grey_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                    return grey_frame

                img_h, img_w = bgr_frame.shape[:2]

                detections: list[vision.Detection] = []
                for eng in engines:
                    frame = bgr_frame if eng.needs_color else _grey()
                    detections.extend(eng.detect(frame))

                if not detections and not include_negatives:
                    continue

                shutil.copy2(img_path, dst_img)
                with open(dst_label, "w") as f:
                    for det in detections:
                        f.write(_detection_to_yolo(det, img_w, img_h) + "\n")

                label_lines = detections  # type: ignore[assignment]  # truthy check only

            if label_lines:
                total_labeled += 1
            else:
                total_negative += 1

    _write_dataset_yaml()

    print("[AutoLabel] Complete.")
    print(f"  Raw images     : {len(raw_files)}")
    print(f"  Pre-labeled    : {len(prelabeled_files)}")
    print(f"  Labeled images : {total_labeled}")
    print(f"  Negative images: {total_negative}")
    print(f"  Train / Val    : {len(split_map['train'])} / {len(split_map['val'])}")
    print(f"  Classes        : {CLASS_NAMES}")