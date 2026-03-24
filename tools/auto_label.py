"""
Auto-labeler: runs PixelEngine on raw screenshots and generates YOLO label files.

Pipeline:
  1. Reads all .png images from data/yolo_dataset/images/raw/
  2. Runs PixelEngine on each frame
  3. Converts detections to YOLO format (normalised x_center, y_center, w, h)
  4. Splits into train/val (default 80/20)
  5. Copies images and labels to train/ and val/ directories
  6. Writes dataset.yaml

YOLO label format (one line per detection):
  <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
"""

from __future__ import annotations

import os
import random
import shutil

import cv2

from calibration.config import TARGETS, ASSETS_DIR
import vision

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR  = os.path.join("data", "yolo_dataset")
RAW_DIR      = os.path.join(DATASET_DIR, "images", "raw")
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

SPLITS = {
    "train": 0.8,
    "val":   0.2,
}

# Class IDs — sorted alphabetically for reproducibility
CLASS_NAMES: list[str] = sorted(TARGETS.keys())
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


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
    # Collect raw images
    if not os.path.isdir(RAW_DIR):
        print(f"[AutoLabel] Error: raw image directory not found: '{RAW_DIR}'")
        print("[AutoLabel] Run 'uv run main.py collect' first.")
        return

    image_files = sorted([
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".png")
    ])

    if not image_files:
        print(f"[AutoLabel] No images found in '{RAW_DIR}'.")
        return

    print(f"[AutoLabel] Found {len(image_files)} images.")

    # Load PixelEngine once
    engine = vision.registry.create("PIXEL")
    engine.load(TARGETS, ASSETS_DIR)

    # Prepare output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1.0 - val_split))
    split_map: dict[str, list[str]] = {
        "train": image_files[:split_idx],
        "val":   image_files[split_idx:],
    }

    total_labeled = 0
    total_negative = 0

    for split, files in split_map.items():
        for filename in files:
            img_path = os.path.join(RAW_DIR, filename)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"[AutoLabel] Warning: could not read '{filename}', skipping.")
                continue

            img_h, img_w = frame.shape
            detections = engine.detect(frame)

            if not detections and not include_negatives:
                continue

            # Copy image to split directory
            dst_img = os.path.join(DATASET_DIR, "images", split, filename)
            shutil.copy2(img_path, dst_img)

            # Write label file (empty if no detections → negative sample)
            label_filename = os.path.splitext(filename)[0] + ".txt"
            dst_label = os.path.join(DATASET_DIR, "labels", split, label_filename)
            with open(dst_label, "w") as f:
                for det in detections:
                    f.write(_detection_to_yolo(det, img_w, img_h) + "\n")

            if detections:
                total_labeled += 1
            else:
                total_negative += 1

    _write_dataset_yaml()

    print(f"[AutoLabel] Complete.")
    print(f"  Labeled images : {total_labeled}")
    print(f"  Negative images: {total_negative}")
    print(f"  Train / Val    : {len(split_map['train'])} / {len(split_map['val'])}")
    print(f"  Classes        : {CLASS_NAMES}")
