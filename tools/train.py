"""
YOLO training wrapper.

Trains a YOLOv8 model on the auto-labeled dataset.
Output model is saved to: data/yolo_dataset/train/weights/best.pt
"""

from __future__ import annotations

import os

DATASET_YAML = os.path.join("data", "yolo_dataset", "dataset.yaml")
OUTPUT_DIR   = os.path.join("data", "yolo_dataset")
DEFAULT_MODEL = "yolov8n.pt"  # nano — fastest inference, good starting point


def run(epochs: int = 100, imgsz: int = 640, base_model: str = DEFAULT_MODEL) -> None:
    """
    Args:
        epochs:     Number of training epochs (default 100).
        imgsz:      Input image size for training (default 640).
        base_model: YOLOv8 base model to fine-tune from (default yolov8n.pt).
                    Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
                    n=fastest, x=most accurate.
    """
    if not os.path.exists(DATASET_YAML):
        print(f"[Train] Error: dataset.yaml not found at '{DATASET_YAML}'.")
        print("[Train] Run 'uv run main.py autolabel' first.")
        return

    from ultralytics import YOLO  # deferred import — only needed when training

    print("[Train] Starting YOLO training...")
    print(f"  Base model : {base_model}")
    print(f"  Dataset    : {DATASET_YAML}")
    print(f"  Epochs     : {epochs}")
    print(f"  Image size : {imgsz}")

    model = YOLO(base_model)
    model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        project=OUTPUT_DIR,
        name="train",
        exist_ok=True,
    )

    best = os.path.join(OUTPUT_DIR, "train", "weights", "best.pt")
    print(f"\n[Train] Done. Best model saved to: {best}")
