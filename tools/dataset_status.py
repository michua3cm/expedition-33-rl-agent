"""
Dataset status reporter.

Scans label files and reports per-class instance counts so you can see
which classes need more data before committing to a training run.

Two sources are checked in order:
  1. labels/labeled/  — trigger-mode saves (pre-autolabel)
  2. labels/train/ + labels/val/ — post-autolabel split

Whichever source has data is reported. If autolabel has already been run
the post-split counts are shown, otherwise the raw labeled/ counts are used.
Both are shown when both exist.

Output example:
  Class              Instances   Images    Status
  BATTLE_WHEEL            52       41      OK
  DODGE                   18       14      LOW  (need 32 more)
  GRADIENT_INCOMING        0        0      MISSING
  ...
"""

import os
from collections import defaultdict

from calibration.config import YOLO_CLASSES, YOLO_LABELED_LABELS_DIR

DATASET_DIR   = os.path.join("data", "yolo_dataset")
TRAIN_LABELS  = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS    = os.path.join(DATASET_DIR, "labels", "val")

DEFAULT_TARGET = 50  # recommended minimum instances per class


def _count_labels(label_dirs: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """
    Scan all .txt files in label_dirs and count per-class instances and images.

    Returns:
        instances: class_name → total bounding-box count
        images:    class_name → number of images containing ≥1 instance
    """
    instances: dict[str, int] = defaultdict(int)
    images: dict[str, int] = defaultdict(int)

    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue
            seen_in_file: set[str] = set()
            with open(os.path.join(label_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(line.split()[0])
                    if class_id < len(YOLO_CLASSES):
                        name = YOLO_CLASSES[class_id]
                        instances[name] += 1
                        seen_in_file.add(name)
            for name in seen_in_file:
                images[name] += 1

    return dict(instances), dict(images)


def _print_table(
    title: str,
    instances: dict[str, int],
    images: dict[str, int],
    target: int,
) -> None:
    print(f"\n{title}")
    print(f"  {'Class':<22} {'Instances':>10} {'Images':>8}    Status")
    print("  " + "─" * 60)

    all_ok = True
    for name in YOLO_CLASSES:
        inst  = instances.get(name, 0)
        imgs  = images.get(name, 0)
        if inst == 0:
            status = "MISSING"
            all_ok = False
        elif inst < target:
            status = f"LOW  (need {target - inst} more)"
            all_ok = False
        else:
            status = "OK"
        print(f"  {name:<22} {inst:>10} {imgs:>8}    {status}")

    if all_ok:
        print(f"\n  All classes meet the target of {target} instances. Ready to train.")


def run(target: int = DEFAULT_TARGET) -> None:
    """
    Args:
        target: Minimum instances per class to be considered ready (default 50).
    """
    print("=== Dataset Status ===")

    has_labeled = (
        os.path.isdir(YOLO_LABELED_LABELS_DIR)
        and any(f.endswith(".txt") for f in os.listdir(YOLO_LABELED_LABELS_DIR))
    )

    has_split = (
        os.path.isdir(TRAIN_LABELS)
        and any(f.endswith(".txt") for f in os.listdir(TRAIN_LABELS))
    )

    if not has_labeled and not has_split:
        print("\n  No label files found. Run 'uv run main.py collect' first.")
        return

    if has_labeled:
        inst, imgs = _count_labels([YOLO_LABELED_LABELS_DIR])
        _print_table(
            f"Pre-autolabel  ({YOLO_LABELED_LABELS_DIR})",
            inst, imgs, target,
        )

    if has_split:
        inst, imgs = _count_labels([TRAIN_LABELS, VAL_LABELS])
        _print_table(
            "Post-autolabel  (labels/train + labels/val)",
            inst, imgs, target,
        )
