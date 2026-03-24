import argparse
import sys
from calibration import run_recorder, run_collector, run_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Expedition 33 RL Agent — Vision System CLI"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Choose a mode to run")

    # ── record ────────────────────────────────────────────────────────────────
    parser_record = subparsers.add_parser(
        "record", help="Run the calibration recorder with a chosen vision engine"
    )
    parser_record.add_argument(
        "-e", "--engine",
        choices=["pixel", "sift", "orb", "yolo"],
        default="pixel",
        help="Vision engine to use (default: pixel)",
    )

    # ── collect ───────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "collect", help="Capture screenshots for YOLO training data (F9/F10/F11)"
    )

    # ── autolabel ─────────────────────────────────────────────────────────────
    parser_label = subparsers.add_parser(
        "autolabel",
        help="Auto-label raw screenshots with PixelEngine and generate YOLO dataset",
    )
    parser_label.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of images for validation (default: 0.2)",
    )
    parser_label.add_argument(
        "--no-negatives",
        action="store_true",
        help="Exclude images with zero detections from the dataset",
    )

    # ── train ─────────────────────────────────────────────────────────────────
    parser_train = subparsers.add_parser(
        "train", help="Train a YOLOv8 model on the auto-labeled dataset"
    )
    parser_train.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser_train.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (default: 640)"
    )
    parser_train.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLOv8 model to fine-tune (default: yolov8n.pt)",
    )

    # ── analyze ───────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "analyze", help="Analyze calibration logs and calculate the optimal ROI"
    )

    # ── routing ───────────────────────────────────────────────────────────────
    args = parser.parse_args()

    if args.mode == "record":
        print(f">> Launching Recorder with {args.engine.upper()} engine...")
        run_recorder(engine=args.engine)

    elif args.mode == "collect":
        run_collector()

    elif args.mode == "autolabel":
        from tools.auto_label import run as run_autolabel
        print(">> Running Auto-Labeler...")
        run_autolabel(
            val_split=args.val_split,
            include_negatives=not args.no_negatives,
        )

    elif args.mode == "train":
        from tools.train import run as run_train
        print(">> Launching YOLO Training...")
        run_train(epochs=args.epochs, imgsz=args.imgsz, base_model=args.model)

    elif args.mode == "analyze":
        print(">> Running Analysis Tool...")
        run_analysis()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
