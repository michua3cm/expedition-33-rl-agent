import argparse
import sys

from calibration import run_analysis, run_collector, run_recorder


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
        "collect", help="Capture screenshots for YOLO training data (F8 trigger / F9 manual / F10 auto)"
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

    # ── status ────────────────────────────────────────────────────────────────
    parser_status = subparsers.add_parser(
        "status", help="Report per-class label counts and flag classes needing more data"
    )
    parser_status.add_argument(
        "--target",
        type=int,
        default=50,
        help="Minimum instances per class to be considered ready (default: 50)",
    )

    # ── demo ──────────────────────────────────────────────────────────────────
    parser_demo = subparsers.add_parser(
        "demo", help="Record human gameplay demonstrations for imitation learning"
    )
    parser_demo.add_argument(
        "--session", default="demo", help="Output filename stem (default: demo)"
    )
    parser_demo.add_argument(
        "--env",
        choices=["vision", "ue4ss"],
        default="vision",
        help="Observation source: 'vision' (screen capture) or 'ue4ss' (UE4SS Lua mod, default: vision)",
    )
    parser_demo.add_argument(
        "--engine", default="PIXEL",
        help="Vision engine — only used when --env vision (default: PIXEL)",
    )
    parser_demo.add_argument(
        "--hz", type=float, default=20.0, help="Capture rate in Hz (default: 20)"
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

    elif args.mode == "status":
        from tools.dataset_status import run as run_status
        run_status(target=args.target)

    elif args.mode == "demo":
        import time

        from tools.demo_recorder import DemoRecorder, UE4SSDemoRecorder

        if args.env == "ue4ss":
            from environment.ue4ss_reader import StateReader
            reader = StateReader()
            if not reader.is_available():
                print(
                    ">> [demo] WARNING: UE4SS state file not found. "
                    "Start the game with the UE4SS StateReader mod before recording."
                )
            rec: DemoRecorder = UE4SSDemoRecorder(
                reader, session_name=args.session, poll_hz=args.hz
            )
            print(f">> [demo] UE4SS mode — obs dim=9, session='{args.session}'")
        else:
            from environment.instance import GameInstance
            game = GameInstance(engine=args.engine)
            rec = DemoRecorder(game, session_name=args.session, poll_hz=args.hz)
            print(
                f">> [demo] Vision mode — engine={args.engine.upper()}, "
                f"obs dim=30, session='{args.session}'"
            )

        rec.start()
        try:
            while True:
                time.sleep(0.5)
                print(
                    f"\r>> [demo] {rec.frame_count} frames captured... (Ctrl+C to stop)",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            print()
            path = rec.stop()
            if path:
                print(f">> [demo] Saved: {path}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
