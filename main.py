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

    # ── gail-train ────────────────────────────────────────────────────────────
    parser_gail = subparsers.add_parser(
        "gail-train", help="Train a GAIL imitation learning agent from human demos"
    )
    parser_gail.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total environment steps to train for (default: 200000)",
    )
    parser_gail.add_argument(
        "-e", "--engine",
        choices=["pixel", "sift", "orb", "yolo"],
        default="pixel",
        help="Vision engine to use (default: pixel)",
    )
    parser_gail.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps per episode before truncation (default: 1000)",
    )
    parser_gail.add_argument(
        "--demos-dir", default="data/demos",
        help="Directory containing .npz demo files (default: data/demos)",
    )
    parser_gail.add_argument(
        "--checkpoint", default="data/models",
        help="Output directory for saved model checkpoint (default: data/models)",
    )
    parser_gail.add_argument(
        "--no-cuda", action="store_true",
        help="Disable GPU and force CPU training",
    )

    # ── rl-train ──────────────────────────────────────────────────────────────
    parser_rl = subparsers.add_parser(
        "rl-train", help="Fine-tune a PPO policy on the live game (Phase 1 RL)"
    )
    parser_rl.add_argument(
        "--bc-checkpoint", type=str, default=None,
        help="BC actor warm-start checkpoint path (optional)",
    )
    parser_rl.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total environment steps (default: 100 000)",
    )
    parser_rl.add_argument(
        "--engine", default="PIXEL",
        help="Vision engine for observation capture (default: PIXEL)",
    )
    parser_rl.add_argument(
        "--n-steps", type=int, default=2048,
        help="PPO rollout steps per update (default: 2048)",
    )
    parser_rl.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO mini-batch size (default: 64)",
    )
    parser_rl.add_argument(
        "--n-epochs", type=int, default=10,
        help="PPO gradient epochs per rollout (default: 10)",
    )
    parser_rl.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser_rl.add_argument(
        "--ent-coef", type=float, default=0.01,
        help="Entropy coefficient (default: 0.01)",
    )
    parser_rl.add_argument(
        "--no-cuda", action="store_true", help="Disable GPU even if available"
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

    elif args.mode == "gail-train":
        from environment.gym_env import Expedition33Env
        from il.gail import train_gail

        device = "cpu" if args.no_cuda else "auto"
        print(
            f">> Starting GAIL training — engine={args.engine.upper()}, "
            f"timesteps={args.timesteps:,}, max_steps={args.max_steps}, "
            f"device={device}"
        )
        env = Expedition33Env(engine=args.engine.upper(), max_steps=args.max_steps)
        checkpoint = train_gail(
            env=env,
            demos_dir=args.demos_dir,
            total_timesteps=args.timesteps,
            checkpoint_dir=args.checkpoint,
            device=device,
        )
        print(f">> Checkpoint saved to: {checkpoint}")

    elif args.mode == "rl-train":
        from rl.train import train as rl_train
        print(">> Launching PPO RL training (Phase 1)...")
        rl_train(
            bc_checkpoint=args.bc_checkpoint,
            engine=args.engine,
            total_timesteps=args.timesteps,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            lr=args.lr,
            ent_coef=args.ent_coef,
            use_cuda=not args.no_cuda,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
