import argparse
import sys
from calibration import run_recorder, run_analysis

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Expedition 33 RL Agent - Vision System CLI")
    
    # Create subcommands (modes)
    subparsers = parser.add_subparsers(dest="mode", help="Choose a mode to run")

    # Command 1: record (The original calibration)
    parser_record = subparsers.add_parser("record", help="Start recording gameplay data for calibration")
    # Add engine selection
    parser_record.add_argument("--engine", choices=["pixel", "sift"], default="pixel", help="Vision engine to use (pixel or sift)")
    
    # Command 2: analyze (The new analysis tool)
    parser_analyze = subparsers.add_parser("analyze", help="Analyze logs and calculate the optimal ROI")

    # Parse arguments
    args = parser.parse_args()

    # Routing Logic
    if args.mode == "record":
        print(f">> Launching Recorder with {args.engine.upper()} engine...")
        run_recorder(engine=args.engine)
        
    elif args.mode == "analyze":
        print(">> Running Analysis Tool...")
        run_analysis()
        
    else:
        # If no arguments provided, show help
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()