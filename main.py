import argparse
import sys
from calibration import run_recorder

def main():
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Expedition 33 RL Agent - Vision System CLI")
    
    # Create subcommands (modes)
    subparsers = parser.add_subparsers(dest="mode", help="Choose a mode to run")

    # Command 1: record (The original calibration)
    parser_record = subparsers.add_parser("record", help="Start recording gameplay data for calibration")

    # Parse arguments
    args = parser.parse_args()

    # Routing Logic
    if args.mode == "record":
        print(">> Launching Recorder...")
        run_recorder()
        
    else:
        # If no arguments provided, show help
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()