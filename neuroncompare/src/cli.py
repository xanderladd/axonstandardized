# neuroncompare/src/cli.py
import argparse
import os
import sys

from neuroncompare.src.config_manager import get_config
from neuroncompare.src.pipeline import Pipeline

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run neuroncompare pipeline')
    parser.add_argument('--config', help='Path to input.txt configuration file')
    parser.add_argument('--stage', help='Run only a specific pipeline stage')
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase verbosity (can be used multiple times)')
    parser.add_argument('--log-file', help='Path to log file')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume pipeline from run directory')
    parser.add_argument('--run-dir', help='Run directory path (for resume mode)')
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # ADD THIS BLOCK before pipeline creation:
    if args.resume:
        # Change to run directory for resume mode
        run_dir = args.run_dir or os.getcwd()
        os.chdir(run_dir)
        # Now we're in the run directory with its input.txt
    
    # Initialize configuration (this stays the same)
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()
    
    # Create and run the pipeline (ADD resume parameter)
    pipeline = Pipeline(resume_mode=args.resume)
    
    # Rest stays the same...
    if args.stage:
        stage_method = getattr(pipeline, args.stage, None)
        if not stage_method:
            print(f"Error: Stage {args.stage} not found")
            return 1
        success = stage_method()
    else:
        success = pipeline.run()
    
    if not success:
        print("Pipeline failed. Check logs for details.")
        return 1
        
    print("Pipeline completed successfully.")
    return 0
    
if __name__ == "__main__":
    sys.exit(main())