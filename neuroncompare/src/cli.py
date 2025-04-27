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
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Initialize configuration
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()
    
    # Create and run the pipeline
    pipeline = Pipeline()
    
    if args.stage:
        # Run only a specific stage
        stage_method = getattr(pipeline, args.stage, None)
        if not stage_method:
            print(f"Error: Stage {args.stage} not found")
            return 1
            
        success = stage_method()
    else:
        # Run the complete pipeline
        success = pipeline.run()
    
    if not success:
        print("Pipeline failed. Check logs for details.")
        return 1
        
    print("Pipeline completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())