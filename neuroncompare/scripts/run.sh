#!/usr/bin/env bash
# neuroncompare/run.py

# Get the directory this script is in
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the Python pipeline
python -m neuroncompare.src.cli "$@"