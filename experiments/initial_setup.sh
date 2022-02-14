#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SCRIPT_DIR/../


echo "Configuring environment..."
python $SCRIPT_DIR/configure_for_environment.py && python $SCRIPT_DIR/download_datasets_fast.py || echo "Fatal Error: Error while trying to configure for environment. The configuration may need to be done manually by modifying the configuration file and downloading datasets directly."
