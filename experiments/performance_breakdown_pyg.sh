#!/bin/bash

# Script to run examples on a single machine. READ ALL INSTRUCTIONS!!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set JOB_NAME used for this script
JOB_NAME=single_gpu_pyg_$1

# Set environment variable SLURMD_NODENAME so that os.environ can get
# it. Usually, on a cluster with SLURM scheduler, the scheduler will
# set this variable. In other cases, manually set it like the
# following:
export SLURMD_NODENAME=`hostname`


# Set SALIENT root and PYTHONPATH
#SALIENT_ROOT=$SCRIPT_DIR/SALIENT_codes

export PYTHONPATH=$SCRIPT_DIR/../

# Set the data paths
DATASET_ROOT=$SCRIPT_DIR/dataset
OUTPUT_ROOT=$SCRIPT_DIR/job_output_single_gpu

mkdir -p $OUTPUT_ROOT

DATASET_NAME=$1


test -f "$SCRIPT_DIR/.ran_config" || echo "[Warning] You have not run the configuration script. You can do so by running 'python experiments/configure_for_environment.py'"

if [[ "$1" = "ogbn-arxiv" ]]	|| [[ "$1" = "ogbn-products" ]] || [[ "$1" = "ogbn-papers100M" ]]; then
	# Run examples. For the full list of options, see driver/parser.py
	#
	# # 1 node, 1 GPU, no ddp
	python -m driver.main $DATASET_NAME $JOB_NAME \
	--config_file $SCRIPT_DIR/performance_breakdown_config.cfg --dataset_root $DATASET_ROOT --output_root $OUTPUT_ROOT --overwrite_job_dir --train_sampler NeighborSampler
else
	echo "[Error] Did not provide a valid dataset. Options shown below"
	echo "	Usage: $0 [ogbn-arxiv, ogbn-products, ogbn-papers100M]"
	exit
fi





