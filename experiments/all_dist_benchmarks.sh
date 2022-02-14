#!/bin/bash
#SBATCH -J OUT_salient_dist_bench
#SBATCH -o %x-%A.%a.out
#SBATCH -e %x-%A.%a.err
#SBATCH --nodes=1
#SBATCH -t 0-4:00
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -a 1-16
#SBATCH --qos=high

source /etc/profile

#module load anaconda/2020b
#module load mpi/openmpi-4.0
#source activate /home/gridsan/tkaler/.conda/envs/tfkmain2

DATASET_NAME="ogbn-papers100M"

# For shorter benchmarks, change the value below (e.g., from 5 to 2).
NUM_TRIALS="5"
NUM_TRIALS_PYG="1"

# Configure the option below based on the nodes you requested from the cluster.
# 
# If you specified '#SBATCH --gres=gpu:k' then set NUM_DEVICES_PER_NODE=k
# If you specified '#SBATCH -a 1-N' then set NUM_NODES=N
NUM_DEVICES_PER_NODE=1
NUM_NODES=16

# When there is 1 GPU per-machine and the machine has N physical cores with 2-way hyperthreading, 
#   a decent rule of thumb is to set the number of sampling workers to N + N/2.
# If there are K GPUs per-machine, a reasonable value is (N+N/2)/k.
# These are just rules of thumb. If the above guidelines suggest you use less than 20 workers, then
#   change the value below. Otherwise, it is fine to keep the number of sampling workers at 20.
NUM_SAMPLING_WORKERS=20


#
# CONFIGURE BATCH SIZES
# 	Configure batch sizes based on GPU memory limits. 
#	The limits below should work on GPUs with at least 16GB of memory.
#

# Most architectures train with fanouts 15,10,5
TRAIN_BATCH_SIZE=1024

# GIN trains with 20,20,20 fanouts so use a smaller batch size.
GIN_TRAIN_BATCH_SIZE=512

# Typical validation and test fanouts are 20,20,20 so use a smaller batch size.
VALID_BATCH_SIZE=512
TEST_BATCH_SIZE=512

# SAGEResInception uses final fanouts 100,100,100 so should set a smaller batch size.
TEST_BATCH_SIZE_SAGERI=32


#
# Obtain script directory from ./run_distributed.sh and setup other paths.
# 	No need to change.
#
SCRIPT_DIR=$2 

# Set JOB_NAME used for this script
export SLURMD_NODENAME=`hostname`
export PYTHONPATH=$SCRIPT_DIR/../
DATASET_ROOT=$SCRIPT_DIR/dataset
OUTPUT_ROOT=$SCRIPT_DIR/distributed_job_output


#
# INSTRUCTIONS: Please review the benchmarks below and uncomment the ones you wish to run.
#

## Experiment on SAGE using SALIENT
#touch $OUTPUT_ROOT/nodelist_$1_sage/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_sage --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 15 10 5 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize 512 --model_name SAGE --ddp_dir $OUTPUT_ROOT/nodelist_$1_sage/ --use_lrs --patience 125 --trials $NUM_TRIALS --use_lrs
#
## Experiment on GIN using SALIENT
#touch $OUTPUT_ROOT/nodelist_$1_gin/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_gin --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $GIN_TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 20 20 20 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize $TEST_BATCH_SIZE --model_name GIN --ddp_dir $OUTPUT_ROOT/nodelist_$1_gin/ --use_lrs --patience 125 --trials $NUM_TRIALS --use_lrs
#
## Experiment on GAT using SALIENT
#touch $OUTPUT_ROOT/nodelist_$1_gat/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_gat --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 15 10 5 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize $TEST_BATCH_SIZE --model_name GAT --ddp_dir $OUTPUT_ROOT/nodelist_$1_gat/ --use_lrs --patience 125 --trials $NUM_TRIALS --use_lrs
#
## Experiment on SAGEResInception using SALIENT
#touch $OUTPUT_ROOT/nodelist_$1_sageri/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_sageri --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.002 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=1024 --num_layers=3 --train_fanouts 12 12 12 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 100 100 100 --final_test_batchsize $TEST_BATCH_SIZE_SAGERI --model_name SAGEResInception --ddp_dir $OUTPUT_ROOT/nodelist_$1_sageri/ --use_lrs --patience 125 --trials $NUM_TRIALS --use_lrs
#
## Experiment on SAGE using PyG
#touch $OUTPUT_ROOT/nodelist_$1_sage_pyg/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_sage_pyg --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 15 10 5 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize $TEST_BATCH_SIZE --model_name SAGE --ddp_dir $OUTPUT_ROOT/nodelist_$1_sage_pyg/ --one_node_ddp --use_lrs --patience 125 --trials $NUM_TRIALS_PYG --use_lrs --train_sampler NeighborSampler
#
## Experiment on GIN using PyG
#touch $OUTPUT_ROOT/nodelist_$1_gin_pyg/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_gin_pyg --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $GIN_TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 20 20 20 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize $TEST_BATCH_SIZE --model_name GIN --ddp_dir $OUTPUT_ROOT/nodelist_$1_gin_pyg/ --one_node_ddp --use_lrs --patience 125 --trials $NUM_TRIALS_PYG --use_lrs --train_sampler NeighborSampler
#
#
## Experiment on SAGEResInception using PyG
#touch $OUTPUT_ROOT/nodelist_$1_sageri_pyg/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_sageri_pyg --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.002 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=1024 --num_layers=3 --train_fanouts 12 12 12 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 100 100 100 --final_test_batchsize $TEST_BATCH_SIZE_SAGERI --model_name SAGEResInception --ddp_dir $OUTPUT_ROOT/nodelist_$1_sageri_pyg/ --one_node_ddp --use_lrs --patience 125 --trials $NUM_TRIALS_PYG --use_lrs --train_sampler NeighborSampler
#
#
## Experiment on GAT using PyG.
#touch $OUTPUT_ROOT/nodelist_$1_gat_pyg/$SLURMD_NODENAME
#python -m driver.main $DATASET_NAME $1_gat_pyg --train_type serial --max_num_devices_per_node $NUM_DEVICES_PER_NODE --total_num_nodes $NUM_NODES --test_epoch_frequency 1 --epochs 25 --overwrite_job_dir --output_root $OUTPUT_ROOT --dataset_root=$DATASET_ROOT --train_batch_size $TRAIN_BATCH_SIZE --test_batch_size $VALID_BATCH_SIZE --lr 0.01 --num_workers=$NUM_SAMPLING_WORKERS  --hidden_features=256 --num_layers=3 --train_fanouts 15 10 5 --batchwise_test_fanouts 20 20 20 --final_test_fanouts 20 20 20 --final_test_batchsize $TEST_BATCH_SIZE --model_name GAT --ddp_dir $OUTPUT_ROOT/nodelist_$1_gat_pyg/ --one_node_ddp --use_lrs --patience 125 --trials $NUM_TRIALS_PYG --use_lrs --train_sampler NeighborSampler
