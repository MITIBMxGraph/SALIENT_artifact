# Artifact Evaluation Guide

This document provides a guide on how to exercise this software artifact to
reproduce key results from the paper "[Accelerating Training and Inference of
Graph Neural Networks with Fast Sampling and Pipelining](https://arxiv.org/abs/2110.08450)" published at MLSys 2022. The `experiments` directory of this repository contains
scripts that streamline the running of certain key experiments. Specifically,
the `experiments` directory includes scripts for producing: (a) the single GPU
performance breakdowns for PyG and SALIENT used for generating Table 1 and
Figure 4; and, (b) the distributed experiments with multiple GNN architectures
used for generating Figure 5 and Figure 6.


You should be able to execute the scripts in `experiments` from any working
directory. For brevity we will assume your current working directory is
`experiments` and all file and directory paths referenced in this document will
be relative to that directory. 


## Install SALIENT

### Manual installation

You may follow the instructions presented in
[INSTALL.md](INSTALL.md) to install SALIENT yourself.

### Docker container

For convenience and reproducibility, we also provide a docker environment
configured with package versions that closely match those referenced in the paper.

To use this container you must install the NVIDIA Container Toolkit, which will enable
docker to use GPUs.
Please refer to the installation instructions for [Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) or the [Windows Subsystem for Linux (WSL2) installation instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
If you plan to use a cloud provider, you can use NVIDIA's cloud images which come preinstalled with
the container toolkit.
NVIDIA provides images and instructions for the following services: [Amazon Web Services (AWS)](https://docs.nvidia.com/ngc/ngc-aws-setup-guide/index.html), [Google Cloud Platform (GCP)](https://docs.nvidia.com/ngc/ngc-gcp-setup-guide/), and [Microsoft Azure](https://docs.nvidia.com/ngc/ngc-azure-setup-guide/).

To build the SALIENT container yourself, you can use the file `docker/Dockerfile`.
To download a prebuilt image for an x86 machine, run

```bash
docker pull nistath/salient:cuda-11.1.1
```

To launch the container and mount your clone of the SALIENT repository, run

```bash
cd <ROOT OF YOUR SALIENT REPOSITORY CLONE>
docker run --ipc=host --gpus all -it -v `pwd`:/salient nistath/salient:cuda-11.1.1
cd /salient/fast_sampler && python setup.py develop
```

## Setup experiments

For clarity, we will use `${SALIENT_ROOT}` to refer to the root of your clone of this repository.
If you are using the docker instructions above, `SALIENT_ROOT=/salient`.

We have provided optional scripts that can be used to automatically configure
experiments based on the execution environment (e.g., number of cores and
available disk space).

To run these setup scripts execute the following command:

```bash
cd ${SALIENT_ROOT}/experiments
./initial_setup.sh
``` 

This script performs the following jobs:

1. Run `python configure_for_environment.py` which will:
	* Detect the number of physical CPUs present on your machine and modify the `--num_workers` argument in the experiment configuration file `performance_breakdown_config.cfg`
	* Detect the available disk space on the device containing the `experiments` directory. Based on this space, the script will determine which OGB datasets should be downloaded for experiments. 

2. Run `python download_dataset_fast.py` which will download
   preprocessed versions of the OGB datasets that are used for the
experiments. These datasets use less space than those downloaded from OGB
because we store the node features on-disk using half precision floating
point numbers. 

Note that use of the `./initial_setup.sh` script is optional. 
If you do not use the `download_dataset_fast.py` script, then you must run the single GPU experiments manually (at least once)
on each dataset.  The first execution on a dataset will download it directly from OGB and perform preprocessing on your local machine.
On large datasets (notably `ogbn-papers100M`)
downloading+preprocessing the dataset is very time consuming and requires
substantial disk space. You may modify the configuration of the single
GPU experiments manually by editing the `performance_breakdown_config.cfg` file.
A description of all runtime arguments can be obtained by running  `python -m driver.main --help` from the
root directory of the repository.

## Run single GPU experiments

A single script can be executed to run all single GPU experiments on SALIENT
and PyG. The results of these experiments produces a table that provides a
breakdown of per-epoch runtime, which shows how much time is spent sampling and slicing, transfering data, and performing
training on GPU.
The data generated can be used to reproduce Table 1 and Figure 4 in the paper.

```bash
cd ${SALIENT_ROOT}/experiments
./run_all_single_gpu_experiments.sh
```

This command will run all single GPU experiments on the OGB datasets present in
the `experiments/dataset` directory.  A table will be displayed in your
terminal summarizing the results. This table can be regenerated using results
from a previous run.  The single GPU experiments results are logged in the
`job_output_single_gpu` directory and the results can be parsed to
produce a table using the command:

```bash
cd ${SALIENT_ROOT}/experiments
python helper_scripts/parse_performance_breakdown.py job_output_single_gpu
```

Example output using NVIDIA V100 GPUs with 32GB memory:

```
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
| Salient/PyG |     Dataset     | Model |      Total (ms)     |     Train (ms)     | Sampling + Slicing (ms) | Data Transfer (ms) |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|     PyG     |    ogbn-arxiv   |  SAGE |  1683.381 ±  31.994 |   441.007 ±  4.383 |    1062.552 ±  35.268   |   179.822 ± 0.106  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|   SALIENT   |    ogbn-arxiv   |  SAGE |   476.831 ±   6.658 |   427.190 ±  3.996 |      47.508 ±   2.704   |     1.102 ± 0.033  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|     PyG     | ogbn-papers100M |  SAGE | 54950.602 ± 896.844 | 13452.845 ± 70.267 |   27394.408 ± 890.841   | 14103.350 ± 3.847  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|   SALIENT   | ogbn-papers100M |  SAGE | 16513.349 ±  43.066 | 14109.076 ± 22.836 |    2373.212 ±  36.256   |    19.297 ± 0.251  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|     PyG     |  ogbn-products  |  SAGE |  8931.308 ± 132.107 |  2415.684 ± 14.200 |    4800.256 ± 138.433   |  1715.368 ± 0.362  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
|   SALIENT   |  ogbn-products  |  SAGE |  2812.475 ±  46.674 |  2430.807 ± 14.073 |     372.100 ±  54.981   |     7.370 ± 0.368  |
+-------------+-----------------+-------+---------------------+--------------------+-------------------------+--------------------+
```

You may also run single GPU experiments manually with either SALIENT or PyG using the following two commands:

```bash
cd ${SALIENT_ROOT}/experiments
./performance_breakdown_pyg.sh <ogbn-arxiv | ogbn-products | ogbn-papers100M>
./performance_breakdown_salient.sh <ogbn-arxiv | ogbn-products | ogbn-papers100M>
```

These commands will download the specified OGB dataset if it is not already
present in the `dataset` directory.

## Configure distributed multi-GPU experiments

Scripts are provided for executing distributed multi-GPU experiments on a SLURM
cluster. The provided scripts for these experiments require customization based
on the hardware provided by your cluster, and any special SLURM options that
are needed to request appropriate compute resources.

Before running any distributed experiments, you should review the slurm batch
file `all_dist_benchmarks.sh`. 

**Important:** The distributed scripts require that there be at most one
scheduled task per compute node. If you cannot otherwise obtain this guarantee,
you must specify `--exclusive` in your batch file.

The following example excerpt from `all_dist_benchmark.sh` is a configuration that
executes on 16 GPUs using 16 compute nodes with 1 GPU each. You should read this
file carefully.


```bash
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

#
# Setup any environments as necessary.
#
#module load anaconda/2020b
#module load mpi/openmpi-4.0
#source activate /home/gridsan/tkaler/.conda/envs/tfkmain2

# Specify the dataset name.
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
```

If the machines in your cluster have multiple GPUs, you may use them by setting
appropriate values of the `NUM_NODES` and `NUM_DEVICES_PER_NODE` variables.

An example excerpt from `all_dist_benchmark.sh` is shown below for a
configuration on 16 GPUs using 8 machines with 2 GPUs each.

```bash
#!/bin/bash
#SBATCH -J OUT_salient_dist_bench
#SBATCH -o %x-%A.%a.out
#SBATCH -e %x-%A.%a.err
#SBATCH --nodes=1
#SBATCH -t 0-4:00
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH -a 1-8
#SBATCH --qos=high

NUM_DEVICES_PER_NODE=2
NUM_NODES=8
```

After making appropriate modifications to `all_dist_benchmark.sh` you should
review the different tasks listed in the file. You may choose to uncomment all
tasks to run all benchmarks in sequence using the same set of machines in the
cluster. It may be prudent to first run on the simpler architectures (e.g.
SAGE) and then try the other architectures afterwards. Failures can occur on
certain architectures (e.g., GIN) when there is insufficient GPU/CPU memory,
and these failures tend to not be graceful (i.e., the job will usually fail to
terminate). These failures should not occur on machines with sufficient memory
--- e.g., 32GB GPU memory and 300GB main memory. We have attempted to set the
default configuration to be more lenient so that it will work with GPUs with
only 16 GB of memory.

## Run distributed multi-GPU experiments

To run the distributed experiments, first make appropriate modifications to `all_dist_benchmark.sh` as detailed above. Then run the following commands:

```bash
cd ${SALIENT_ROOT}/experiments
./run_distributed.sh
```

The data generated can be used to reproduce Figure 5 and Figure 6 in the paper. The full set of distributed experiments will take several hours to run, approximately 12 hours depending on configuration. After the distributed jobs
have completed, the outputs will be present in a directory named
`distributed_job_output`. You may produce a summary table of per-epoch runtime and model accuracy across all experiments by running the helper script `helper_scripts/parse_timings.py` as shown below.

```bash
cd ${SALIENT_ROOT}/experiments
python helper_scripts/parse_timings.py distributed_job_output/
```

Example output for a 16-GPU execution using 8 machines with 2 GPUs each using AWS g5.24xlarge instances is provided below.

```
+------------------+---------+-------------------------+----------------+---------------+---------------+
|      Model       |  System |          Params         | Epoch time (s) |   Valid acc   |    Test acc   |
+------------------+---------+-------------------------+----------------+---------------+---------------+
|       GAT        | SALIENT | Dataset:ogbn-papers100M |  5.712 ± 0.207 | 0.681 ± 0.000 | 0.649 ± 0.000 |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:2      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
|       GIN        |   PyG   | Dataset:ogbn-papers100M | 19.385 ± 0.084 | 0.667 ± N/A   | 0.629 ± N/A   |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:1      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
|       GIN        | SALIENT | Dataset:ogbn-papers100M |  8.028 ± 0.194 | 0.691 ± 0.001 | 0.654 ± 0.001 |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:2      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
|       SAGE       |   PyG   | Dataset:ogbn-papers100M |  5.030 ± 0.135 | 0.672 ± N/A   | 0.644 ± N/A   |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:1      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
|       SAGE       | SALIENT | Dataset:ogbn-papers100M |  2.025 ± 0.224 | 0.678 ± 0.003 | 0.645 ± 0.002 |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:2      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
| SAGEResInception |   PyG   | Dataset:ogbn-papers100M | 12.298 ± 0.634 | 0.685 ± N/A   | 0.648 ± N/A   |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:1      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
| SAGEResInception | SALIENT | Dataset:ogbn-papers100M |  8.108 ± 0.309 | 0.699 ± 0.000 | 0.663 ± 0.004 |
|                  |         |     GPUs-Per-Node:2     |                |               |               |
|                  |         |         Nodes:8         |                |               |               |
|                  |         |      CPU per GPU:20     |                |               |               |
|                  |         |      Num epochs:25      |                |               |               |
|                  |         |       Num trials:2      |                |               |               |
+------------------+---------+-------------------------+----------------+---------------+---------------+
```


# Acknowledgements
This research was sponsored by MIT-IBM Watson AI Lab and in part by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
