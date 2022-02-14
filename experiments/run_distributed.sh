#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_ROOT=$SCRIPT_DIR/distributed_job_output

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_sage
mkdir $OUTPUT_ROOT/all_dist_benchmarks_sage

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_sageri
mkdir $OUTPUT_ROOT/all_dist_benchmarks_sageri

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_gin
mkdir $OUTPUT_ROOT/all_dist_benchmarks_gin

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_gat
mkdir $OUTPUT_ROOT/all_dist_benchmarks_gat

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_sage_pyg
mkdir $OUTPUT_ROOT/all_dist_benchmarks_sage_pyg

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_sageri_pyg
mkdir $OUTPUT_ROOT/all_dist_benchmarks_sageri_pyg

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_gin_pyg
mkdir $OUTPUT_ROOT/all_dist_benchmarks_gin_pyg

mkdir -p $SCRIPT_DIR/distributed_job_output/nodelist_all_dist_benchmarks_gat_pyg
mkdir $OUTPUT_ROOT/all_dist_benchmarks_gat_pyg


sbatch all_dist_benchmarks.sh all_dist_benchmarks $SCRIPT_DIR
