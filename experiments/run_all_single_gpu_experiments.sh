#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

test -f "$SCRIPT_DIR/.ran_config" || echo "[Warning] You have not run the configuration script. You can do so by running 'python experiments/configure_for_environment.py'"
echo ""
echo "Running single GPU experiments on the following datasets: "
for FILE in $SCRIPT_DIR/dataset/ogbn-*; do echo "	-`basename $FILE`"; 
done
echo ""
echo "To run on additional datasets, download them using the downloads_datasets_fast.py script or by running ./performance_breakdown_salient.sh <dataset_name> manually --- which will download the dataset from OGB."
echo ""
echo "Starting experiments in 3 seconds..."
sleep 3
echo ""
for FILE in $SCRIPT_DIR/dataset/ogbn-*; do
        $SCRIPT_DIR/performance_breakdown_salient.sh `basename $FILE`
        $SCRIPT_DIR/performance_breakdown_pyg.sh `basename $FILE`
done
echo ""
echo "Generating a summary of all results in $SCRIPT_DIR/job_output_single_gpu..."
echo ""
python $SCRIPT_DIR/helper_scripts/parse_performance_breakdown.py $SCRIPT_DIR/job_output_single_gpu
