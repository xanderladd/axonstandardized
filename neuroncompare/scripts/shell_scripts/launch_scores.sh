#!/bin/bash


# change array 1-n, where n is the number of stims
CURRENTDATE=`date +%m_%d_%Y`
input="input.txt"
while IFS= read -r line
do
    IFS='=' read -ra inputs <<< "$line"
    name="${inputs[0]}"
    data="${inputs[1]}"
done < "$input"
source ./input.txt


SLURM_ARRAY_TASK_ID=$1
SLURM_ARRAY_JOB_ID=${SLURM_JOB_ID}

wrkDir=${NEURON_COMPARE_ROOT}/runs/${model}_${peeling}_${runDate}_${custom}
echo 'my wrkDir='${wrkDir}

export OMP_NUM_THREADS=1

arrIdx=$SLURM_ARRAY_TASK_ID

srun -n 64 python -m neuroncompare.src.score_volts_hdf5_efficent_sandbox $arrIdx #> SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out
# fi
echo DONE!E!E!!!!!!
