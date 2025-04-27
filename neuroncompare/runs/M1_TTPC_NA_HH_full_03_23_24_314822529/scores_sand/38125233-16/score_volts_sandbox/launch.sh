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


srcDir=runs/${model}_${peeling}_${runDate}_${custom}
coreN=${srcDir}/'scores_sand'/${SLURM_ARRAY_JOB_ID}
arrIdx=${SLURM_ARRAY_TASK_ID}
wrkDir=${coreN}-${arrIdx}
echo 'my wrkDir='${wrkDir}
mkdir -p ${wrkDir}

dirToRun='score_volts_sandbox'
cp -rp ${dirToRun} ${wrkDir}/
cp input.txt ${wrkDir}/${dirToRun}/
cd ${wrkDir}/${dirToRun}
export OMP_NUM_THREADS=1


srun -n 64 python score_volts_hdf5_efficent_sandbox.py $arrIdx > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out
# fi
echo DONE!E!E!!!!!!
