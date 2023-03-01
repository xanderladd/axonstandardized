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


srcDir=`pwd`

coreN=${srcDir}/'runs'/${model}_${peeling}_${runDate}_${custom}/'scores_sand'/${SLURM_ARRAY_JOB_ID}
arrIdx=${SLURM_ARRAY_TASK_ID}
wrkDir=${coreN}-${arrIdx}
echo 'my wrkDir='${wrkDir}
mkdir -p ${wrkDir}

dirToRun='score_volts_efficent_sandbox'
cp -rp ${dirToRun} ${wrkDir}/
cp -rp params ${wrkDir}/${dirToRun}/
cp input.txt ${wrkDir}/${dirToRun}/
cd ${wrkDir}/${dirToRun}

# echo inventore at start
# pwd
# ls -l *
export OMP_NUM_THREADS=1



echo $arrIdx ${model} ${peeling}

# if [[ ${model} = 'allen' ]]; then
#     srun --mpi=pmi2 -n 64 -N 1 python score_volts_hdf5_efficent_sandbox_allen.py $arrIdx ${model}  ${peeling} > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out
# elif [[ ${model} = 'compare_allen' ]]; then
#     srun --mpi=pmi2 -n 64 -N 1 python score_volts_hdf5_efficent_sandbox_allen.py $arrIdx ${model}  ${peeling} > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out
# else 
srun --mpi=pmi2 -n 64 -N 1 python score_volts_hdf5_efficent_sandbox.py $arrIdx ${model} ${peeling} > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out
# fi
echo DONE!E!E!!!!!!
