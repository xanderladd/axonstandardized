
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


coreN=${srcDir}/'runs'/${model}_${peeling}_${runDate}_${custom}/'volts_sand'/${SLURM_ARRAY_JOB_ID}
arrIdx=${SLURM_ARRAY_TASK_ID}
wrkDir=${coreN}-${arrIdx}
echo 'my wrkDir='${wrkDir}
mkdir -p ${wrkDir}

dirToRun="run_volts/run_volts_${model}${run_volts_extension}"
cp -rp ${dirToRun} ${wrkDir}/
cd ${wrkDir}/"run_volts_${model}${run_volts_extension}"
# nrnivmodl

# echo inventore at start
# pwd
# ls -l *

export OMP_NUM_THREADS=1

srun --mpi=pmi2 -n 64 -N 1 python run_stim_hdf5.py $arrIdx ${peeling} > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out

# mv slurm log to final destination - it is alwasy a job-array
echo slurm left at:
pwd
# mv slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ..

#Y_TASK_ID}.out ..
