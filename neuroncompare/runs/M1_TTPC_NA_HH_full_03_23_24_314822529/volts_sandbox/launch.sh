
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



echo "start-A "`hostname`" task="${job_sh}
echo  'pscratch='${PSCRATCH}
echo  'scratch='${SCRATCH}
echo SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}
echo SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]
then
    echo Not running Slurm Array
    SLURM_ARRAY_TASK_ID=0
fi


srcDir=runs/${model}_${peeling}_${runDate}_${custom}
coreN=${srcDir}/'volts_sand'/${SLURM_ARRAY_JOB_ID}
arrIdx=${SLURM_ARRAY_TASK_ID}
wrkDir=${coreN}-${arrIdx}
echo 'my wrkDir='${wrkDir}
mkdir -p ${wrkDir}

cp -rp volts_sandbox/run_volts ${wrkDir}/run_volts
cd ${wrkDir}/"run_volts"
cd neuron_files/${model}/
nrnivmodl 
cd ../../

export OMP_NUM_THREADS=1



echo 'about to run run_stim_hdf5.py'
echo 'current dir: ' `pwd`
srun -n 64 python run_stim_hdf5.py $arrIdx ${peeling} > SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out

# mv slurm log to final destination - it is alwasy a job-array
echo slurm left at:
pwd
# mv slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out ..

#Y_TASK_ID}.out ..
