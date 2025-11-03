
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


SLURM_ARRAY_TASK_ID=0
SLURM_ARRAY_JOB_ID=0
# SLURM_ARRAY_TASK_ID=$1
# SLURM_ARRAY_JOB_ID=${SLURM_JOB_ID}


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


wrkDir=${NEURON_COMPARE_ROOT}/runs/${model}_${peeling}_${runDate}_${custom}
echo 'my wrkDir='${wrkDir}

cp -rp ${NEURON_COMPARE_ROOT}/cell_models ${wrkDir}/cell_models
cd ${wrkDir}/"cell_models"
cd ${model}
rm -rf x86_64
nrnivmodl 
nrnivmodl mechanisms

cd $wrkDir

export OMP_NUM_THREADS=1

arrIdx=0

echo 'about to run run_stim_hdf5.py'
echo 'current dir: ' `pwd`
python -m neuroncompare.src.run_stim_hdf5 $arrIdx ${peeling} #> SLURM${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.out

