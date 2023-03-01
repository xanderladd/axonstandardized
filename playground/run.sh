#!/bin/bash
echo `pwd`
source ./input.txt
input="input.txt"
while IFS= read -r line
do
    IFS="=" read -ra inputs <<< "$line"
    name="${inputs[0]}"
    data="${inputs[1]}"
done < "$input"

true=True # bash script pro lol



if [ ${makeParams} == ${true} ]
  then
    python param_stim_generator/makeParamSet.py
    if [ $? != 0 ];
    then
        echo "failed making params... exiting"
        exit 1
    fi
    echo "Params made"
  fi
  


#making directory for the run
mkdir -p runs/${model}_${peeling}_${runDate}${custom}
wrkDir=runs/${model}_${peeling}_${runDate}${custom}
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'scores'
mkdir -p runs/${model}_${peeling}_${runDate}${custom}/'slurm'


#module load tensorflow/intel-1.12.0-py36
#module load python/3.6-anaconda-4.4

#source ~/neuron-setup.ext

# set sandbox array parameters in score_sandbox and volt_sandbox
# to match those in input.txt
# if num_volts is 0 and num_nodes is 10 will split all stims between 10 nodes 
python modifySandboxArray.py $num_volts $num_nodes
#LOCAL, uses shell script for local imitation
if [ ${makeVolts} == ${true} ]
  then
    sbatch volts_sandbox_setup/sbatch_run.slr
  fi
  
if [ ${makeVoltsGPU} == ${true} ]
  then
    module load cgpu
    sbatch volts_sandbox_setup/voltsGPU.slr
    module unload cgpu
  fi
#sh passive/volts_sandbox_setup/sbatch_local_volts.sh

if  [ $num_volts == 0 ]; then num_volts=400; fi

echo making volts....
#waits until slurm has put enough volts in directory
shopt -s nullglob
STIMFILE="stims/${stim_file}.hdf5"
VOLT_PREFIX="runs/${model}_${peeling}_${runDate}${custom}/volts"
h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
    if [[ "$line" == *"DATASET"* ]]; then
        INPUT="$line"
        fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
        fileName="${VOLT_PREFIX}/${fileName}_volts.hdf5"
        while [ ! -f "${fileName}" ]; do sleep 1; done
        echo found "${fileName}"
    fi
done
shopt -u nullglob

#move the slurm into runs
mv slurm* runs/${model}_${peeling}_${runDate}${custom}/'slurm'



if [ ${makeScores} == ${true} ]
  then
    sbatch score_volts_efficent_sandbox/sbatch_run.slr
  fi


echo making scores....
shopt -s nullglob
STIMFILE="stims/${stim_file}.hdf5"
VOLT_PREFIX="runs/${model}_${peeling}_${runDate}${custom}/scores"
h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
    if [[ "$line" == *"DATASET"* ]]; then
        INPUT="$line"
        fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
        fileName="${VOLT_PREFIX}/${fileName}_scores.hdf5"
        while [ ! -f "${fileName}" ]; do sleep 1; done
        echo found "${fileName}"
    fi
done
shopt -u nullglob

#move slurm into runs
mv slurm* runs/${model}_${peeling}_${runDate}${custom}/'slurm'
mkdir ${wrkDir}/genetic_alg
mkdir ${wrkDir}/genetic_alg/optimization_results/
mkdir ${wrkDir}/genetic_alg/objectives/

if [ ${makeOpt} == ${true} ]
  then
    sbatch analyze_p_bbp_full/analyze_p.slr
  fi

echo waiting on optimzation...
shopt -s nullglob
found=0
target_files=1
while [ $found -ne $target_files ]
do
        found=`ls -lR ${wrkDir}/genetic_alg/optimization_results/*${model}_${peeling}_full.hdf5 | wc -l`
done
echo finished optimzation
shopt -u nullglob

if [ ${makeObj} == ${true} ]
  then
    python analyze_p_bbp_full/analyze_p_multistims.py --model ${model} --peeling ${peeling} \
    --CURRENTDATE ${runDate} --custom ${custom}
  fi

# shopt -s nullglob
# found=0
# target_files=1
# wrkDir=runs/${model}_${peeling}_${runDate}${custom}
# while [ $found -ne $target_files ]
# do
#         found=`ls -lR ${wrkDir}/genetic_alg/objectives/*.hdf5 | wc -l`
# done
# echo finished creating objectives file
# shopt -u nullglob

wrkDir=runs/${model}_${peeling}_${runDate}$_{custom}/genetic_alg
cp -r stims $wrkDir/
cp -r params $wrkDir/
cp param_stim_generator/params_reference/* $wrkDir/params/



if [ ${gaGPU} == ${true} ]
    then
        module purge
        module load cgpu
        sbatch ${wrkDir}/GPU_genetic_alg/BigGaGPU.slr
fi

if [ ${runGA} == ${true} ]
  then
    sbatch ${wrkDir}/neuron_genetic_alg/runGA.slr
  fi



# TODO: what to log when we are done?? later in the road we will a spreadsheet or something
# human interpretable...  maybe put a script doing all the plotting here along with logging this info
# in a meta file
# endTIME =`date +%T`
# python log.py $CURRENTDATE $startTIME $user $model $peeling $nSubZones $nPerSubZone $norm $seed $wrkDir