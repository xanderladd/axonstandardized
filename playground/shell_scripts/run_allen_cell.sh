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


#making directory for the run
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}
wrkDir=runs/${model}_${peeling}_${runDate}_${custom}
cp input.txt ${wrkDir}/
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'scores'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'slurm'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'stims'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'target_volts'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'objectives'



if [ ${makeStims} == ${true} ]
  then
    cd param_stim_generator/allen_generator
    if [ ${passive} == ${true} ];
    then
        echo "passive version"
        python assemble_model_components.py --model  ${modelNum} --pdf --passive --timestep ${timesteps} --force

    else
        python assemble_model_components.py --model  ${modelNum} --pdf --force --timestep ${timesteps}
    fi
    
    if [ $? != 0 ];
    then
        echo "failed making stims / target volts ... exiting"
        exit 1
    fi
    echo "stims / target volts made"
    cd ../../
fi


 
# check that files are seteup correctly
sh param_stim_generator/allen_generator/check_files.sh ${modelNum} ${passive}

if [ $? != 0 ];
then
    echo "failed making stims / target volts ... exiting"
    exit 1
fi
echo "stims / target volts made"

# move them up
sh param_stim_generator/allen_generator/move_files.sh ${modelNum} ${passive} runs/${model}_${peeling}_${runDate}_${custom}/


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


if [ ${wait4volts} == ${true} ] # if we're making volts, check we've made em all
  then

    echo making volts....
    #waits until slurm has put enough volts in directory

    shopt -s nullglob
    STIMFILE="stims/${stim_file}.hdf5"
    VOLT_PREFIX="runs/${model}_${peeling}_${runDate}_${custom}/volts"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            fileName="${VOLT_PREFIX}/${fileName}_volts.hdf5"
            # if filename has dt in it, skip
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            while [ ! -f "${fileName}" ]; do sleep 1; done
            echo found "${fileName}"
        fi
    done
    shopt -u nullglob
fi
#move the slurm into runs
mv slurm* runs/${model}_${peeling}_${runDate}_${custom}/'slurm'




if [ ${makeScores} == ${true} ]
  then
    sbatch score_volts_efficent_sandbox/sbatch_score.slr
  fi



if [ ${wait4scores} == ${true} ] # if making scores, check we made em
  then
    echo making scores....


    shopt -s nullglob

    STIMFILE="stims/${stim_file}.hdf5"
    VOLT_PREFIX="runs/${model}_${peeling}_${runDate}_${custom}/scores"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            fileName="${VOLT_PREFIX}/${fileName}_scores.hdf5"
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            while [ ! -f "${fileName}" ]; do 
                sleep 5; 
                # echo looking fr "${fileName}"
            done
            echo found "${fileName}"
        fi
    done
    shopt -u nullglob
fi

#move slurm into runs
mv slurm* runs/${model}_${peeling}_${runDate}_${custom}/'slurm'
mkdir ${wrkDir}/genetic_alg
dirToRun="genetic_alg/neuron_genetic_alg"
cp -rp ${dirToRun} ${wrkDir}/genetic_alg/
dirToRun="genetic_alg/GPU_genetic_alg"
cp -rp ${dirToRun} ${wrkDir}/genetic_alg/
dirToRun="genetic_alg/*"
cp -p ${dirToRun} ${wrkDir}/genetic_alg/

mkdir ${wrkDir}/genetic_alg/optimization_results/
mkdir ${wrkDir}/genetic_alg/objectives/



if [ ${makeOpt} == ${true} ]
  then
    sbatch analyze_p_bbp_full/analyze_p.slr
  

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
    fi

if [ ${makeObj} == ${true} ]
  then
    python analyze_p_bbp_full/analyze_p_multistims.py --model ${model} --peeling ${peeling} \
    --CURRENTDATE ${runDate} --custom ${custom}
  fi

# shopt -s nullglob
# found=0
# target_files=1
# wrkDir=runs/${model}_${peeling}_${runDate}_${custom}
# while [ $found -ne $target_files ]
# do
#         found=`ls -lR ${wrkDir}/genetic_alg/objectives/*.hdf5 | wc -l`
# done
# echo finished creating objectives file
# shopt -u nullglob

wrkDir=runs/${model}_${peeling}_${runDate}_${custom}/genetic_alg
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


echo DONE

# TODO: what to log when we are done?? later in the road we will a spreadsheet or something
# human interpretable...  maybe put a script doing all the plotting here along with logging this info
# in a meta file
# endTIME =`date +%T`
# python log.py $CURRENTDATE $startTIME $user $model $peeling $nSubZones $nPerSubZone $norm $seed $wrkDir