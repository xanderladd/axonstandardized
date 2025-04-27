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
mkdir -p runs/${model}_model_${peeling}_${runDate}_${custom}
wrkDir=runs/${model}_model_${peeling}_${runDate}_${custom}
cp input.txt ${wrkDir}/
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'scores'
mkdir -p runs/${model}_model_${peeling}_${runDate}_${custom}/'slurm'
mkdir -p runs/${model}_model_${peeling}_${runDate}_${custom}/'stims'
mkdir -p runs/${model}_model_${peeling}_${runDate}_${custom}/'target_volts'
mkdir -p runs/${model}_model_${peeling}_${runDate}_${custom}/'objectives'



if [ ${makeStims} == ${true} ]
  then
    cd param_stim_generator/allen_generator
    if [ ${passive} == ${true} ];
    then
        python assemble_model_components_nwb.py --model  ${modelNum} --pdf --timestep ${timesteps} --force
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
# sh param_stim_generator/allen_generator/check_files.sh ${modelNum} ${passive}

if [ $? != 0 ];
then
    echo "failed making stims / target volts ... exiting"
    exit 1
fi
echo "stims / target volts made"

# move them up

#TODO: add nwb option here

sh param_stim_generator/allen_generator/move_files.sh ${modelNum} ${passive} runs/${model}_model_${peeling}_${runDate}_${custom}/


# GO INTO DIRECTORY
cd runs/${model}_model_${peeling}_${runDate}_${custom}/


if [ ${preprocess} == ${true} ]; then
    python -m biophys_optimize.scripts.run_preprocessing.py --input_json ./json_files/preprocess_input.json
fi

if [ ${passive} == ${true} ]; then
    python -m biophys_optimize.scripts.passive_fittting.py --input_json ./json_files/passive_input_1.json
python -m biophys_optimize.scripts.passive_fittting.py --input_json ./json_files/passive_input_2.json
python -m biophys_optimize.scripts.passive_fittting.py --input_json ./json_files/passive_input_elec.json
python -m biophys_optimize.scripts.run_consolidate_passive_fitting.py --input_json ./json_files/consolidate_input.json
fi



if [ ${runGA} == ${true} ]
  then
    python -m biophys_optimize.scripts.run_optimize.py --input_json ./json_files/optimize_input.json
  fi


