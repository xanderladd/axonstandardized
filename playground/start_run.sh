#!/bin/bash
echo `pwd`
module load cray-hdf5

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



if [ ${ingestCell} == ${true} ]
  then
    cd param_stim_generator/allen_generator
    python cell_ingest.py --cell_id ${modelNum}
    python exp_data_sample.py --cell_id ${modelNum}
    cd ../../
fi

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
# sh param_stim_generator/allen_generator/move_files.sh ${modelNum} ${passive} runs/${model}_${peeling}_${runDate}_${custom}/


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
  

cp -rp ${data_dir}/params ${wrkDir}/${dirToRun}/
cp -rp ${data_dir}/stims ${wrkDir}/${dirToRun}/
cp -rp ${data_dir}/target_volts ${wrkDir}/${dirToRun}/
cp -rp  python_scripts ${wrkDir}/${dirToRun}/

cp *.py ${wrkDir}
cp -r volts_sandbox ${wrkDir}
cp -r score_volts_sandbox ${wrkDir}
cp -r analyze_p_bbp_full ${wrkDir}

cp run_remainder.sh ${wrkDir}

