#!/bin/bash
echo `pwd`
module load cray-hdf5

source ./input.txt
true=True 

# set sandbox array parameters in score_sandbox and volt_sandbox
# to match those in input.txt
# if num_volts is 0 and num_nodes is 10 will split all stims between 10 nodes 
python python_scripts/modifySandboxArray.py $num_volts $num_nodes
#LOCAL, uses shell script for local imitation
if [ ${makeVolts} == ${true} ]
  then
    sbatch volts_sandbox/sbatch_run_local.slr
  fi


if  [ $num_volts == 0 ]; then num_volts=400; fi

if [ ${wait4volts} == ${true} ] # if we're making volts, check we've made em all
  then

    echo making volts....
    #waits until slurm has put enough volts in directory

    shopt -s nullglob
    STIMFILE="${data_dir}/stims/${stim_file}.hdf5"
    VOLT_PREFIX="volts"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            fileName="${VOLT_PREFIX}/${fileName}_volts.hdf5"
            # if filename has dt in it, skip
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            if [[ $fileName == *"types"* ]]; then
              continue
            fi
            while [ ! -f "${fileName}" ]; do 
            sleep 1;
            # echo "looking for ${fileName}"
            done
            echo found "${fileName}"
        fi
    done
    shopt -u nullglob
fi
#move the slurm into runs
mv slurm*  slurm/



if [ ${makeScores} == ${true} ]
  then
    sbatch score_volts_sandbox/sbatch_score_local.slr
  fi



if [ ${wait4scores} == ${true} ] # if making scores, check we made em
  then
    echo making scores....


    shopt -s nullglob

    STIMFILE="${data_dir}/stims/${stim_file}.hdf5"
    VOLT_PREFIX="scores"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            fileName="${VOLT_PREFIX}/${fileName}_scores.hdf5"
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            if [[ $fileName == *"types"* ]]; then
              continue
            fi
            while [ ! -f "${fileName}" ]; do 
                sleep 5; 
                echo looking fr "${fileName}"
            done
            echo found "${fileName}"
        fi
    done
    shopt -u nullglob
fi

#move slurm into runs
mv slurm* 'slurm'
wrkDir=`pwd`
mkdir ${wrkDir}/genetic_alg
dirToRun="genetic_alg/neuron_genetic_alg"
cp -rp ../../${dirToRun} ${wrkDir}/genetic_alg/
dirToRun="genetic_alg/*"
cp -p ../../${dirToRun} ${wrkDir}/genetic_alg/

mkdir ${wrkDir}/genetic_alg/optimization_results
mkdir ${wrkDir}/genetic_alg/objectives

cp -r ../analyze_p_bbp_full ${wrkDir}

currDir=`pwd`


if [ ${makeOpt} == ${true} ]
  then
    cd ${wrkDir}
    sbatch analyze_p_bbp_full/analyze_p.slr
    cd ${currDir}

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
    cd ${wrkDir}
    python analyze_p_bbp_full/analyze_p_multistims.py --model ${model} --peeling ${peeling} \
    --CURRENTDATE ${runDate} --custom ${custom}
    cd ${currDir}

  fi

shopt -s nullglob
found=0
target_files=1
wrkDir=`pwd`

while [ $found -ne $target_files ]
do
        found=`ls -lR ${wrkDir}/genetic_alg/objectives/*.hdf5 | wc -l`
done
echo finished creating objectives file
shopt -u nullglob

ga_dir=genetic_alg




if [ ${runGA} == ${true} ]
  then
    cd ${ga_dir}/neuron_genetic_alg/slurm_scripts
    sbatch runGA_allen_perl.slr
    cd -
  fi
  
if [ ${comapre2allen} == ${true} ]
  then
    cd ${ga_dir}
    sbatch compare2allen.slr
    cd -
  fi




echo DONE

# TODO: what to log when we are done?? later in the road we will a spreadsheet or something
# human interpretable...  maybe put a script doing all the plotting here along with logging this info
# in a meta file
# endTIME =`date +%T`
# python log.py $CURRENTDATE $startTIME $user $model $peeling $nSubZones $nPerSubZone $norm $seed $wrkDir