
source ./input.txt
input="input.txt"
while IFS= read -r line
do
    IFS="=" read -ra inputs <<< "$line"
    name="${inputs[0]}"
    data="${inputs[1]}"
done < "$input"
true=True # ???


#making directory for the run
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}
wrkDir=runs/${model}_${peeling}_${runDate}_${custom}
cp input.txt ${wrkDir}/
mkdir -p ${wrkDir}/'volts'
mkdir -p ${wrkDir}/'scores'
mkdir -p ${wrkDir}/'objectives'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'slurm'
mkdir -p runs/${model}_${peeling}_${runDate}_${custom}/'stims'


if [ ${ingestCell} == ${true} ]
  then
    python -m neuroncompare.src.cell_ingest pull  --cell_id ${modelNum}  
fi

if [ ${makeStims} == ${true} ]
  then
    python -m neuroncompare.src.cell_ingest assemble  --model  ${modelNum}  --pdf --force --timestep ${timesteps}
fi

sh scripts/shell_scripts/check_files.sh ${modelNum} ${passive} ${data_dir}


if [ $? != 0 ];
then
    echo "failed making stims / target volts ... exiting"
    exit 1
fi
echo "stims / target volts made"

# move them up
python -m neuroncompare.src.cell_ingest copy_to_run --model ${modelNum} --passive ${passive} --dest runs/${model}_${peeling}_${runDate}_${custom}


if [ ${makeParams} == ${true} ]
    then
    python -m neuroncompare.src.make_params
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
cp run_remainder.sh ${wrkDir}

# WORKING END


# config stuff
launch_cmd="sh"
launch_ext=".sh"
launch_dir="scripts/shell_scripts"
prefix="shell"
if [ "${sbatch}" = "${true}" ]; then
    launch_cmd="sbatch"
    launch_ext=".slr"
    launch_dir="scripts/slurm"
    prefix='sbatch'
    module load cray-hdf5
    # set sandbox array parameters in score_sandbox and volt_sandbox
    # to match those in input.txt
    # if num_volts is 0 and num_nodes is 10 will split all stims between 10 nodes 
    python -m neuroncompare.src.modifySandboxArray $num_volts $num_nodes
elif [ "${srun}" = "${true}" ]; then
    # srun uses shell scripts but then uses srun versions
    prefix='srun'
    module load cray-hdf5

fi

# actually make the volts
if [ ${makeVolts} == ${true} ]
  then
    echo "$launch_cmd ${launch_dir}/${prefix}_volts${launch_ext}"
fi

  

if  [ $num_volts == 0 ]; then num_volts=400; fi


if [ ${wait4volts} == ${true} ] # if we're making volts, check we've made em all
  then

    echo making volts....
    #waits until slurm has put enough volts in directory

    shopt -s nullglob
    STIMFILE="${data_dir}/stims/${stim_file}.hdf5"
    VOLT_PREFIX="runs/${model}_${peeling}_${runDate}_${custom}/volts"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            postfix=${fileName}
            
            fileName="${VOLT_PREFIX}/${fileName}_volts.hdf5"
            # if filename has dt in it, skip
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            
            if [[ ! $postfix =~ [0-9] ]]; then
              
               continue
           fi
           

            while [ ! -f "${fileName}" ]; do 
                sleep 5; 
                echo looking fr "${fileName}"
            done
            # while [ ! -f "${fileName}" ]; do sleep 1; done
            echo found "${fileName}"
        fi
    done
    shopt -u nullglob
fi
#move the slurm into runs
mv slurm* runs/${model}_${peeling}_${runDate}_${custom}/'slurm'




if [ ${makeScores} == ${true} ]
  then
    # sbatch score_volts_sandbox/sbatch_score.slr
    echo "$launch_cmd ${launch_dir}/${prefix}_scores${launch_ext}"

  fi



if [ ${wait4scores} == ${true} ] # if making scores, check we made em
  then
    echo making scores....


    shopt -s nullglob

    STIMFILE="${data_dir}/stims/${stim_file}.hdf5"
    VOLT_PREFIX="runs/${model}_${peeling}_${runDate}_${custom}/scores"
    h5dump --header $STIMFILE | head -n $(expr 2 + ${num_volts} \* 4) | while read line; do
        if [[ "$line" == *"DATASET"* ]]; then
            INPUT="$line"
            fileName=$(echo "${INPUT}" | cut -d '"' -f 2)
            postfix=${fileName}
            
            fileName="${VOLT_PREFIX}/${fileName}_volts.hdf5"
            # if filename has dt in it, skip
            if [[ $fileName == *"dt"* ]]; then
              continue
            fi
            
            if [[ ! $postfix =~ [0-9] ]]; then
              
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
# dirToRun="genetic_alg/GPU_genetic_alg"
# cp -rp ${dirToRun} ${wrkDir}/genetic_alg/
dirToRun="genetic_alg/*"
cp -p ${dirToRun} ${wrkDir}/genetic_alg/
mkdir ${wrkDir}/genetic_alg/optimization_results
mkdir ${wrkDir}/genetic_alg/objectives

cp -r analyze_p_bbp_full ${wrkDir}

currDir=`pwd`

exit

if [ ${makeOpt} == ${true} ]
  then
    cd ${wrkDir}
    $launch_cmd analyze_p_bbp_full/analyze_p.slr
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
    python -m neuroncompare.src.analyze_p_multistims --model ${model} --peeling ${peeling} \
    --CURRENTDATE ${runDate} --custom ${custom}
    cd ${currDir}

  fi

shopt -s nullglob
found=0
target_files=1
wrkDir=runs/${model}_${peeling}_${runDate}_${custom}
while [ $found -ne $target_files ]
do
        found=`ls -lR ${wrkDir}/genetic_alg/objectives/*.hdf5 | wc -l`
done
echo finished creating objectives file
shopt -u nullglob

ga_dir=runs/${model}_${peeling}_${runDate}_${custom}/genetic_alg




if [ ${runGA} == ${true} ]
  then
    cd ${ga_dir}/neuron_genetic_alg/slurm_scripts
    $launch_cmd runGA_allen_perl.slr
    cd -
  fi
  
if [ ${comapre2allen} == ${true} ]
  then
    cd ${ga_dir}
    $launch_cmd compare2allen.slr
    cd -
  fi

echo DONE
  



