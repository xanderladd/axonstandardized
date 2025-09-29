#!/bin/bash

#SBATCH --qos=regular
#SBATCH --time=16:00:00
#SBATCH --nodes=64
#SBATCH --constraint=haswell
#SBATCH --mail-user=zladd@berkeley.edu
#SBATCH --mail-type=ALL

cd ../
CURRENTDATE=`date +%m_%d_%Y`
startTIME=`date +%T`
custom=''
source ../../input.txt
echo running GA
echo OFFSPRING_SIZE is ${OFFSPRING_SIZE}
echo for ${MAX_NGEN} generations


seed=1998 # not used

#seed=1132 # not used
#seed=1178 # used
seed=$((10000 + $RANDOM % 100000))
export BLUEPYOPT_SEED=${seed}




echo seed: ${seed}
export OMP_NUM_THREADS=1

srun -n 1000 python optimize_parameters_genetic_alg.py \
    -vv                                \
    --compile                          \
    --offspring_size=4000              \
    --max_ngen=500                   \
    --seed=${seed}                     \
    --checkpoint ckpts/${seed}_ckpt      \
    --start  > GA_out${seed}.log  
    

    


