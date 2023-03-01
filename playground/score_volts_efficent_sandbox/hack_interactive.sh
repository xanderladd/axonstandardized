#!/bin/bash


#!/bin/bash


nnodes=`srun -n 1 echo $SLURM_NNODES`
source ./input.txt
n_batches=$((num_volts / nnodes))
for n_batch in $(eval echo "{0..$n_batches}"); do
    start_ind=$((n_batch * nnodes))
    end_ind=$(((n_batch+1) * nnodes))
    for i in  $(eval echo "{$start_ind..$end_ind}")}; do
        i=$(echo "$i" | tr -dc '0-9')
        sh score_volts_efficent_sandbox/launch.sh $i &
    done
    wait
done



# for i in {38..57}; do
#     sh score_volts_efficent_sandbox/launch.sh  $i &
# done
# wait


# for i in {57..76}; do
#     sh score_volts_efficent_sandbox/launch.sh  $i &
# done


