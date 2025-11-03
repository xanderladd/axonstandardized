#!/bin/bash


nnodes=`srun -n 1 echo $SLURM_NNODES`
nnodes=80
source ./input.txt
n_batches=$((num_volts / nnodes))
for n_batch in $(eval echo "{0..$n_batches}"); do
    start_ind=$((n_batch * nnodes))
    end_ind=$(((n_batch+1) * nnodes))
    for i in  $(eval echo "{$start_ind..$end_ind}")}; do
        echo "$i"  #| tr -dc '0-9'  #&
    done
    #wait
done
