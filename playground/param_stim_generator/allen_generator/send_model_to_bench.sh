#!bin/bash

cp results/$1/allen$1_objectives2.hdf5 ../benchmarking/GPU_genetic_alg/python/objectives/allen$1_objectives.hdf5
cp results/$1/target_volts_$1.hdf5 ../benchmarking/GPU_genetic_alg/python/target_volts
cp results/$1/stims_$1.hdf5 ../benchmarking/GPU_genetic_alg/stims
