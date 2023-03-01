#!bin/bash



modelNum=$1
passive=$2
dest=$3


if [ "$passive" == "True" ]; then

    cp param_stim_generator/allen_generator/results/${modelNum}/target_volts_${modelNum}_passive.hdf5 $dest/target_volts
        
    cp param_stim_generator/allen_generator/results/${modelNum}/stims_${modelNum}_passive.hdf5 $dest/stims
    cp param_stim_generator/allen_generator/results/${modelNum}/stims_${modelNum}_passive.hdf5 stims/stims_${modelNum}_passive.hdf5
    
    cp param_stim_generator/allen_generator/results/${modelNum}/allen${modelNum}_objectives_passive.hdf5 $dest/objectives
    
else
    
    cp param_stim_generator/allen_generator/results/${modelNum}/target_volts_${modelNum}.hdf5 $dest/target_volts
        
    cp param_stim_generator/allen_generator/results/${modelNum}/stims_${modelNum}.hdf5 $dest/stims
    
    cp param_stim_generator/allen_generator/results/${modelNum}/stims_${modelNum}.hdf5 stims/stims_${modelNum}.hdf5
    
    cp param_stim_generator/allen_generator/results/${modelNum}/allen${modelNum}_objectives.hdf5 $dest/objectives
    
fi