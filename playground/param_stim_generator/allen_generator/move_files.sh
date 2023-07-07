#!bin/bash



modelNum=$1
passive=$2
dest=$3


if [ "$passive" == "True" ]; then

    cp ../../axonstandardized_data/results/${modelNum}/target_volts_${modelNum}_passive.hdf5 $dest/target_volts/
        
    cp ../../axonstandardized_data/results/${modelNum}/stims_${modelNum}_passive.hdf5 $dest/stims/
    cp ../../axonstandardized_data/results/${modelNum}/stims_${modelNum}_passive.hdf5 ../../axonstandardized_data/stims/stims_${modelNum}_passive.hdf5
    
    cp ../../axonstandardized_data/results/${modelNum}/allen${modelNum}_objectives_passive.hdf5 $dest/objectives/
    
else
    
    cp ../../axonstandardized_data/results/${modelNum}/target_volts_${modelNum}.hdf5 $dest/target_volts/
        
    cp ../../axonstandardized_data/results/${modelNum}/stims_${modelNum}.hdf5 $dest/stims/
    
    cp ../../axonstandardized_data/results/${modelNum}/stims_${modelNum}.hdf5 ../../axonstandardized_data/stims/stims_${modelNum}.hdf5
    
    cp ../../axonstandardized_data/results/${modelNum}/allen${modelNum}_objectives.hdf5 $dest/objectives/
    
fi