#!/bin/bash



cd param_stim_generator/allen_generator

modelNum=$1
passive=$2

if [ "$passive" == "True" ]; then
    FILE1=results/${modelNum}/target_volts_${modelNum}_passive.hdf5
    FILE2=results/${modelNum}/stims_${modelNum}_passive.hdf5
else
    FILE1=results/${modelNum}/target_volts_${modelNum}.hdf5
    FILE2=results/${modelNum}/stims_${modelNum}.hdf5
    
fi

if test -f "$FILE1" &&  test -f "$FILE2"; then
    echo "$FILE1 and $FILE2 exists."
    exit 0
else
    echo "$FILE1 and $FILE2 do not exist."
    exit 1
fi