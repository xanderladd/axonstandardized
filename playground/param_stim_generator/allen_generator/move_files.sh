#!bin/bash



modelNum=$1
passive=$2
dest=$3

source ./input.txt
input="./input.txt"
while IFS= read -r line
do
    IFS="=" read -ra inputs <<< "$line"
    name="${inputs[0]}"
    data="${inputs[1]}"
done < "$input"

if [ "$passive" == "True" ]; then

    cp ${data_dir}/results/${modelNum}/target_volts_${modelNum}_passive.hdf5 $dest/target_volts/
        
    cp ${data_dir}/results/${modelNum}/stims_${modelNum}_passive.hdf5 $dest/stims/
    
    cp ${data_dir}/results/${modelNum}/stims_${modelNum}_passive.hdf5 ${data_dir}/stims/stims_${modelNum}_passive.hdf5
    
    cp ${data_dir}/results/${modelNum}/allen${modelNum}_objectives_passive.hdf5 $dest/objectives/
    
else
    
    cp ${data_dir}/results/${modelNum}/target_volts_${modelNum}.hdf5 $dest/target_volts/
        
    cp ${data_dir}/results/${modelNum}/stims_${modelNum}.hdf5 $dest/stims/
    
    cp ${data_dir}/results/${modelNum}/stims_${modelNum}.hdf5 ${data_dir}/stims/stims_${modelNum}.hdf5
    
    cp ${data_dir}/results/${modelNum}/allen${modelNum}_objectives.hdf5 $dest/objectives/
    
fi