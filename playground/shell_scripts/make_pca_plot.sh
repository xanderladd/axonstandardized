#!/bin/bash


source ./input.txt




for i in {2..77..2}
  do
     python analyze_p_bbp_full/analyze_p_multistims_PCA.py --model ${model} --peeling ${peeling}     --CURRENTDATE ${runDate} --custom ${custom} --n_components $i
done