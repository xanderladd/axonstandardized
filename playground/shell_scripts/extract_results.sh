#!/bin/bash


#### LATER: MAKE LIST OF RESULTS TO EXTRACT EASY ####

mkdir ../pipeline_extraction
mkdir ../pipeline_extraction/genetic_alg

mkdir ../pipeline_extraction/genetic_alg/neuron_genetic_alg/
cd genetic_alg/neuron_genetic_alg/
cp ../*.py ../../../pipeline_extraction/genetic_alg/neuron_genetic_alg
shopt -s extglob
cp -r !(scores) ../../../pipeline_extraction/genetic_alg/neuron_genetic_alg/
cd ../../

touch ../pipeline_extraction/.gitignore
echo "*" >> ../pipeline_extraction/.gitignore
cp -r stims/ ../pipeline_extraction/


cp plot_runs.ipynb ../pipeline_extraction
mkdir ../pipeline_extraction/runs/

mkdir ../pipeline_extraction/runs/bbp_passive_02_03_2021base2
mkdir ../pipeline_extraction/runs/bbp_potassium_02_11_2021full
mkdir ../pipeline_extraction/runs/bbp_calcium_02_14_2021full
mkdir ../pipeline_extraction/runs/bbp_sodium_02_18_2021full

mkdir ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg
mkdir ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg
mkdir ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg
mkdir ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg

mkdir ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg/GPU_genetic_alg
mkdir ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg/GPU_genetic_alg
mkdir ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg/GPU_genetic_alg
mkdir ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg/GPU_genetic_alg


cp -r runs/bbp_passive_02_03_2021base2/genetic_alg/GPU_genetic_alg/python  ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg/GPU_genetic_alg/
cp -r runs/bbp_potassium_02_11_2021full/genetic_alg/GPU_genetic_alg/python  ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg/GPU_genetic_alg/
cp -r runs/bbp_calcium_02_14_2021full/genetic_alg/GPU_genetic_alg/python  ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg/GPU_genetic_alg/
cp -r runs/bbp_sodium_02_18_2021full/genetic_alg/GPU_genetic_alg/python  ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg/GPU_genetic_alg/

cp -r runs/bbp_passive_02_03_2021base2/genetic_alg/objectives ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg
cp -r runs/bbp_potassium_02_11_2021full/genetic_alg/objectives ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg
cp -r runs/bbp_calcium_02_14_2021full/genetic_alg/objectives ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg
cp -r runs/bbp_sodium_02_18_2021full/genetic_alg/objectives ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg

cp -r runs/bbp_passive_02_03_2021base2/genetic_alg/params ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg
cp -r runs/bbp_potassium_02_11_2021full/genetic_alg/params ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg
cp -r runs/bbp_calcium_02_14_2021full/genetic_alg/params ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg
cp -r runs/bbp_sodium_02_18_2021full/genetic_alg/params ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg

cp -r runs/bbp_passive_02_03_2021base2/scores ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/

cp -r runs/bbp_potassium_02_11_2021full/scores ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/

cp -r runs/bbp_calcium_02_14_2021full/scores ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/

cp -r runs/bbp_sodium_02_18_2021full/scores ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/

mkdir ../pipeline_extraction/plots




cp -r ../pipeline_extraction/genetic_alg/neuron_genetic_alg/ ../pipeline_extraction/runs/bbp_passive_02_03_2021base2/genetic_alg/

cp -r ../pipeline_extraction/genetic_alg/neuron_genetic_alg/ ../pipeline_extraction/runs/bbp_potassium_02_11_2021full/genetic_alg/

cp -r ../pipeline_extraction/genetic_alg/neuron_genetic_alg/ ../pipeline_extraction/runs/bbp_calcium_02_14_2021full/genetic_alg/

cp -r ../pipeline_extraction/genetic_alg/neuron_genetic_alg/ ../pipeline_extraction/runs/bbp_sodium_02_18_2021full/genetic_alg/
# cp -r runs/bbp_potassium_02_11_2021full  ../pipeline_extraction/runs/
# cp -r runs/bbp_calcium_02_14_2021full  ../pipeline_extraction/runs/
# cp -r runs/bbp_sodium_02_18_2021full  ../pipeline_extraction/runs/
