#!/bin/bash

python set_compare_v_init.py

python pull_allen_model.py
cd allen_model
nrnivmodl modfiles

cd ../

python prepare_allen_model.py

srun -n 200 python run_compare.py &
cd allen_model;  python -m allensdk.model.biophysical.runner manifest.json &

cd ../

python draw_comparison.py

srun -n 200 python compare_efel.py
python combine_efel.py

