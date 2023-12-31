

# Notebook for running currentscape.
# Run your model first, then run currentscape.
# One sticking point I found is that if you don't specify your current_names, it will automatically label them with their currents in sim_config > 'currents'.
# However, the colors may not line up between 2 conditions (Eg. WT vs HET) since they will be ordered based on percentage of current contribution.
# Some more info https://currentscape.readthedocs.io/en/latest/tutorial.html
import pickle
import pandas as pd
from NeuronModelClass import NeuronModel
# from NrnHelper import *
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np
import os
from hoc_utils import decode_list, retrieve_dt
import config
import hoc_utils
import h5py
import math

from NeuronModelClass import NeuronModel
os.chdir("neuron_files/M1_TTPC_NA_HH")
from neuron import h
os.chdir("../../")



def read_best_indvs(path, gen=-1):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data[gen]

def unlog_params(bases, best_full, config):
    if config.log_transform_params:
        for i in range(len(best_full)):
            if bases[i] > config.base_thresh:
                best_full[i] = math.pow(bases[i], best_full[i])
    return best_full

params_path = '/pscratch/sd/z/zladd/axonstandardized/playground/runs/M1_TTPC_NA_HH_full_07_25_23_314831019/params/params_M1_TTPC_NA_HH_full.csv'

GA_result_path = '/pscratch/sd/z/zladd/axonstandardized/playground/runs/M1_TTPC_NA_HH_full_07_25_23_314831019/genetic_alg/neuron_genetic_alg_passive/best_indv_logs23075/best_indvs_gen_143.pkl'
model_num = str(config.model_num)
stims_path = f"../../stims/stims_{model_num}.hdf5"
target_volts_path = f'../../target_volts/allen_data_target_volts_{model_num}.hdf5'

#read stims and target volts
stims = h5py.File(stims_path,"r")
target_volts = h5py.File(target_volts_path, 'r')

test_stim = stims['34'][:]
test_dt = stims['34_dt'][:]


best_indv = read_best_indvs(GA_result_path)
df = pd.read_csv(params_path, skipinitialspace=True, usecols=['Param name','Base value', 'Lower bound', 'Upper bound'])
lbs = np.array(df['Lower bound'])
ubs = np.array(df['Upper bound'])
base_full = np.array(df['Base value'])
names = df['Param name']
bases, orig_params, mins, maxs = hoc_utils.log_params(ubs, lbs, base_full)

best_full = list(base_full)
opt_ind = np.arange(21)

for i in range(len(opt_ind)):
    best_full[opt_ind[i]] = best_indv[i]
    
best_full = unlog_params(bases, best_full, config)
best_full = np.abs(best_full)
best_full[config.negative_param_inds] = - best_full[config.negative_param_inds]


print("\n\n")
count = 0
for best, lb, ub, name in zip(best_full, lbs, ubs, names):
    if count in opt_ind:
        print(count, "(optimized)", "name :", name, " | best: ", round(best,9) , " | lb: ", round(lb,8), " | ub: ", round(ub,8))

    else:
        print(count, "name :", name, " | best: ", round(best,9) , " | lb: ", round(lb,8), " | ub: ", round(ub,8))
    print("------------------------------------------------------------")
    count += 1


## I normally create an instance of my model and do the plotting in a separate file/jupyter notebook but as an example...

model = NeuronModel(mod_dir = './neuron_files/M1_TTPC_NA_HH/')

model.update_params(best_full)

# model.make_currentscape_plot(amp=0.5, time1=0,time2=100,stim_start=40, sweep_len=100)
model.make_currentscape_plot_stim(test_stim, test_dt, start_Vm=-76.6875 )
plt.show()
