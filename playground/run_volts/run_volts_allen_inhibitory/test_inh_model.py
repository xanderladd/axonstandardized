import numpy as np
import h5py
import os, sys
os.chdir('neuron_files/allen')
from neuron import h
os.chdir('../../')
import math
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages

##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]

# modification to run stims 10 Nodes X 30 stims = 300stims
with open('./input.txt', 'r') as f:
    for line in f:
        if "=" in line:
            name, value = line.split("=")
            if name == "num_nodes":
                num_nodes = int(value)
            if name == "num_volts":
                num_volts = int(value)
            if name == "stim_file":
                stim_file = value.replace('\n','')
            if name == "model":
                model = value.replace('\n','')
            if name== "params":
                param_opt_inds = value.split(",")
            if name == "timesteps":
                ntimestep = int(value)
           
         
if model == 'allen':
    run_file = 'neuron_files/allen/run_model_cori.hoc'
elif model == 'bbp':
    run_file = 'neuron_files/bbp/run_model_cori.hoc'
else:
    raise NotImplementedError

stims_file_path = './' + stim_file + '.hdf5'
# Number of timesteps for the output volt.
# ntimestep = 10000


# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
neg_index =None

stims_name_list = sorted(list(stims_hdf5.keys()))
stims_name_list = [elem for elem in stims_name_list if "dt" not in elem]
curr_stim_name_list = stims_name_list
curr_stim_name_list.reverse()
print(ntimestep)
print("stim name list", curr_stim_name_list)
curr_stim_name_list_copy = copy.deepcopy(curr_stim_name_list)


import pandas as pd
df = pd.read_csv('./params_allen_full_inh.csv')
param_set = df['Base value'].values

    



def run_model(run_file, param_set, stim_name, ntimestep):
    
    
    h.load_file(run_file)
    total_params_num = len(param_set)
    dt = stims_hdf5[stim_name+'_dt']
    stim_data = stims_hdf5[stim_name][:]
    curr_ntimestep = len(stim_data)
    timestamps = np.array([dt for i in range(curr_ntimestep)])
    h.curr_stim = h.Vector().from_python(stim_data)
    h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
    h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
   
    h.ntimestep = curr_ntimestep
    h.runStim()
    out = h.vecOut.to_python()
    
    return np.array(out)

def plot_volts(volts, pdf, name):
    fig = plt.figure()
    plt.title(name)
    plt.plot(volts)
    pdf.savefig(fig)
    plt.close(fig)
    
    
pdf =  PdfPages('inh_model_responses.pdf')
# curr_stim_name_list = curr_stim_name_list[15]#[::20]
for stim_ind in range(len(curr_stim_name_list)):
    curr_stim_name = curr_stim_name_list[stim_ind]
    volts = run_model(run_file,param_set, curr_stim_name, ntimestep) 
    plot_volts(volts, pdf, curr_stim_name)
    
pdf.close()
