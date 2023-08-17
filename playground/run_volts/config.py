import sys
import os
import h5py
import copy
# changed 12/18 ^^


##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]

# modification to run stims 10 Nodes X 30 stims = 300stims
dt = None
with open('../../../input.txt', 'r') as f:
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
            if name == "dt":
                dt = float(value)
                

           
         
if 'compare' in model and 'bbp' in model:
    print("******TURNING E PAS NEGATIVE HACK*******")
    if "2" in param_opt_inds:
        neg_index = 1
    neuron_path = 'neuron_files/compare_bbp'
    run_file = 'neuron_files/compare_bbp/run_model_cori.hoc'
elif model == 'bbp':
    neuron_path = 'neuron_files/bbp'
    run_file = 'neuron_files/bbp/run_model_cori.hoc'
elif model == 'allen':
    neuron_path = 'neuron_files/allen'
    run_file = None
elif model == 'M1_TTPC_NA_HH':
    neuron_path = 'neuron_files/M1_TTPC_NA_HH'
    run_file = None

    
os.chdir(neuron_path)	
from neuron import h	
os.chdir('../../')



params_file_path = '../../../params/params_' + model + '_' + peeling+ '.hdf5'
stims_file_path = '../../../stims/' + stim_file + '.hdf5'
# Number of timesteps for the output volt.
# ntimestep = 10000

# Output destination.
volts_path = '../../../volts/'

# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
params_hdf5 = h5py.File(params_file_path, 'r')
params_name_list = list(params_hdf5.keys())
neg_index = None

stims_name_list = sorted(list(stims_hdf5.keys()))
stims_name_list = [elem for elem in stims_name_list if "dt" not in elem]

num_stims_to_run = 1
i=int(sys.argv[1])
if i == 0 and num_nodes == 1:
    curr_stim_name_list = stims_name_list
elif num_nodes > 1 and num_volts == 0:
    num_stims_to_run = math.ceil(len(stims_name_list) / num_nodes)
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]
    print(len(curr_stim_name_list))
else:
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]



curr_stim_name_list.reverse()
curr_stim_name_list = curr_stim_name_list[:1]
print(ntimestep)
print("params names list",params_name_list)
print("stim name list", curr_stim_name_list)

ignore_stim_names = ['stim_types']

curr_stim_name_list_copy = copy.deepcopy(curr_stim_name_list)
for curr_stim_name in curr_stim_name_list_copy:
    filepath = volts_path+curr_stim_name+'_volts.hdf5'
    if os.path.isfile(filepath) or curr_stim_name in ignore_stim_names:
        curr_stim_name_list.remove(curr_stim_name)
        
if len(curr_stim_name_list) < 1:
    print("STIM NAME LIST is EMPTY CUS ITS COMPLETE, EXITING")
    exit()


pin_set_size = None
pdx_set_size = None