import sys
import os
import h5py
import copy
from neuroncompare.src.utils import read_inputfile
import numpy as np
import pandas as pd
##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]

# modification to run stims 10 Nodes X 30 stims = 300stims
inputs = read_inputfile('../../input.txt')
inputs['param_opt_inds'] = np.array(inputs['params'].split(','), dtype=int)-1
if 'compare' in inputs['model'] and 'bbp' in inputs['model']:
    print("******TURNING E PAS NEGATIVE HACK*******")
    if "2" in inputs['param_opt_inds']:
        neg_idxs = [1]
    
    neuron_path = 'cell_models/compare_bbp'
    run_file = 'cell_models/compare_bbp/run_model_cori.hoc'
elif inputs['model'] == 'bbp':
    neuron_path = 'cell_models/bbp'
    run_file = 'cell_models/bbp/run_model_cori.hoc'
elif inputs['model'] == 'allen':
    neuron_path = 'cell_models/allen'
    run_file = None
elif inputs['model'] == 'M1_TTPC_NA_HH':
    neuron_path = 'cell_models/M1_TTPC_NA_HH'
    run_file = None
    print("******TURNING E PAS NEGATIVE HACK*******")
    if "20" in inputs['param_opt_inds']:
        neg_idxs = [23, 24]

    
os.chdir(neuron_path)	
from neuron import h	
os.chdir('../../')



params_file_path = os.path.join(inputs['base_dir'], f"runs/{inputs['model']}_{peeling}_{inputs['runDate']}_{inputs['custom']}", 'params/params_' + inputs['model'] + '_' + peeling+ '.hdf5')
stims_file_path = os.path.join(inputs['base_dir'], f"runs/{inputs['model']}_{peeling}_{inputs['runDate']}_{inputs['custom']}",'stims',  inputs['stim_file'] + '.hdf5')
# Number of timesteps for the output volt.
# inputs['ntimestep'] = 10000

# Output destination.
volts_path = '../../volts/'

# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
params_hdf5 = h5py.File(params_file_path, 'r')
params_name_list = list(params_hdf5.keys())

stims_name_list = sorted(list(stims_hdf5.keys()))
stims_name_list = [elem for elem in stims_name_list if "dt" not in elem]

num_stims_to_run = 1
i=int(sys.argv[1])
if i == 0 and inputs['num_nodes'] == 1:
    curr_stim_name_list = stims_name_list
elif inputs['num_nodes'] > 1 and inputs['num_volts'] == 0:
    num_stims_to_run = math.ceil(len(stims_name_list) / inputs['num_nodes'])
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]
    print(len(curr_stim_name_list))
else:
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]



curr_stim_name_list.reverse()
# why was this here???
# curr_stim_name_list = curr_stim_name_list[:1]
ntimestep = int(inputs['timesteps'])
print(inputs['timesteps'])
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

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
data_dir = inputs['data_dir']

stims_path = data_dir + '/stims/' + inputs['stim_file'] + '.hdf5'



usePrev = inputs['usePrevParams']
model_num = inputs['modelNum']
passive = eval(inputs['passive'])
orig_name = "orig_" + peeling
orig_params = h5py.File('../../params/params_' + model + '_' + peeling + '.hdf5', 'r')[orig_name][0]


if 'log_transform_params' in inputs:
    log_transform_params = bool(inputs['log_transform_params'])
else:
    log_transform_params = False
if usePrev == "True":
    params_csv = data_dir + '/params/params_' + inputs['model'] + '_' + peeling + '_prev.csv'
else:
    params_csv = data_dir + '/params/params_' + inputs['model'] + '_' + peeling + '.csv'
    
negative_param_inds = []
for idx, param in enumerate(pd.read_csv(params_csv).to_dict(orient='records')):
    if 'e_pas' in param['Param name']:
        negative_param_inds.append(idx)
        
