from mpi4py import MPI
import numpy as np
import h5py
import scipy.stats as stat
import os
import sys
import argparse
import copy
from noisyopt import minimizeCompass
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Analyze P Parllel')
parser.add_argument('--model', type=str, required=True, help='specifies model for AnalyzeP')
parser.add_argument('--peeling', type=str, required=True, help='specifies peeling for AnalyzeP')
parser.add_argument('--CURRENTDATE', type=str, required=True, help='specifies date')
parser.add_argument('--custom', type=str, required=False, help='specifies custom postfix')

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.

    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def construct_stim_score_function_list(score_path):
    stim_list = []
    file_list = os.listdir(score_path)
    for file_name in file_list:
        if '.hdf5' in file_name:
            curr_scores = h5py.File(score_path + file_name, 'r')
            stim_list.append(curr_scores['stim_name'][0].decode('ascii'))
            score_function_list = [e.decode('ascii') for e in curr_scores['score_function_names'][:]]
    return sorted(stim_list), sorted(score_function_list)



args = parser.parse_args()
model = args.model
peeling = args.peeling
currentdate = args.CURRENTDATE
custom = args.custom


if custom is not None:
    wrkDir = 'runs/' + model + '_' + peeling + '_' + currentdate + '_' + custom
else:
    wrkDir = 'runs/' + model + '_' + peeling + '_' + currentdate
    
score_path =  wrkDir + '/scores/'
stim_path =  wrkDir + '/stims/'
input_path = os.path.join(wrkDir, 'input.txt')
# modification to run stims 10 Nodes X 30 stims = 300stims
with open(input_path, 'r') as f:
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
                
def is_float(element) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False

optimization_results_path = wrkDir + '/genetic_alg/optimization_results/'

stims_file_path = os.path.join(stim_path, stim_file + '.hdf5')
stim_file = h5py.File(stims_file_path,'r')
available_stims = list(stim_file.keys())
stim_types = list(stim_file['stim_types'])
available_stims = [ stim for stim in available_stims if is_float(stim)]
  
available_stims = {stim:s_type for stim, s_type in zip(available_stims,stim_types)}
stim_choices = ['Long Square','Long Square','Long Square','Short Square', 'Short Square', 'Short Square', 'Noise', 'Noise', 'Noise', 'Square - 0.5ms', 'Square - 2s', 'Square - 2s']
stims_optimal_order = []

for stim_selection in stim_choices:
    available_stims_copy = copy.deepcopy(available_stims)
    for available_stim in available_stims_copy.keys():
        if stim_selection in available_stims_copy[available_stim].decode('ASCII'):
            stims_optimal_order.append(available_stim.encode('ASCII'))
            del available_stims[available_stim]
            break
            

if not os.path.isdir(optimization_results_path):
    os.makedirs(optimization_results_path, exist_ok=True)
params_file_path = 'params/params_' + model + '_' + peeling + '.hdf5'


score_function_list = construct_stim_score_function_list(score_path)
ordered_score_function_list = np.repeat(np.concatenate(score_function_list), len(stim_choices))


opt_result_hdf5 = h5py.File(optimization_results_path+'opt_result_single_stim_' + model + \
    '_' + peeling + '_full.hdf5', 'w')

ordered_score_function_list_as_binary = np.array([np.string_(e) for e in ordered_score_function_list])
    
opt_result_hdf5.create_dataset('ordered_score_function_list', data=ordered_score_function_list_as_binary)

opt_result_hdf5.create_dataset('stims_optimal_order', data=np.array(stims_optimal_order))
opt_result_hdf5.close()
