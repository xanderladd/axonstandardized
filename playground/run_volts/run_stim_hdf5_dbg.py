from mpi4py import MPI
import numpy as np
import h5py
import os, sys
os.chdir('neuron_files/modfiles')
from neuron import h
os.chdir('../../')
import math
from allensdk.model.biophysical.utils import create_utils
import allensdk.model.biophysical.runner as runner
import time

import copy
##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]

# modification to run stims 10 Nodes X 30 stims = 300stims
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
           
         
if model == 'allen':
    run_file = 'neuron_files/allen/run_model_cori.hoc'
elif model == 'bbp':
    run_file = 'neuron_files/bbp/run_model_cori.hoc'
else:
    raise NotImplementedError

params_file_path = '../../../../../params/params_' + model + '_' + peeling+ '.hdf5'
stims_file_path = '../../../../../stims/' + stim_file + '.hdf5'


args = {'manifest_file': '/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_1_25_222_487664663/genetic_alg/neuron_genetic_alg/manifest.json','axon_type': 'truncated'}

# Number of timesteps for the output volt.
# ntimestep = 10000

# Output destination.
volts_path = '../../../volts/'

# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
params_hdf5 = h5py.File(params_file_path, 'r')
params_name_list = list(params_hdf5.keys())
neg_index =None
if "2" in param_opt_inds:
    neg_index = 1

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


curr_stim_name_list = ['551']
curr_stim_name_list.reverse()
curr_stim_name_list = curr_stim_name_list[:1]
print(ntimestep)
print("params names list",params_name_list)
print("stim name list", curr_stim_name_list)

curr_stim_name_list_copy = copy.deepcopy(curr_stim_name_list)
for curr_stim_name in curr_stim_name_list_copy:
    filepath = volts_path+curr_stim_name+'_volts.hdf5'
    if os.path.isfile(filepath):
        curr_stim_name_list.remove(curr_stim_name)
        
if len(curr_stim_name_list) < 1:
    print("STIM NAME LIST is EMPTY CUS ITS COMPLETE, EXITING")
    exit()

    


pin_set_size = None
pdx_set_size = None

##########################
# Utility Functions      #
##########################
def split(container, count):
    """
    Simple function splitting a container into equal length chunks.

    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]



def run_model(param_set,  sweep, ntimestep):
    
    description = runner.load_description(args)
    utils = runner.create_utils(description)
    param_set =[0.0012383893612445851, 0.0006338765849686779, 0.500453361737873, 0.00001758571027409006, 0.0006796868759747499, 0.00001559856737237686, 0.0005304617338248371,0.14322461216868007,0.0001483591621125165,0.009600187388548958, 0.004787288788001426, 365.1020624075615, 0.000014062989960779357, 0.00013123040228366434, 1.0064836074181785e-7, 0.00007215275674504716]
    h = utils.h
    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY').encode('ascii', 'ignore')
    morphology_path = morphology_path.decode("utf-8")
    utils.generate_morphology(morphology_path)
    utils.load_parameters(param_set)
    # utils.load_cell_parameters()
    responses = []
    dt = stims_hdf5[str(sweep) + "_dt"][:][0]
    stim = stims_hdf5[str(sweep) ][:]
    sweep = int(str(sweep))
    # configure stimulus and recording
    stimulus_path = description.manifest.get_path('stimulus_path')
    run_params = description.data['runs'][0]
    # utils.setup_iclamp(stimulus_path, sweep=sweep)
    # change this so they don't change our dt
    v_init = -78.4 # HARDCODED
    utils.setup_iclamp2(stimulus_path, sweep=sweep, stim=stim, dt=dt, v_init=v_init)
    vec = utils.record_values()
    tstart = time.time()

    if abs(dt*h.nstep_steprun*h.steps_per_ms - 1)  != 0:
        h.steps_per_ms = 1/(dt * h.nstep_steprun)

    h.finitialize()
    h.run()
    tstop = time.time()
    res =  utils.get_recorded_data(vec)
        
    import pdb; pdb.set_trace()
    import matplotlib.pyplot as plt
    target_volts = h5py.File('/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_4_20_22_487664663/target_volts/target_volts_487664663.hdf5','r')
    
    
    final_voltage = res['v']*1000
    plt.plot(final_voltage)
    plt.plot(target_volts['551'])
    plt.savefig('tst.png')
    
    return final_voltage

        

# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

for stim_ind in range(len(curr_stim_name_list)):
    # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        # Each job should contain params_name, a single param set index
        # and number of total params as a list: [params_name, param_ind, stim_ind, n]
        jobs = []
        for params_name in params_name_list:
            if 'orig' in params_name:
                jobs.append([params_name, 0, stim_ind, 1])
            elif 'pin' in params_name:
                n = params_hdf5[params_name].shape[0]
                pin_set_size = n
                for param_ind in range(n):
                    jobs.append([params_name, param_ind, stim_ind, n])
            elif 'pdx' in params_name:
                n = params_hdf5[params_name].shape[0]
                pdx_set_size = n
                for param_ind in range(n):
                    jobs.append([params_name, param_ind, stim_ind, n])
            else:
                continue
        # Split into however many cores are available.
        jobs = split(jobs, COMM.size)
    else:
        jobs = None

    jobs = COMM.scatter(jobs, root=0)
    # Now each rank just does its jobs and collects everything in a results list.
    # Make sure to not use super big objects in there as they will be pickled to be
    # exchanged over MPI.
    results = {}
    for job in jobs:
        # Compute voltage trace for each param sets.
        [params_name, param_ind, stim_ind, n] = job
        curr_stim_name = curr_stim_name_list[stim_ind]
        print("Currently working on stim " + curr_stim_name + " and params " + str(param_ind+1) + " of " + str(n))
        params_data = params_hdf5[params_name][param_ind]
        if neg_index:
            params_data[neg_index] =  - np.abs(params_data[neg_index])
            
        volts_at_i = run_model(params_data, curr_stim_name, ntimestep)
        result_key = (params_name, param_ind, stim_ind)
        results[result_key] = volts_at_i

    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        flattened_dict = {}
        for d in results:
            k = d.keys()
            for key in k:
                flattened_dict[key] = d[key]

        curr_stim_name = curr_stim_name_list[stim_ind]
        curr_stim_size = len(stims_hdf5[curr_stim_name])
        volts_hdf5 = h5py.File(volts_path+curr_stim_name+'_volts.hdf5', 'w')
        for params_name in params_name_list:
            if 'orig' in params_name:
                volts = flattened_dict[(params_name, 0, stim_ind)]
                name_to_write = 'orig' + '_' + curr_stim_name
                print("Processing ", name_to_write)
                volts_hdf5.create_dataset(name_to_write, data=volts)
            elif 'pin' in params_name:
                volts = np.empty((pin_set_size, curr_stim_size))
                name_to_write = 'pin' + '_' + curr_stim_name
                for param_ind in range(pin_set_size):
                    volts[param_ind] = flattened_dict[(params_name, param_ind, stim_ind)]
                    print("Processing ", name_to_write, str(param_ind+1)+"/"+str(pin_set_size))
                volts_hdf5.create_dataset(name_to_write, data=volts)
            elif 'pdx' in params_name:
                volts = np.empty((pdx_set_size, curr_stim_size))
                name_to_write = 'pdx' + '_' + curr_stim_name
                for param_ind in range(pdx_set_size):
                    volts[param_ind] = flattened_dict[(params_name, param_ind, stim_ind)]
                    print("Processing ", name_to_write, str(param_ind+1)+"/"+str(pdx_set_size))
                volts_hdf5.create_dataset(name_to_write, data=volts)
        volts_hdf5.close()
