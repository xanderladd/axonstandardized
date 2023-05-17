from mpi4py import MPI
import numpy as np
import h5py
import os, sys
import math
import copy
import config 

# test

 

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



# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

for stim_ind in range(len(config.curr_stim_name_list)):
    # Collect whatever has to be done in a list. Here we'll just collect a list of
    # numbers. Only the first rank has to do this.
    if COMM.rank == 0:
        # Each job should contain params_name, a single param set index
        # and number of total params as a list: [params_name, param_ind, stim_ind, n]
        jobs = []
        for params_name in config.params_name_list:
            if 'orig' in params_name:
                jobs.append([params_name, 0, stim_ind, 1])
            elif 'pin' in params_name:
                n = config.params_hdf5[params_name].shape[0]
                config.pin_set_size = n
                for param_ind in range(n):
                    jobs.append([params_name, param_ind, stim_ind, n])
            elif 'pdx' in params_name:
                n = config.params_hdf5[params_name].shape[0]
                config.pdx_set_size = n
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
        curr_stim_name = config.curr_stim_name_list[stim_ind]
        print("Currently working on stim " + curr_stim_name + " and params " + str(param_ind+1) + " of " + str(n))
        params_data = config.params_hdf5[params_name][param_ind]
        if config.neg_index:
            params_data[config.neg_index] =  - np.abs(params_data[config.neg_index])
            
        volts_at_i = config.run_model(params_data, curr_stim_name, ntimestep)
        result_key = (params_name, param_ind, stim_ind)
        results[result_key] = volts_at_i

    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        flattened_dict = {}
        for d in results:
            k = d.keys()
            for key in k:
                flattened_dict[key] = d[key]

        curr_stim_name = config.curr_stim_name_list[stim_ind]
        curr_stim_size = len(config.stims_hdf5[curr_stim_name])
        volts_hdf5 = h5py.File(config.volts_path+curr_stim_name+'_volts.hdf5', 'w')
        for params_name in config.params_name_list:
            if 'orig' in params_name:
                volts = flattened_dict[(params_name, 0, stim_ind)]
                name_to_write = 'orig' + '_' + curr_stim_name
                print("Processing ", name_to_write)
                volts_hdf5.create_dataset(name_to_write, data=volts)
            elif 'pin' in params_name:
                volts = np.empty((config.pin_set_size, curr_stim_size))
                name_to_write = 'pin' + '_' + curr_stim_name
                for param_ind in range(config.pin_set_size):
                    volts[param_ind] = flattened_dict[(params_name, param_ind, stim_ind)]
                    print("Processing ", name_to_write, str(param_ind+1)+"/"+str(config.pin_set_size))
                volts_hdf5.create_dataset(name_to_write, data=volts)
            elif 'pdx' in params_name:
                volts = np.empty((config.pdx_set_size, curr_stim_size))
                name_to_write = 'pdx' + '_' + curr_stim_name
                for param_ind in range(config.pdx_set_size):
                    volts[param_ind] = flattened_dict[(params_name, param_ind, stim_ind)]
                    print("Processing ", name_to_write, str(param_ind+1)+"/"+str(config.pdx_set_size))
                volts_hdf5.create_dataset(name_to_write, data=volts)
        volts_hdf5.close()
