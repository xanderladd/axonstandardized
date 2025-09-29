import numpy as np
import h5py
import os, sys
import math
import copy

# Import existing configuration utilities
from neuroncompare.src.path_manager import get_path_manager
from neuroncompare.src.config_manager import get_config
# Get the path manager and config instance
paths = get_path_manager()
config = get_config()
# Log our working directory to help with debugging
print(f"Working directory: {os.getcwd()}")
from neuroncompare.src.run_model import run_model
from mpi4py import MPI

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


if __name__ == "__main__":
    #  SET UP
    num_stims_to_run = 1
    i=int(sys.argv[1])
    stims_hdf5 = h5py.File(config.stims_file_path, 'r')
    params_hdf5 = h5py.File(config.params_file_path, 'r')
    params_name_list = list(params_hdf5.keys())
    stims_name_list = sorted(list(stims_hdf5.keys()))
    stims_name_list = [elem for elem in stims_name_list if "dt" not in elem]
    stims_name_list = ["10","15","35","42","51","53","89","69","68"]


    if i == 0 and config['num_nodes'] == 1:
        curr_stim_name_list = stims_name_list
    elif config.config['num_nodes'] > 1 and config.config['num_volts'] == 0:
        num_stims_to_run = math.ceil(len(stims_name_list) / config.config['num_nodes'] )
        curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]
        print(len(curr_stim_name_list), curr_stim_name_list)
    else:
        curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]

    curr_stim_name_list.reverse()
    # why was this here???
    # curr_stim_name_list = curr_stim_name_list[:1]
    ntimestep = int(config.ntimestep)
    print("params names list",params_name_list)
    print("stim name list", curr_stim_name_list)
    params_name_list = params_name_list
    curr_stim_name_list = curr_stim_name_list

    ignore_stim_names = ['stim_types']

    curr_stim_name_list_copy = copy.deepcopy(curr_stim_name_list)
    for curr_stim_name in curr_stim_name_list_copy:
        filepath = config.volts_path+curr_stim_name+'_volts.hdf5'
        if os.path.isfile(filepath) or curr_stim_name in ignore_stim_names:
            curr_stim_name_list.remove(curr_stim_name)


    if len(curr_stim_name_list) < 1:
        print("STIM NAME LIST is EMPTY CUS ITS COMPLETE, EXITING")
        exit()


    pin_set_size = None
    pdx_set_size = None

    orig_name = "orig_" + config.config['peeling']
    orig_params = h5py.File('../../params/params_' + config.config['model'] + '_' + config.config['peeling']+ '.hdf5', 'r')[orig_name][0]

    # SET UP DONE

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
            if len(config.negative_param_inds):
                for neg_idx in config.negative_param_inds:
                    params_data[neg_idx] =  - np.abs(params_data[neg_idx])
            # don't set dt here, set it in run model
            volts_at_i = run_model(params_data, curr_stim_name, dt=None)
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
            volts_hdf5 = h5py.File(config.volts_path+curr_stim_name+'_volts.hdf5', 'w')
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