import pandas as pd
import math
import numpy as np
import h5py


def decode_list(stim_name_list):
    res = []
    for stim_name in stim_name_list:
        if type(stim_name) != str:
            stim_name = stim_name.decode('ASCII')
        res.append(stim_name)
    return res
    



def retrieve_dt(curr_stim_name, stims_hdf5, dt=None):
    if type(curr_stim_name) ==  bytes or type(curr_stim_name) ==  np.bytes_: 
        curr_stim_name = curr_stim_name.decode('ASCII')
    if not dt:
        dt = stims_hdf5[curr_stim_name + '_dt'][:]
    
    assert dt, "DT not specified"
    # assert dt < .1, "DT is too high"
    
    return dt


# def ea_debug_function_ill_probably_never_use_but_is_helpful():
    # if global_rank == argmin or global_rank == 776:
    #     import matplotlib.pyplot as plt
    #     plt.plot(data_volts_list[0], color='red')
    #     plt.plot(self.target_volts_list[0], color='black')
    #     plt.savefig(f'volts_{global_rank}.png')
    #     plt.close()
    #     np.save(f'volts_{global_rank}.npy', np.array(data_volts_list[0]))
