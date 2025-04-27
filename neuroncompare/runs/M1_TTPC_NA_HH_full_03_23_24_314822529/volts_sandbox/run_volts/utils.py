import pandas as pd
import math
import numpy as np
import h5py
import config


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