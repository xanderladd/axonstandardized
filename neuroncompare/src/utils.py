import pandas as pd
import math
import numpy as np
import h5py

def read_inputfile(input_path=''):
    if not len(input_path):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input.txt')
    # Get parameters from input file
    input_file = open(input_path, "r")
    inputs = {}
    input_lines = input_file.readlines()
    for line in input_lines:
        vals = line.split("=")
        if len(vals) != 2 and "\n" not in vals:
            raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
        if "\n" not in vals:
            inputs[vals[0]] = vals[1][:len(vals[1])-1]
    int_keys = ['num_nodes', 'num_volts', 'timesteps']
    for key in int_keys:
        inputs[key] = int(inputs[key])
         
    return inputs

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