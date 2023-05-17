import csv
import pandas as pd
import os
import numpy as np
import h5py
import utils
input_file = open('../../../../input.txt', "r")
inputs = {}
input_lines = input_file.readlines()
for line in input_lines:
    vals = line.split("=")
    if len(vals) != 2 and "\n" not in vals:
        raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
    if "\n" not in vals:
        inputs[vals[0]] = vals[1][:len(vals[1])-1]

assert 'params' in inputs, "No params specificed"
assert 'user' in inputs, "No user specified"
assert 'model' in inputs, "No model specificed"
assert 'peeling' in inputs, "No peeling specificed"
assert 'seed' in inputs, "No seed specificed"
assert inputs['model'] in ['mainen', 'bbp'], "Model must be from: \'mainen\', \'bbp\'. Do not include quotes."
assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
params_opt_ind = [int(p)-1 for p in inputs['params'].split(",")]
date = inputs['runDate']
orig_name = "orig_" + peeling
# orig_params = h5py.File('../../params/params_' + model + '_' + peeling + '.hdf5', 'r')[orig_name][0]
# paramsCSV = '../../params/params_' + model + '_' + peeling + '.csv'
# scores_path = '../../../scores/'
# objectives_file = h5py.File('../../objectives/multi_stim_without_sensitivity_bbp_' + peeling + "_" + date + '_stims.hdf5', 'r')
# opt_weight_list = objectives_file['opt_weight_list'][:]
# opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
# score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
# stims_path = '../../stims/' + inputs['stim_file'] + '.hdf5'
# stim_file = h5py.File(stims_path, 'r')
#target_volts_path = './target_volts/allen_data_target_volts_10000.hdf5'
#target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#params_opt_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#params_opt_ind = np.arange(24) 
model_dir = './'
data_dir = model_dir+'/Data/'
run_dir = './bin'
vs_fn = '/tmp/Data/VHotP'

# Number of timesteps for the output volt.
ntimestep = 10000


