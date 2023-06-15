import csv
import pandas as pd
import os
import numpy as np
import h5py
import pickle
import warnings
warnings.filterwarnings('ignore')

input_file = open('./input.txt', "r")
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
assert inputs['model'] in ['allen', 'mainen', 'bbp', 'compare_bbp'], "Model must be from: \'allen\' \'mainen\', \'bbp\'. Do not include quotes."
assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
params_opt_ind = [int(p)-1 for p in inputs['params'].split(",")]
date = inputs['runDate']
usePrev = inputs['usePrevParams']
model_num = inputs['modelNum']
passive = eval(inputs['passive'])

volts_path = '../../../volts/'
output_path = '../../../scores/'

num_nodes = int(inputs['num_nodes'])
num_volts = int(inputs['num_volts'])
ntimestep = int(inputs['timesteps'])
stim_file = inputs['stim_file']
custom = inputs['custom']
data_dir = inputs['data_dir']
run_dir = f"runs/{model}_{peeling}_{date}_{custom}"

if 'dt' in inputs:
    dt = .02
else:
    dt = None
    
if usePrev == "True":
    params_path = f'../../../params/params_' + model + '_' + peeling + '_prev.hdf5'
else:
    params_path = f'../../../params/params_' + model + '_' + peeling + '.hdf5'
