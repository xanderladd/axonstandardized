import csv
import pandas as pd
import os
import numpy as np
import h5py
import pickle
import warnings
warnings.filterwarnings('ignore')

input_file = open('../input.txt', "r")
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
assert inputs['model'] in ['allen', 'mainen', 'bbp', 'compare_bbp', 'M1_TTPC_NA_HH'], "Model must be from: \'allen\' \'mainen\', \'bbp\'. Do not include quotes."
assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
data_dir = inputs['data_dir']
params_opt_ind = [int(p)-1 for p in inputs['params'].split(",")]
date = inputs['runDate']
usePrev = inputs['usePrevParams']
model_num = inputs['modelNum']
biophys_model_id = inputs['biophys_model_id']
log_transform_params = bool(inputs['log_transform_params'])
passive = eval(inputs['passive'])
params_csv = '../params/params_' + model + '_' + peeling + '.csv'