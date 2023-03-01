import sys
sys.path.insert(0, '../../')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import h5py
import pickle
import bluepyopt
import re
import os


import plot_helper as ph

def readParamsCSVToDF(fileName):
    df = pd.read_csv(fileName,skipinitialspace=True)
    return df

input_file = open('../../../input.txt', "r")
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
usePrev = inputs['usePrevParams']
stims_path = '../stims/' + inputs['stim_file'] + '.hdf5'

if peeling == "passive":
    next_peeling_step_name = "potassium"
elif peeling == "potassium":
    next_peeling_step_name = "sodium"
elif peeling == "sodium":
    next_peeling_step_name = "calcium"
    
indvLogs = [f for f in os.listdir("./GPU_genetic_alg/python/best_indv_logs/") if os.path.isfile(os.path.join("./GPU_genetic_alg/python/best_indv_logs/", f))]
indvLogNums = []
for log in indvLogs:
    indvLogNums.append(int(re.findall(r'\d+', log)[0]))

lastInd = max(indvLogNums)
print("USING ", lastInd, " AS LAST IND")
GA_result_path = './GPU_genetic_alg/python/best_indv_logs/best_indvs_gen_' \
                    + str(lastInd) + '.pkl'
params_path = './params/params_bbp_'+peeling+'.hdf5'
print(GA_result_path)
base_passive = h5py.File(params_path, 'r')['orig_'+peeling][0]
base = [base_passive[i] for i in params_opt_ind]
lbs = [0.01*p for p in base]
ubs = [100*p for p in base]
params_bbp_passive = [ph.params_bbp[i] for i in params_opt_ind]
normalized_indvs_passive_bbp, best_indvs_passive_bbp = ph.read_and_normalize_with_neg(GA_result_path, base, lbs, ubs)


best_passive = list(base_passive)
for i in range(len(params_opt_ind)):
    best_passive[params_opt_ind[i]] = best_indvs_passive_bbp[-1][i]

best = [best_passive[i] for i in params_opt_ind]
quartiles = [0,.25,.5,.75,1]
log_lbs = []
log_ubs = []
for i in range(len(ubs)):
    bins = [np.quantile(np.log([lbs[i],ubs[i]]),q) for q in quartiles]
    dataQ = 1 - (np.log(ubs[i]) - np.log(best)[i]) / (np.log(ubs[i]) -  np.log(lbs[i]) )
    lowerQ = max(0.01,dataQ - .125)
    upperQ = min(1,dataQ + .125)
    # base case where we hit edges of percentile ranges
    if lowerQ == .01:
        # UNTESTED
        upperQ += abs(dataQ - .125)
    if upperQ == 1:
        lowerQ -= (dataQ + .125) - 1
    logBounds = (np.quantile(np.log([lbs[i],ubs[i]]),lowerQ), \
              np.quantile(np.log([lbs[i],ubs[i]]),upperQ))
    bounds = (np.exp(logBounds[0]), np.exp(logBounds[1]))
    log_lbs.append(bounds[0])
    log_ubs.append(bounds[1])
    
print("+/- 25% of range")
for i in range(len(log_lbs)):
    print("log lower bound: ", log_lbs[i], "log upper bound: ",\
          log_ubs[i], "best: " ,best[i], "\n")
paramsCSV = './params/params_' + model + '_' + peeling + '.csv'
params = readParamsCSVToDF(paramsCSV)

params['Base value'] = best_passive
params['Lower bound'][params_opt_ind] = log_lbs
params['Upper bound'][params_opt_ind] = log_ubs

params = params.set_index("Param name").astype({'Base value': 'float32','Lower bound': 'float32','Upper bound': 'float32'})
print(params)
params.rename( columns={'Unnamed: 10':''}, inplace=True )

nextParamsCSV = '../../../param_stim_generator/params_reference/params_' + model + '_' + next_peeling_step_name + '.csv'
nextParams = readParamsCSVToDF(nextParamsCSV)
nextParams = nextParams.set_index("Param name").astype({'Base value': 'float32','Lower bound': 'float32','Upper bound': 'float32'})
nextParams.rename( columns={'Unnamed: 10':''}, inplace=True )


for col_name in ['Base value', 'Lower bound', 'Upper bound']:
    zero_mask = params.loc[:,col_name] == 0
    #[print(col) for col in nextParams.columns.values if col not in params.columns.values]
    params.loc[zero_mask, col_name] = nextParams.loc[zero_mask, col_name]
params.to_csv("../../../param_stim_generator/params_reference/params_" \
              + model + "_" + next_peeling_step_name + "_prev"  + ".csv")
