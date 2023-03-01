from allensdk.core.nwb_data_set import NwbDataSet
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py 
import os
import efel
import pickle as pkl
os.chdir('neuron_genetic_alg')
from config import *
os.chdir('../')

rank = int(os.environ['SLURM_PROCID'])
all_features = efel.api.getFeatureNames()
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['agg.path.chunksize'] = 10000


parsed_stim_response_path = f'./allen_model_sota_model_parsed_cell_{model_num}.hdf5'
data_file = h5py.File(parsed_stim_response_path, 'r')
sweep_keys = [e.decode('ascii') for e in data_file['sweep_keys']]

def show_efel(efel_feature_dict, feature_names):
    def normalize_efel_score(cell_features, allen_features, compare_features):
        cell_feature_mean = np.mean(cell_features)
        if cell_feature_mean != 0:
            return np.array(cell_features)/cell_feature_mean, np.array(allen_features)/cell_feature_mean,\
                    np.array(compare_features)/cell_feature_mean 
        else:
            return np.array(cell_features), np.array(allen_features), np.array(compare_features) 
    width=0.4
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, l in enumerate(feature_names):
        efel_data = efel_feature_dict[l]
        cell_features, allen_features, compare_features = efel_data["cell_features"], efel_data["allen_features"], efel_data["compare_features"]
        if cell_features is None or len(cell_features)==0:
            cell_features = [0]
            continue
        if allen_features is None or len(allen_features)==0:
            allen_features = [0]
            continue
        if compare_features is None or len(compare_features)==0:
            compare_features = [0]
            continue
        x_cell = np.ones(len(cell_features))*i + (np.random.rand(len(cell_features))*width/3.-width/3.)
        x_allen = np.ones(len(allen_features))*i + (np.random.rand(len(allen_features))*width/3.)
        x_compare = np.ones(len(compare_features))*i + (np.random.rand(len(compare_features))*width/3.+width/3.)
        cell_features, allen_features, compare_features = normalize_efel_score(cell_features, allen_features, compare_features)
        ax.scatter(x_cell, cell_features, color="black", s=25)
        ax.scatter(x_allen, allen_features, color="blue", s=25)
        ax.scatter(x_compare, compare_features, color="crimson", s=25)

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.scatter([], [], color="black", label="Cell response")
    ax.scatter([], [], color="blue", label="Allen's model response")
    ax.scatter([], [], color="crimson", label="CoMPare model response")
    ax.legend(bbox_to_anchor=(1.5, 1))
    plt.show()
    
def eval_efel(feature_name, target, data, dt):
    def diff_lists(lis1, lis2):
        def safe_mean(lis):
            if np.size(lis) == 0:
                return 0
            return np.mean(lis)
        if lis1 is None and lis2 is None:
            return 0
        if lis1 is None:
            lis1 = [0]
        if lis2 is None:
            lis2 = [0]
        len1, len2 = len(lis1), len(lis2)
        if len1 > len2:
            lis2 = np.concatenate((lis2, np.zeros(len1 - len2)), axis=0)
        if len2 > len1:
            lis1 = np.concatenate((lis1, np.zeros(len2 - len1)), axis=0)
        return np.sqrt(safe_mean((lis1 - lis2)**2))
    time_stamps = len(target)
    time = np.cumsum([dt for i in range(time_stamps)])
    curr_trace_target, curr_trace_data = {}, {}
    stim_start, stim_end = dt, time_stamps*dt
    curr_trace_target['T'], curr_trace_data['T'] = time, time
    curr_trace_target['V'], curr_trace_data['V'] = target, data
    curr_trace_target['stim_start'], curr_trace_data['stim_start'] = [stim_start], [stim_start]
    curr_trace_target['stim_end'], curr_trace_data['stim_end'] = [stim_end], [stim_end]
    traces = [curr_trace_target, curr_trace_data]
    traces_results = efel.getFeatureValues(traces, [feature_name], raise_warnings=False)
    diff_feature = diff_lists(traces_results[0][feature_name], traces_results[1][feature_name])
    return diff_feature, traces_results


efel_data = {}
# for sweep_key in sweep_keys:
try:
    sweep_key = sweep_keys[rank]
except:
    exit()
    
if int(sweep_key) > 78:
    exit()


print(f"processing {sweep_key} sweep for {len(all_features)}")
stim_val = data_file[sweep_key+'_stimulus'][:]
cell_response = data_file[sweep_key+'_cell_response'][:]
allen_response = data_file[sweep_key+'_allen_model_response'][:]
compare_response = data_file[sweep_key+'_compare_model_response'][:]
dt_val = data_file[sweep_key+'_dt'][0]
#plot_sampled(sweep_key, stim_val, cell_response, allen_response, compare_response)
efel_feature_dict = {}
for efel_name in all_features:
    # print(f'efel name : {efel_name}')
    l2_val_allen, efel_values_allen = eval_efel(efel_name, cell_response, allen_response, dt_val)
    l2_val_compare, efel_values_compare = eval_efel(efel_name, cell_response, compare_response, dt_val)
    efel_feature_dict[efel_name] = {"cell_features": efel_values_allen[0][efel_name],\
                                    "allen_features": efel_values_allen[1][efel_name],\
                                    "compare_features": efel_values_compare[1][efel_name]}
#         print('eFEL name:', efel_name)
#         #print(efel_name+" value for target:", efel_values[0][efel_name])
#         #print(efel_name+" value for best model response:", efel_values[1][efel_name])
#         print("Euclidean Distance for Allen's Model:", l2_val_allen)
#         print("Euclidean Distance for CoMParE Model:", l2_val_compare)
#         print('\n')
efel_data[sweep_key] = efel_feature_dict
#     show_efel(efel_feature_dict)
os.makedirs(f'./efel_data/subsets/', exist_ok=True)
pkl.dump(efel_data, open(f'./efel_data/subsets/{sweep_key}_efel.pkl', 'wb'))
