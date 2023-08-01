import csv
import pandas as pd
import os
import numpy as np
import h5py
import pickle
import warnings
warnings.filterwarnings('ignore')

input_file = open('../../input.txt', "r")
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
passive = eval(inputs['passive'])
orig_name = "orig_" + peeling
orig_params = h5py.File('../../params/params_' + model + '_' + peeling + '.hdf5', 'r')[orig_name][0]


if 'log_transform_params' in inputs:
    log_transform_params = bool(inputs['log_transform_params'])
else:
    log_transform_params = False
if usePrev == "True":
    params_csv = '../../params/params_' + model + '_' + peeling + '_prev.csv'
else:
    params_csv = '../../params/params_' + model + '_' + peeling + '.csv'
base_thresh = 50


if 'added_stims' in inputs:
    added_stims = [elem.encode('ASCII') for elem in inputs['added_stims'].split(',')]
else:
    added_stims = []

starting_pop_hack = os.path.join(data_dir, 'populations', 'starting_pop.pkl')

if log_transform_params and starting_pop_hack:
    starting_pop_hack = None
print('log_transform_params: ', log_transform_params)
print('starting_pop_hack: ', starting_pop_hack)
# starting_pop_hack = None
passive_scaler = 0
# constant to scale passsive scores by
if passive:
    PASSIVE_PERCTENAGE  = 1
else:
    PASSIVE_PERCTENAGE = 1 # was 2


if 'dt' in inputs and inputs['dt'] != 'null':
    dt = float(inputs['dt'])
else:
    dt = None
    
if 'timesteps' in inputs and inputs['timesteps'] != 'null':
    ntimestep = int(inputs['timesteps'])
else:
    ntimestep = 10000

if model == 'bbp':
    neuron_path = './neuron_files/bbp/'
    run_file = './neuron_files/bbp/run_model_cori.hoc'
elif model == 'compare_bbp':
    neuron_path = './neuron_files/compare_bbp/'
    run_file = './neuron_files/compare_bbp/run_model_cori.hoc'
elif model == "allen":
    hoc_files = ["stdgui.hoc", "import3d.hoc", "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/cell.hoc"]
    compiled_mod_library = "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/x86_64/.libs/libnrnmech.so"
    args = {'manifest_file': '/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/manifest.json','axon_type': 'truncated'}
elif model == 'M1_TTPC_NA_HH':
    neuron_path = 'neuron_files/M1_TTPC_NA_HH'
    run_file = None



custom_score_functions = [
            'chi_square_normal',\
            'traj_score_1',\
            'traj_score_2',\
            'traj_score_3',\
            'isi',\
            'rev_dot_product',\
            'KL_divergence']
scores_path = '../../scores/'






if not passive:
    objectives_file = h5py.File('../objectives/multi_stim_without_sensitivity_' + model + '_' + peeling + "_" + date + '_stims.hdf5', 'r')
    score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
    weights = objectives_file['opt_weight_list'][:]
    opt_stim_names = objectives_file['opt_stim_name_list'][:]
    stims_path = '../../stims/' + inputs['stim_file'] + '.hdf5'
    stim_file = h5py.File(stims_path, 'r')
    assert len(opt_stim_names) == (len(weights) /  len(score_function_ordered_list)), "Score function weights and stims are mismatched"
    
    # BESPOKE
    opt_stim_names = np.append(opt_stim_names, added_stims)
    print(opt_stim_names, "STIMS IN USE")
    target_volts_path = '../../target_volts/target_volts_{}.hdf5'.format(inputs['modelNum'])
    if os.path.isfile(target_volts_path):
        print('found allen target volts')
        target_volts = h5py.File(target_volts_path,'r')
        target_volts = [target_volts[elem] for elem in opt_stim_names]
    else:
        target_volts = None
        
        
    ap_tune_stim_name = '18'
else:
    objectives_file = h5py.File(f'../../objectives/allen{model_num}_objectives_passive.hdf5', 'r')
    opt_weight_list = objectives_file['opt_weight_list'][:]
    opt_stim_names = objectives_file['opt_stim_name_list'][:]
    score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
    stims_path = '../../stims/' + inputs['stim_file'] + '_passive.hdf5'
    stim_file = h5py.File(stims_path, 'r')
    weights = []

    scores_path = '../../scores/'
    target_volts_path = '../../target_volts/target_volts_{}_passive.hdf5'.format(inputs['modelNum'])
    target_volts_hdf5 = h5py.File(target_volts_path, 'r')
    


negative_param_inds = []
for idx, param in enumerate(pd.read_csv(params_csv).to_dict(orient='records')):
    if 'e_pas' in param['Param name']:
        negative_param_inds.append(idx)
        
        
normalizers = {}
for stim_name in opt_stim_names:
    with open(os.path.join(scores_path,'normalizers', stim_name.decode('ASCII'))+ \
              '_normalizers.pkl','rb') as f:
        normalizers[stim_name] = pickle.load(f)
        
    
