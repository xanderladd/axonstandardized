import csv
import pandas as pd
import os
import numpy as np
import h5py
from allensdk.model.biophysical.utils import create_utils
import allensdk.model.biophysical.runner as runner
import allensdk.core.json_utilities as ju
import pickle

from biophys_optimize.environment import NeuronEnvironment

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
assert inputs['model'] in ['allen', 'mainen', 'bbp'], "Model must be from: \'allen\' \'mainen\', \'bbp\'. Do not include quotes."
assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
params_opt_ind = [int(p)-1 for p in inputs['params'].split(",")]
date = inputs['runDate']
usePrev = inputs['usePrevParams']
model_num = inputs['modelNum']
model_num = 487664663
passive = eval(inputs['passive'])

orig_name = "orig_" + peeling
orig_params = h5py.File('../params/params_' + model + '_' + peeling + '.hdf5', 'r')[orig_name][0]


if usePrev == "True":
    paramsCSV = '../params/params_' + model + '_' + peeling + '_prev.csv'
else:
    paramsCSV = '../params/params_' + model + '_' + peeling + '.csv'

    
if peeling == "sodium":
    templateCSV = "../../params/params_bbp_peeling_description.csv"
else:
    templateCSV = "../../params/params_bbp_{}.csv".format(peeling)
    
    
run_file = './neuron_files/allen/run_model_cori.hoc'
run_volts_path = './'

if not passive:
    objectives_file = h5py.File('../objectives/multi_stim_without_sensitivity_' + model + '_' + peeling + "_" + date + '_stims.hdf5', 'r')
    score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
    opt_weight_list = objectives_file['opt_weight_list'][:]
    opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
    stims_path = '../../stims/' + inputs['stim_file'] + '.hdf5'
    stim_file = h5py.File(stims_path, 'r')
    print(stims_path, "STIM PTA")
    # deletable : 53_2, (4,5,6), 64, 65, 51_3, , 48_3, 
    assert len(opt_stim_name_list) == (len(opt_weight_list) /  len(score_function_ordered_list)), "Score function weights and stims are mismatched"
    # opt_stim_name_list_decode = np.char.decode(opt_stim_name_list)
    # keep_list = [b'10', b'7', b'71', b'75', b'76', b'77', b'18', b'20']
    # keep_list = [b'10', b'7', b'71', b'75', b'76', b'77', b'18', b'20', b'21', b'23', b'24']
    print(opt_stim_name_list, "orig stim set")
    delete = [key for key in opt_stim_name_list if np.char.decode(key) not in list(stim_file.keys())] #or not key in keep_list ] 
    #delete = [b'10', b'7',  b'11', b'14', b'18', b'8', b'19']
    


#     delete_inds = np.flip(np.where(np.isin(opt_stim_name_list, delete))[0])
#     for ind in delete_inds:
#         delete_slice = np.arange(ind*len(score_function_ordered_list), (ind+1)*len(score_function_ordered_list))
#         opt_weight_list = np.delete(opt_weight_list, delete_slice)
    
#     opt_stim_name_list = np.array([elem for elem in opt_stim_name_list if elem not in delete])
    # last_weights = opt_weight_list[-79:]
    
    # add stim 17
    # opt_stim_name_list = np.append(opt_stim_name_list,b'17') # short square
    # opt_weight_list = np.append(opt_weight_list,last_weights)
    
    # add stim 27
#     opt_stim_name_list = np.append(opt_stim_name_list,b'27') # short square
#     opt_weight_list = np.append(opt_weight_list,last_weights)
    
    
    # opt_stim_name_list = np.append(opt_stim_name_list,b'33') # short square
    # opt_weight_list = np.append(opt_weight_list,last_weights)
    
    # opt_stim_name_list = np.append(opt_stim_name_list,b'36') # short square
    # opt_weight_list = np.append(opt_weight_list,last_weights)
    
    assert len(opt_stim_name_list)  == (len(opt_weight_list) /  len(score_function_ordered_list)), "Score function weights and stims are mismatched"
    # opt_stim_name_list = np.delete(opt_stim_name_list,[2,16,5])
    # opt_stim_name_list = np.append(opt_stim_name_list,[b'8',b'21', b'16',b'23', b'34'])
    # opt_stim_name_list = [elem for elem in opt_stim_name_list if elem != b'103']
    print(opt_stim_name_list, "STIMS IN USE")
    
    ap_tune_weight = 0
    scores_path = '../../scores/'
    target_volts_path = '../../target_volts/target_volts_{}.hdf5'.format(inputs['modelNum'])
    # target_volts_path = 't_volts.hdf5'
    target_volts_hdf5 = h5py.File(target_volts_path, 'r')
#     with open('t_volts.pkl','rb') as f:
#         target_volts_hdf5 = pickle.load(f)
        
    ap_tune_stim_name = '18'
else:
    objectives_file = h5py.File(f'../../objectives/allen{model_num}_objectives_passive.hdf5', 'r')
    opt_weight_list = objectives_file['opt_weight_list'][:]
    opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
    print("MONKEY PATCH HERE TO ADD PASSIVE STIMS AND REMOVE SOME ACTIVE")
    # deletable : 53_2, (4,5,6), 64, 65, 51_3, , 48_3, 
    # opt_stim_name_list = np.delete(opt_stim_name_list,[64,16,5, 65, ])
    # opt_stim_name_list = np.append(opt_stim_name_list,[b'19',b'7',b'18',b'26', b'30'])
    # opt_stim_name_list = [elem for elem in opt_stim_name_list if elem != b'103']
    print(opt_stim_name_list)
    score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
    stims_path = '../../stims/' + inputs['stim_file'] + '_passive.hdf5'
    stim_file = h5py.File(stims_path, 'r')
    ap_tune_weight = 0

    scores_path = '../../scores/'
    target_volts_path = '../../target_volts/target_volts_{}_passive.hdf5'.format(inputs['modelNum'])
    target_volts_hdf5 = h5py.File(target_volts_path, 'r')
    



# negative_inds = []
# for idx, param in enumerate(pd.read_csv(paramsCSV).to_dict(orient='records')):
#     if 'e_pas' in param['Param name']:
#         negative_inds.append(idx)
        
# Number of timesteps for the output volt.
ntimestep = int(inputs['timesteps'])

# target = h5py.File('/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_1_25_222_487664663/target_volts/target_volts_487664663.hdf5','r')

# stimulus_path = '487664663_ephys_sim.nwb'
# params = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/cell_487664663/orig_fit.json')
# param_set = []
# for elem in params['genome']:
#     param_set.append(elem['value'])
# hardcode
hoc_files = ["stdgui.hoc", "import3d.hoc", "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/cell.hoc"]
compiled_mod_library = "/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/x86_64/.libs/libnrnmech.so"

# morph_filename = '/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/cell_487664663/reconstruction.swc'
# passive_results = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/passive_results.json')
# fit_style_data = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/biophys_optimize/fit_styles/f9_fit_style.json')

# preprocess_results = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/test_storage/preprocess_results.json')
# stim_params = preprocess_results['stimulus']

args = {'manifest_file': '/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/neuron_genetic_alg/manifest.json','axon_type': 'truncated'}