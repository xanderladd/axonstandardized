import h5py
import cfg
import copy
import shutil
import os

target_volts_path = f'../target_volts/target_volts_{cfg.model_num}.hdf5'
if not os.path.isfile(target_volts_path):
    target_volts_path = f'../target_volts/allen_data_target_volts_{cfg.model_num}.hdf5'

target_volts = h5py.File(target_volts_path, 'r')
target_volts_keys = target_volts.keys()

v_init = target_volts[list(target_volts_keys)[3]][0]
target_volts.close()

if cfg.model == 'compare_bbp':
    run_model_cori_path = 'neuron_genetic_alg/neuron_files/compare_bbp/run_model_cori.hoc'
    with open(run_model_cori_path, 'r') as f:
        lines = f.readlines()
    new_lines = copy.deepcopy(lines)
    
    for idx, line in enumerate(lines):
        if 'v_init' in line:
            new_lines[idx] = f'v_init = {v_init}\n'
    
    shutil.copyfile(run_model_cori_path, run_model_cori_path + 'backup')
    
    with open(run_model_cori_path, 'w') as f:
        f.writelines(new_lines)