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


manifest_path = [os.path.join('allen_model',elem) for elem in os.listdir('allen_model') if 'fit' in elem and 'json' in elem][0] 

# if not os.path.isfile(manifest_path):
#     manifest_path = f'allen_model/{cfg.model_num}_fit.json'
with open(manifest_path, 'r') as f:
    lines = f.readlines()
new_lines = copy.deepcopy(lines)

for idx, line in enumerate(lines):
    if 'v_init' in line:
        new_lines[idx] = f"      \"v_init\": {v_init}\n"

shutil.copyfile(manifest_path, manifest_path + 'backup')

with open(manifest_path, 'w') as f:
    f.writelines(new_lines)
    
#[os.path.join('allen_model',file) for file in os.listdir('allen_model') if 'ephys.nwb' in file][0]
ephys_path = os.path.join(cfg.data_dir,'nwb_files', f'{cfg.model_num}.nwb')

new_ephys_path = os.path.join('allen_model',f'{cfg.model_num}_ephys.nwb')
shutil.copyfile(ephys_path, new_ephys_path)