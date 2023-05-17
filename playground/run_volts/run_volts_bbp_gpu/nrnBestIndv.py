import os
os.chdir("../../neuron_genetic_alg/neuron_files/bbp/")
from neuron import h
os.chdir("../../../GPU_genetic_alg/python")

inputFile = open("../../../../../input.txt","r") 
for line in inputFile.readlines():
    if "bbp" in line:
        from config.bbp19_config import *
    elif "allen" in line:
        from config.allen_config import *

def run_single_volts(param_set, stim_data, ntimestep = 10000, dt = 0.02):
    run_file = '../../neuron_genetic_alg/neuron_files/bbp/run_model_cori.hoc'
    h.load_file(run_file)
    total_params_num = len(param_set)
    timestamps = np.array([dt for i in range(ntimestep)])
    h.curr_stim = h.Vector().from_python(stim_data)
    h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
    h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
    h.ntimestep = ntimestep
    h.runStim()
    out = h.vecOut.to_python()
    return np.array(out),np.cumsum(timestamps)

param_set = list(orig_params)
opt_stim_name_list = objectives_file['opt_stim_name_list'][:]
score_function_ordered_list = objectives_file['ordered_score_function_list'][:]
stims_path = '../../stims/' + inputs['stim_file'] + '.hdf5'
stim_file = h5py.File(stims_path, 'r')
res = []
for i in range(len(opt_stim_name_list)):
    res.append(run_single_volts(param_set,stim_file[opt_stim_name_list[i]][:])[0])

res = np.array(res)
print(np.shape(res))
np.savetxt("neuronVolts.csv", res, delimiter=",")
