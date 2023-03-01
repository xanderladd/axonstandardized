import numpy as np
import h5py
import os
os.chdir("neuron_files/allen/")
from neuron import h
os.chdir("../../")
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
#import ap_tuner as tuner
from config import *



normalizers = {}
for stim_name in opt_stim_name_list:
    with open(os.path.join(scores_path,'normalizers', stim_name.decode('ASCII'))+ \
              '_normalizers.pkl','rb') as f:
        normalizers[stim_name] = pickle.load(f)
# constant to scale passsive scores by
if passive:
    PASSIVE_SCALAR = 1
else:
    PASSIVE_SCALAR = .05 # was 2

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

def run_model(param_set, stim_name_list):
    h.load_file(run_file)
    volts_list = []
    for elem in stim_name_list:
        curr_stim = stim_file[elem][:]
        total_params_num = len(param_set)
        if type(elem) != str:
            elem = elem.decode('ASCII')
        dt = stim_file[elem+'_dt']
        if type(dt) != int: # weird dataset formatting
            dt = stim_file[elem+'_dt'][:][0]
        # print(f'dt : {dt} ')
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(curr_stim)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()        
        volts_list.append(out)
    return np.array(volts_list)

def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list, weights):
    
    def passive_chisq(target, data):
        return np.linalg.norm(target-data)**2   / np.linalg.norm(target)

    def eval_function(target, data, function, dt):
        if function in custom_score_functions:
            score = getattr(sf, function)(target, data, dt)
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score

    total_score = 0
    psv_scores = 0
    actv_scores = 0
    active_ind = 0
    for i in range(len(stim_name_list)):
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt_name = stim_name_list[i]
        if type(dt_name) != str:
            dt_name = dt_name.decode('ASCII')
        dt = stims_hdf5[dt_name+'_dt'][0]
        assert dt < .2
        
        curr_stim = stims_hdf5[dt_name][:]
        # HANDLE PASSIVE STIM
        if np.max(curr_target_volt) < 0:
            psv_score = PASSIVE_SCALAR * len(score_function_ordered_list) * passive_chisq(curr_target_volt, curr_data_volt)
            total_score += psv_score
            psv_scores += psv_score
            continue
        # HANDLE ACTIVE STIM
        for j in range(len(score_function_ordered_list)):
            curr_sf = score_function_ordered_list[j].decode('ascii')
            curr_weight = weights[len(score_function_ordered_list)*active_ind + j]
            
            if curr_weight == 0:
                curr_score = 0
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
            if not np.isfinite(curr_score):
                norm_score = 1000 # relatively a VERY high score
            else:
                norm_score = curr_score
                norm_score = normalizers[stim_name_list[i]][curr_sf].transform(np.array(curr_score).reshape(-1,1))[0]  # load and use a saved sklean normalizer from previous step
                norm_score = min(max(norm_score,-2),2) # allow a little lee-way
                        
            # print("ACTIVE SCORE: ", norm_score * curr_weight)
            total_score += norm_score * curr_weight
            actv_scores += norm_score * curr_weight
        # we have evaled active stim, increment weight index by one
        active_ind += 1
    print('ACTIVE :', actv_scores, "PASV:", psv_scores)
    return total_score



class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        params_ = nrnUtils.readParamsCSV(paramsCSV)
        super(hoc_evaluator, self).__init__()
        self.opt_ind = params_opt_ind
        params_ = [params_[i] for i in self.opt_ind]
        self.orig_params = orig_params
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in params_]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in params_])
        self.weights = opt_weight_list
        self.opt_stim_list = [e for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
        
    def evaluate_with_lists(self, param_values):
        input_values = np.copy(self.orig_params)
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
        if len(negative_inds) > 0:
            for negative_ind in negative_inds:
                input_values[negative_ind] = - np.abs(input_values[negative_ind])
        data_volts_list = run_model(input_values, self.opt_stim_list)
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights)
#         ap_tune_score = ap_tune(input_values, self.ap_tune_target, self.ap_tune_stim_name, self.ap_tune_weight)
        return [score] #+ ap_tune_score] NOT USING AP TUNE


