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


# constant to scale passsive scores by
PASSIVE_SCALAR = .5

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']
# Number of timesteps for the output volt.
ntimestep = 10000

def run_model(param_set, stim_name_list):
    h.load_file(run_file)
    volts_list = []
    for elem in stim_name_list:
        stims_hdf5 = h5py.File(stims_path, 'r')
        curr_stim = stims_hdf5[elem][:]
        total_params_num = len(param_set)
        if type(elem) != str:
            elem = elem.decode('ASCII')
        dt = stims_hdf5[elem+'_dt']
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

    def get_fist_zero(stim):
        for i in range(len(stim)-2, -1, -1):
            if stim[i] > 0 and stim[i+1] == 0:
                return i+1
        return None
    def check_ap_at_zero(stim, volt):
        first_zero_ind = get_fist_zero(stim)
        if first_zero_ind:
            if np.mean(stim[first_zero_ind:]) == 0:
                first_ind_to_check = first_zero_ind + 1000
                APs = [True if v > -50 else False for v in volt[first_ind_to_check:]]
                if True in APs:
                    return 400
        return 0
    def eval_function(target, data, function, dt):
        if function in custom_score_functions:
            score = getattr(sf, function)(target, data, dt)
        else:
            score = sf.eval_efel(function, target, data, dt)
        return score
    def normalize_single_score(newValue, transformation):
        # transformation contains: [bottomFraction, numStds, newMean, std, newMax, addFactor, divideFactor]
        # indices for reference:   [      0       ,    1   ,    2   ,  3 ,    4  ,     5    ,      6      ]
        if newValue > transformation[4]:
            newValue = transformation[4]                                            # Cap newValue to newMax if it is too large
        normalized_single_score = (newValue + transformation[5])/transformation[6]  # Normalize the new score
        if transformation[6] == 0:
            return 1
        return normalized_single_score

    total_score = 0
    for i in range(len(stim_name_list)):
        curr_data_volt = data_volts_list[i]
        curr_target_volt = target_volts_list[i]
        stims_hdf5 = h5py.File(stims_path, 'r')
        dt_name = stim_name_list[i]
        if type(dt_name) != str:
            dt_name = dt_name.decode('ASCII')
        dt = stims_hdf5[dt_name+'_dt'][0]
        curr_stim = stims_hdf5[dt_name][:]
        total_score += check_ap_at_zero(curr_stim, curr_data_volt)
        for j in range(len(score_function_ordered_list)):
            curr_sf = score_function_ordered_list[j].decode('ascii')
            if curr_sf == 'min_voltage_between_spikes':
                curr_weight = 200
            elif "AHP" in curr_sf:
                curr_weight = 150
            else:
                curr_weight = weights[len(score_function_ordered_list)*i + j]
            transformation = h5py.File(scores_path+dt_name+'_scores.hdf5', 'r')['transformation_const_'+curr_sf][:]
#             print(curr_sf, curr_weight)
            if curr_weight == 0:
                curr_score = 0
            elif np.max(curr_target_volt) < 0:
                curr_score = PASSIVE_SCALAR  * passive_chisq(curr_target_volt, curr_data_volt)
#                 print("passive score : ", curr_score)
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
#                 print("active score : ", curr_score)

            norm_score = normalize_single_score(curr_score, transformation)
            if np.isnan(norm_score):
                norm_score = 1
            total_score += norm_score * curr_weight
    return total_score

def ap_tune(param_values, target_volts, stim_name, weight):
    stim_list = [stim_name]
    data_volt = run_model(param_values, stim_list)[0]
    stims_hdf5 = h5py.File(stims_path, 'r')
    dt = stims_hdf5[stim_name+'_dt']
    return tuner.fine_tune_ap(target_volts, data_volt, weight, dt)

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
        self.ap_tune_stim_name = ap_tune_stim_name
        self.ap_tune_weight = ap_tune_weight
        self.ap_tune_target = target_volts_hdf5[self.ap_tune_stim_name][:]
        
    def evaluate_with_lists(self, param_values):
        input_values = np.copy(self.orig_params)
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
        if negative_ind:
            input_values[negative_ind] = - np.abs(input_values[negative_ind])
        data_volts_list = run_model(input_values, self.opt_stim_list)
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights)
#         ap_tune_score = ap_tune(input_values, self.ap_tune_target, self.ap_tune_stim_name, self.ap_tune_weight)
        return [score] #+ ap_tune_score] NOT USING AP TUNE












