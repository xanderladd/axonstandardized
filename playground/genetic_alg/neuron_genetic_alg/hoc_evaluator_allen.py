import numpy as np
import h5py
from neuron import h
import bluepyopt as bpop
import nrnUtils
import score_functions as sf
import efel
import pandas as pd
import pickle
from config import *
# from mpi4py import MPI
import allensdk.core.json_utilities as ju
# from biophys_optimize.utils import Utils
from allensdk.model.biophysical.utils import create_utils
import allensdk.model.biophysical.runner as runner
import allensdk.core.json_utilities as ju

from biophys_optimize.environment import NeuronEnvironment
import time
from sklearn.preprocessing import MinMaxScaler
import logging
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.disabled = True
# os.chdir("../../")

# comm = MPI.COMM_WORLD
# global_rank = comm.Get_rank()
# size = comm.Get_size()


normalizers = {}
for stim_name in opt_stim_name_list:
    with open(os.path.join(scores_path,'normalizers', stim_name.decode('ASCII'))+ \
              '_normalizers.pkl','rb') as f:
        normalizers[stim_name] = pickle.load(f)



global GEN
GEN = 0
# # constant to scale passsive scores by
# if passive:
#     PASSIVE_SCALAR = 1
# else:
#     PASSIVE_SCALAR = .01 # turns passive scores off
PASSIVE_SCALAR = 0 

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

# Number of timesteps for the output volt.
ntimestep = 30000


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

# Value of dt in miliseconds
dt = 0.02


def run_model(param_set, stim_name_list):
    
    description = runner.load_description(args)


    utils = runner.create_utils(description)
    h = utils.h

    # configure model
    manifest = description.manifest
    morphology_path = description.manifest.get_path('MORPHOLOGY').encode('ascii', 'ignore')
    morphology_path = morphology_path.decode("utf-8")
    utils.generate_morphology(morphology_path)
    utils.load_parameters(param_set)
    responses = []
    for sweep in stim_name_list:
        dt = stim_file[str(sweep.decode('ascii')) + "_dt"][:][0]
        stim = stim_file[str(sweep.decode('ascii')) ][:]

        sweep = int(str(sweep.decode('ascii')))
        # configure stimulus and recording
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        # change this so they don't change our dt
        v_init = target_volts_hdf5[str(sweep)][0] # - 14
        utils.setup_iclamp2(stimulus_path, sweep=sweep, stim=stim, dt=dt, v_init=v_init)
        vec = utils.record_values()
        tstart = time.time()
        # ensure they don't change dt during the sim
        if abs(dt*h.nstep_steprun*h.steps_per_ms - 1)  != 0:
            h.steps_per_ms = 1/(dt * h.nstep_steprun)
            
        h.finitialize()
        h.run()
        tstop = time.time()
        res =  utils.get_recorded_data(vec)
        # rescale recorded data to mV
        res['v'] = res['v']*1000
        responses.append(res['v'] )

    return responses

def evaluate_score_function(stim_name_list, target_volts_list, data_volts_list, weights):
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
        with open(f"../../scores/normalizers/{stim_name_list[i].decode('ascii')}_normalizers.pkl",'rb') as f:
            normalizers = pickle.load(f)
        for j in range(len(score_function_ordered_list)):
            curr_sf = score_function_ordered_list[j].decode('ascii')
            curr_weight = weights[len(score_function_ordered_list)*i + j]
            transformation = normalizers[curr_sf]
            if curr_weight == 0:
                norm_score = 0
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
                norm_score = transformation.transform(np.array(curr_score).reshape(-1,1))[0][0]
            if np.isnan(norm_score):
                norm_score = 1
            total_score += norm_score * curr_weight
    return total_score

class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        super(hoc_evaluator, self).__init__()

        params = ju.read('/global/cscratch1/sd/zladd/allen_optimize/biophys_optimize/biophys_optimize/fit_styles/f9_fit_style.json')
        
        param_df = pd.read_csv('/global/cscratch1/sd/zladd/axonstandardized/playground/runs/allen_full_09_12_22_487664663_base5/genetic_alg/params/params_allen_full.csv')
        mins, maxs = param_df['Lower bound'].values, param_df['Upper bound'].values
        
        
        self.bases = maxs / mins
        # log bound min/max
        mins, maxs = np.array([math.log(mins[i],self.bases[i]) for i in range(len(mins))]), np.array([math.log(maxs[i],self.bases[i]) for i in range(len(maxs))])
        
        fit_params = ju.read('487664663_fit.json')
        names = [elem['name'] for elem in fit_params['genome']]
        target_params = [elem['value'] for elem in fit_params['genome']]
        self.orig_params = param_df['Base value'].values
        # log bound orig params
        self.orig_params = np.array([math.log(self.orig_params[i],self.bases[i]) for i in range(len(mins))])
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in zip(names,mins,maxs)]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in zip(names,mins,maxs)])
        self.weights = opt_weight_list
        self.opt_stim_list = [e for e in opt_stim_name_list]
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]
        print("Init target volts")
        self.target_volts_list = [target_volts_hdf5[s][:] for s in self.opt_stim_list]
        self.opt_ind = np.arange(16)
    
        
    def evaluate_with_lists(self, param_values):
        input_values = self.orig_params
        for i in range(len(param_values)):
            curr_opt_ind = self.opt_ind[i]
            input_values[curr_opt_ind] = param_values[i]
            
        # undo log x-form
        input_values = np.array([math.pow(self.bases[i], input_values[i]) for i in range(len(input_values))])
            
        data_volts_list = run_model(input_values, self.opt_stim_list)
        score = evaluate_score_function(self.opt_stim_list, self.target_volts_list, data_volts_list, self.weights)
        return [score]











