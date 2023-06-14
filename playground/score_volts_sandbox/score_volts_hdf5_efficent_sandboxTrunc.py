import os, sys
# really try to prevent auto multithreadings
os.environ["OMP_NUM_THREADS"] = "1" 
print("num threads", os.environ["OMP_NUM_THREADS"])
from mpi4py import MPI
import numpy as np
import score_functions as sf
import score_normalizer as sn
import efel
import h5py
import math

def split(container, count):
    return [container[_i::count] for _i in range(count)]

def get_name(obj):
    """
    Returns the name of a (score) function, given the actual function handle.
    """
    if isinstance(obj, str):
        return obj
    else:
        return obj.__name__

def eval_function(target, data, function, dt):
    if isinstance(function, str):
        score = sf.eval_efel(function, target, data, dt)
    else:
        score = function(target, data, dt)
    return score

# time step in miliseconds
dt = 0.02
max_score = 1000
model = sys.argv[2]
peeling = sys.argv[3]

# path setting required
params = "/params/params_" + model + "_" + peeling + ".hdf5"
params_file_name = "./params/params_" + model + "_" + peeling + ".hdf5"


volts_path = '../../../volts/'
output_path = '../../../scores/'

# modification to run stims 10 Nodes X 30 stims = 300stims
with open('./input.txt', 'r') as f:
    for line in f:
        if "=" in line:
            name, value = line.split("=")
            if name == "num_nodes":
                num_nodes = int(value)
            if name == "num_volts":
                num_volts = int(value)
            if name == "stim_file":
                stim_file = value.replace('\n','')

volts_name_list = sorted(os.listdir(volts_path))
volts_name_list = [volt_name for volt_name in volts_name_list if "hdf5" in volt_name]

params = h5py.File(params_file_name, 'r')

num_volts_to_run = 1
i=int(sys.argv[1])
if i == 0 and num_nodes == 1:
    volts_name_list = volts_name_list
elif num_nodes > 1 and num_volts == 0:
    num_volts_to_run = math.ceil(len(volts_name_list) / num_nodes)
    volts_name_list = volts_name_list[(i-1)*num_volts_to_run:(i)*num_volts_to_run]
else:
    volts_name_list = volts_name_list[(i-1)*num_volts_to_run:(i)*num_volts_to_run]
print("looking at volts: ",volts_name_list)

custom_score_functions = [
                    sf.chi_square_normal,\
                    #sf.abs_cumsum_diff,\
                    #sf.comp_rest_potential,\
                    #sf.comp_width,\
                    #sf.comp_width_avg,\
                    #sf.comp_height,\
                    #sf.comp_height_avg,\
                    sf.traj_score_1,\
                    sf.traj_score_2,\
                    sf.traj_score_3,\
                    sf.isi,\
                    sf.rev_dot_product,\
                    sf.KL_divergence]


if stim_file == "neg_stims":
    efel_score_functions = sorted([
                        'time_constant',\
                        'voltage_deflection',\
                        'voltage_deflection_vb_ssse',\
                        'ohmic_input_resistance',\
                        'ohmic_input_resistance_vb_ssse',\
                        'maximum_voltage',\
                        'minimum_voltage',\
                        'steady_state_voltage',\
                        'steady_state_hyper',\
                        'voltage_deflection_begin',\
                        'voltage_after_stim',\
                        'steady_state_voltage_stimend',\
                        'voltage_base',\
                        'decay_time_constant_after_stim',\
                        'maximum_voltage_from_voltagebase',\
                        'sag_amplitude',\
                        'sag_ratio1',\
                        'sag_ratio2'])
elif stim_file == "both_stims":
    print("BOTH STIMS SO BOTH SFS")
    efel_score_functions2 = sorted([
                        'AP_duration', 'AP_fall_time', 'AP_amplitude_change', 'spike_half_width', 'AP2_AP1_diff', 'fast_AHP', 'min_voltage_between_spikes', 'AP1_width', 'steady_state_voltage', 'depolarized_base', 'time_to_last_spike', 'minimum_voltage', 'AP_width', 'max_amp_difference', 'adaptation_index2', 'AP2_width', 'AHP_depth_abs', 'peak_indices', 'time_to_first_spike', 'ISI_values', 'peak_time', 'adaptation_index', 'time_constant', 'AP_fall_indices', 'AP_amplitude', 'AP_rise_indices', 'voltage_after_stim', 'burst_mean_freq', 'AHP2_depth_from_peak', 'AP_rise_rate', 'mean_AP_amplitude', 'time_to_second_spike', 'AP_begin_width', 'amp_drop_first_second', 'AP_fall_rate', 'AP1_peak', 'AP_fall_rate_change', 'AHP_time_from_peak', 'AP1_amp', 'AP_begin_time', 'AP1_begin_width', 'AP_amplitude_from_voltagebase', 'AP2_peak', 'peak_voltage', 'AHP1_depth_from_peak', 'AP_height', 'AHP_depth', 'ohmic_input_resistance', 'AP2_begin_width', 'AHP_depth_from_peak', 'AP_duration_half_width_change'])
    efel_score_functions = efel_score_functions2#sorted(efel_score_functions1 + efel_score_functions2)
    
else:
    efel_score_functions = sorted([
                        'peak_indices',\
                        'ISI_values',\
                        'peak_voltage',\
                        'mean_frequency',\
                        'peak_time',\
                        'time_to_first_spike',\
                        'adaptation_index',\
                        'adaptation_index2',\
                        'spike_width2',\
                        'AP_width',\
                        'burst_mean_freq',\
                        'burst_number',\
                        'interburst_voltage',\
                        'AP_height',\
                        'AP_amplitude',\
                        'AHP_depth_abs_slow',\
                        'AHP_slow_time',\
                        'depolarized_base',\
                        'Spikecount',\
                        'AHP_depth',\
                        'AP_rise_indices',\
                        'AP_end_indices',\
                        'AP_fall_indices',\
                        'AP_duration',\
                        'AP_duration_half_width',\
                        'AP_rise_time',\
                        'AP_fall_time',\
                        'AP_rise_rate',\
                        'AP_fall_rate',\
                        'fast_AHP',\
                        'AP_amplitude_change',\
                        'AP_duration_change',\
                        'AP_rise_rate_change',\
                        'AP_fall_rate_change',\
                        'fast_AHP_change',\
                        'AP_duration_half_width_change',\
                        'amp_drop_first_second',\
                        'amp_drop_first_last',\
                        'amp_drop_second_last',\
                        'max_amp_difference',\
                        'AP_amplitude_diff',\
                        'irregularity_index',\
                        'AP1_amp',\
                        'APlast_amp',\
                        'AP2_amp',\
                        'AP1_peak',\
                        'AP2_peak',\
                        'AP2_AP1_diff',\
                        'AP2_AP1_peak_diff',\
                        'AP1_width',\
                        'AP2_width',\
                        'AHP_depth_from_peak',\
                        'AHP_time_from_peak',\
                        'AHP1_depth_from_peak',\
                        'AHP2_depth_from_peak',\
                        'time_to_second_spike',\
                        'time_to_last_spike',\
                        'spike_half_width',\
                        'AP_begin_indices',\
                        'AHP_depth_abs',\
                        'AP_begin_width',\
                        'AP_begin_voltage',\
                        'AP_begin_time',\
                        'AP1_begin_voltage',\
                        'AP2_begin_voltage',\
                        'AP1_begin_width',\
                        'AP2_begin_width',\
                        'is_not_stuck',\
                        'mean_AP_amplitude',\
                        'voltage_after_stim',\
                        'AP_amplitude_from_voltagebase',\
                        'min_voltage_between_spikes'])
score_functions = custom_score_functions + efel_score_functions
print("USING : ", score_functions)

COMM = MPI.COMM_WORLD
print(COMM.size, "COM SIZE")
for k in range(len(volts_name_list)):
    curr_volts_name = volts_name_list[k]
    curr_stim_name = curr_volts_name.replace('_volts.hdf5', '')
    orig_volts_name = 'orig_'+curr_stim_name
    pin_volts_name = 'pin_'+curr_stim_name
    #pdx_volts_name = 'pdx_'+curr_stim_name
    if "hdf5" in curr_volts_name:
        volts = h5py.File(volts_path+curr_volts_name, 'r')
    else:
        continue
    pin_size = volts[pin_volts_name].shape[0]
    #pdx_size = volts[pdx_volts_name].shape[0]

    # A job will be a list with prefix, function index,
    # volts data index and total indices: [prefix, function_ind, volts_ind, n]
    if COMM.rank == 0:
        jobs = []
        for i in range(len(score_functions)):
            for j in range(pin_size):
                jobs.append(['pin', i, j, pin_size])
            #for j in range(pdx_size):
                #jobs.append(['pdx', i, j, pdx_size])
        # Split into however many cores are available.
        jobs = split(jobs, COMM.size)
    else:
        jobs = None

    COMM.Barrier()
    # Scatter jobs across cores.
    jobs = COMM.scatter(jobs, root=0)

    results = {}
    for job in jobs:
        [prefix, function_ind, volts_ind, n] = job
        curr_function = score_functions[function_ind]
        orig_volts_data = volts[orig_volts_name][:]
        if prefix == 'pin':
            curr_volts_data = volts[pin_volts_name][volts_ind]
        #elif prefix == 'pdx':
            #curr_volts_data = volts[pdx_volts_name][volts_ind]
        if volts_ind % 1000 == 0:
            print('Working on', prefix, curr_stim_name, get_name(curr_function), str(volts_ind)+'/'+str(n))

        score = eval_function(orig_volts_data, curr_volts_data, curr_function, dt)
        results[(prefix, function_ind, volts_ind)] = score

    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        flattened_dict = {}
        for d in results:
            k = d.keys()
            for key in k:
                flattened_dict[key] = d[key]

        scores_hdf5 = h5py.File(output_path+curr_stim_name+'_scores.hdf5', 'w')
        score_function_names = []
        for i in range(len(score_functions)):
            curr_function_name = get_name(score_functions[i])
            score_function_names.append(np.string_(curr_function_name))
            pin_scores = np.empty((pin_size, 1))
            #pdx_scores = np.empty((pdx_size, 1))
            params_sample_pin_ind = params['sample_ind'][:]
            params_dx = params['dx'][0]
            free_params_size = params['param_num'][0]
            for j in range(pin_size):
                pin_scores[j] = flattened_dict[('pin', i, j)]
            #for j in range(pdx_size):
                #pdx_scores[j] = flattened_dict[('pdx', i, j)]
            print('Saving', curr_function_name)
            sampled_pin_scores = np.array([pin_scores[p_ind] for p_ind in params_sample_pin_ind])
            sampled_pin_repeat = np.repeat(sampled_pin_scores, free_params_size, axis=0)
            #sensitivity_mat = abs(pdx_scores - sampled_pin_repeat)/params_dx
            norm_pin_scores, transformation = sn.normalize(pin_scores)
            scores_hdf5.create_dataset('raw_pin_scores_'+curr_function_name, data=pin_scores)
            #scores_hdf5.create_dataset('raw_pdx_scores_'+curr_function_name, data=pdx_scores)
            scores_hdf5.create_dataset('norm_pin_scores_'+curr_function_name, data=norm_pin_scores)
            #scores_hdf5.create_dataset('sensitivity_mat_'+curr_function_name, data=sensitivity_mat)
            scores_hdf5.create_dataset('transformation_const_'+curr_function_name, data=transformation)
        scores_hdf5.create_dataset('score_function_names', data=score_function_names)
        scores_hdf5.create_dataset('stim_name', data=np.array([np.string_(curr_stim_name)]))
        scores_hdf5.close()
