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
from sklearn.preprocessing import MinMaxScaler
import pickle
import re
import config

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
dt = config.dt
max_score = 1000

stim_file = h5py.File(f'{config.data_dir}/stims/{config.stim_file}.hdf5','r')
volts_name_list = sorted(os.listdir(config.volts_path))
volts_name_list = [volt_name for volt_name in volts_name_list if "hdf5" in volt_name]
params = h5py.File(config.params_path, 'r')

num_volts_to_run = 1
i=int(sys.argv[1])
if i == 0 and config.num_nodes == 1:
    volts_name_list = volts_name_list
elif config.num_nodes > 1 and config.num_volts == 0:
    num_volts_to_run = math.ceil(len(volts_name_list) / num_nodes)
    volts_name_list = volts_name_list[(i-1)*num_volts_to_run:(i)*num_volts_to_run]
else:
    volts_name_list = volts_name_list[(i-1)*num_volts_to_run:(i)*num_volts_to_run]
    
for volts in volts_name_list:
    if os.path.isfile(os.path.join(config.output_path,volts.replace('volts','scores'))):
        volts_name_list.remove(volts)

print(volts_name_list, "volts to run"
     )
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
                    sf.isi]
                    # sf.rev_dot_product,\
                    # sf.KL_divergence]


if stim_file == "neg_stims":
    efel_score_functions = sorted([
                        'time_constant',\
                        'voltage_deflection',\
                        'voltage_deflection_vb_ssse',\
                        'ohmic_input_resistance',\
                        # 'ohmic_input_resistance_vb_ssse',\
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
    efel_score_functions1 = sorted([
                        'time_constant',\
                        'voltage_deflection',\
                        'voltage_deflection_vb_ssse',\
                        'ohmic_input_resistance',\
                        # 'ohmic_input_resistance_vb_ssse',\
                        'maximum_voltage',\
                        'minimum_voltage',\
                        'steady_state_voltage',\
                        'steady_state_hyper',\
                        'voltage_deflection_begin',\
                        'steady_state_voltage_stimend',\
                        'voltage_base',\
                        'decay_time_constant_after_stim',\
                        'maximum_voltage_from_voltagebase',\
                        'sag_amplitude',\
                        'sag_ratio1',\
                        'sag_ratio2'])
    efel_score_functions2 = sorted([
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
                        # 'depolarized_base',\
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
                        'min_voltage_between_spikes', \
        # added for fig3 sfs
                        'ISI_CV', \
                        'inv_first_ISI'])
    efel_score_functions = sorted(efel_score_functions1 + efel_score_functions2)
    
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
                        # 'depolarized_base',\
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
                        # # added for fig3 sfs
                        # 'ISI_CV', \
                        # 'inv_first_ISI'])
score_functions = custom_score_functions + efel_score_functions
COMM = MPI.COMM_WORLD
print(COMM.size, "COM SIZE")
for k in range(len(volts_name_list)):
    curr_volts_name = volts_name_list[k]
    curr_stim_name = curr_volts_name.replace('_volts.hdf5', '')
    orig_volts_name = 'orig_'+curr_stim_name
    pin_volts_name = 'pin_'+curr_stim_name
    #pdx_volts_name = 'pdx_'+curr_stim_name
    
    if "hdf5" in curr_volts_name:
        volts = h5py.File(config.volts_path+curr_volts_name, 'r')
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
        
        volt_num = re.findall(r'\d+', orig_volts_name)
        
        if len(volt_num) > 1: # names like 60_2
            volt_num = f'{volt_num[0]}_{volt_num[1]}'
        else: # names like 60
            volt_num = volt_num[0]
            
        curr_function = score_functions[function_ind]
        orig_volts_data = volts[orig_volts_name][:]
        
        if len(orig_volts_data.shape) > 1 and orig_volts_data.shape[1] == config.ntimestep:
            orig_volts_data = orig_volts_data[0]
            
        if prefix == 'pin':
            curr_volts_data = volts[pin_volts_name][volts_ind]
        #elif prefix == 'pdx':
            #curr_volts_data = volts[pdx_volts_name][volts_ind]
        if volts_ind % 1000 == 0:
            print('Working on', prefix, curr_stim_name, get_name(curr_function), str(volts_ind)+'/'+str(n))
        if volt_num+'_dt' in stim_file:
            dt = stim_file[volt_num+'_dt'][:][0]
        score = eval_function(orig_volts_data, curr_volts_data, curr_function, dt)
        # if np.isnan(score):
        #     score = max_score
        # assert score < 1000000
        
        #add
        # assert np.isfinite(score)
        results[(prefix, function_ind, volts_ind)] = score

    results = MPI.COMM_WORLD.gather(results, root=0)

    if COMM.rank == 0:
        flattened_dict = {}
        for d in results:
            k = d.keys()
            for key in k:
                flattened_dict[key] = d[key]

        scores_hdf5 = h5py.File(config.output_path+curr_stim_name+'_scores.hdf5', 'w')
        score_function_names = []
        normalizers = {}
        for i in range(len(score_functions)):
            curr_function_name = get_name(score_functions[i])
            score_function_names.append(np.string_(curr_function_name))
            pin_scores = np.empty((pin_size, 1))
            #pdx_scores = np.empty((pdx_size, 1))
            params_sample_pin_ind = params['sample_ind'][:]
            # params_dx = params['dx'][0]
            free_params_size = params['param_num'][0]
            for j in range(pin_size):
                pin_scores[j] = flattened_dict[('pin', i, j)]
            #for j in range(pdx_size):
                #pdx_scores[j] = flattened_dict[('pdx', i, j)]
            print('Saving', curr_function_name)
            sampled_pin_scores = np.array([pin_scores[p_ind] for p_ind in params_sample_pin_ind])
            sampled_pin_repeat = np.repeat(sampled_pin_scores, free_params_size, axis=0)
            # sensitivity_mat = abs(pdx_scores - sampled_pin_repeat)/params_dx
            # norm_pin_scores, transformation = sn.normalize(pin_scores)
            
            # if not finite (nan or inf) replace with finite max
            nan_mask = (~np.isfinite(pin_scores)) | (np.isnan(pin_scores))
            pin_scores = np.where(nan_mask, np.nanmax(pin_scores[~nan_mask]), pin_scores)
            mm_scaler = MinMaxScaler()
            
            #norm
            norm_pin_scores = mm_scaler.fit_transform(pin_scores)
            
            # if not finite (nan or inf) replace with finite max
            norm_pin_scores = np.where(~np.isfinite(norm_pin_scores), np.nanmax(norm_pin_scores[np.isfinite(norm_pin_scores)]), norm_pin_scores)
            normalizers[curr_function_name] = mm_scaler
            
            assert np.max(norm_pin_scores) < 1.01
            assert np.isfinite(norm_pin_scores).all()
            scores_hdf5.create_dataset('raw_pin_scores_'+curr_function_name, data=pin_scores)
            #scores_hdf5.create_dataset('raw_pdx_scores_'+curr_function_name, data=pdx_scores)
            scores_hdf5.create_dataset('norm_pin_scores_'+curr_function_name, data=norm_pin_scores)
            #scores_hdf5.create_dataset('sensitivity_mat_'+curr_function_name, data=sensitivity_mat)
            # scores_hdf5.create_dataset('transformation_const_'+curr_function_name, data=transformation)
        scores_hdf5.create_dataset('score_function_names', data=score_function_names)
        scores_hdf5.create_dataset('stim_name', data=np.array([np.string_(curr_stim_name)]))
        scores_hdf5.close()
        
        norm_dir = os.path.join(config.output_path, 'normalizers')
        norm_path = os.path.join(norm_dir, f"{curr_stim_name}_normalizers.pkl")
        os.makedirs(norm_dir, exist_ok=True)
        with open(norm_path, 'wb') as f:
            pickle.dump(normalizers,f)
