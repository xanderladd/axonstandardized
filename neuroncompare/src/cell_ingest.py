from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.api.queries.biophysical_api as lib_biophysical_api
import allensdk.api.queries.cell_types_api as lib_cell_api
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import h5py
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import shutil


np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['agg.path.chunksize'] = 10000

# Configure matplotlib plot styling
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

tick_major = 6
tick_minor = 4
plt.rcParams["xtick.major.size"] = tick_major
plt.rcParams["xtick.minor.size"] = tick_minor
plt.rcParams["ytick.major.size"] = tick_major
plt.rcParams["ytick.minor.size"] = tick_minor

font_small = 12
font_medium = 13
font_large = 14
plt.rc('font', size=font_small)          # controls default text sizes
plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_small)    # legend fontsize
plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

def cm_to_in(cm):
    return cm/2.54

def plot_stim_volts_pair(stim, volts, title_stim, title_volts, file_path_to_save=None):
    plt.figure(figsize=(cm_to_in(16), cm_to_in(12)))
    plt.subplot(2, 1, 1)
    plt.title(title_stim)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nA)')
    plt.plot(stim, color='black', linewidth=0.7)
    plt.subplot(2, 1, 2)
    plt.title('Voltage Response '+title_volts)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.plot(volts, color='black')
    plt.tight_layout(pad=1)
    plt.show()
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")
        
def sample(orig_list, final_len, is_stim, sampling_rate):
    rate = int(len(orig_list)/final_len)
    sample_dt = rate*1/sampling_rate*1000
    if not rate:
        return orig_list, sample_dt
    print('dt for sampled stim: '+str(sample_dt)+' ms')
    sampled_stim = []
    for i in range(100, len(orig_list), rate):
        if is_stim:
            in_window_max = max(orig_list[i:i+rate])
            in_window_min = min(orig_list[i:i+rate])
            if in_window_min < 0:
                sampled_stim.append(in_window_min)
            else:
                sampled_stim.append(in_window_max)
        else:
            sampled_stim.append(max(orig_list[i:i+rate]))
    return sampled_stim+[sampled_stim[-1] for i in range(final_len-len(sampled_stim))], sample_dt

def filtr(lis, ind_lis):
    return lis[ind_lis[0]:ind_lis[1]]

def plot_sampled(sweep_number, stimulus, response, sampled_stim, sampled_response):
    plt.figure(figsize=(cm_to_in(40), cm_to_in(30)))
    plt.subplot(3, 2, 1)
    plt.title('Stim number '+str(sweep_number))
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nA)')
    plt.plot(stimulus, color='black', linewidth=0.7)
    plt.subplot(3, 2, 2)
    plt.title('Voltage Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.plot(response, color='black')
    plt.subplot(3, 2, 3)
    plt.title('Stim number '+str(sweep_number))
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nA)')
    plt.plot(sampled_stim, color='black', linewidth=0.7)
    plt.subplot(3, 2, 4)
    plt.title('Voltage Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.plot(sampled_response, color='black')
    plt.tight_layout(pad=1)
    plt.show()
    print('\n \n')

def plot_detailed(sweep_number, stimulus, response):
    plt.figure(figsize=(cm_to_in(40), cm_to_in(12)))
    plt.subplot(1, 3, 1)
    plt.title('Stim number '+str(sweep_number))
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude (nA)')
    plt.plot(stimulus, color='black', linewidth=0.7)
    plt.subplot(1, 3, 2)
    plt.title('Voltage Response')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude (mV)')
    plt.plot(response, color='black')
    plt.subplot(1, 3, 3)
    plt.title('First few steps of response')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude (mV)')
    plt.plot(response[:int(len(response)*0.09)], color='black')

def read_and_plot(data_set, sweep_list, sweeps_to_drop, long_sq_ind, short_sq_tri_81_ind, 
                 short_sq_tri_82_ind, short_sq_tri_83_ind, short_sq_tri_84_ind, 
                 short_sq_tri_85_ind, short_sq_ind, ramp_to_rheo_ind, ramp_ind, 
                 sq_0_5_ind, sq_2_ind, noise_ind_1, noise_ind_2, noise_ind_3, final_len):
    for sweep_number in sweep_list:
        sweep_data = data_set.get_sweep(sweep_number)
        stimulus = sweep_data['stimulus']*10**9
        response = sweep_data['response']*10**3
        sampling_rate = sweep_data['sampling_rate']
        meta_data = data_set.get_sweep_metadata(sweep_number)
        stim_kind = meta_data['aibs_stimulus_name']
        print('Stim kind: '+stim_kind)
        if stim_kind in sweeps_to_drop:
            continue
        if 'Test' in stim_kind:
            print(sweep_number)
        if 'Long Square' in stim_kind:
            stimulus = filtr(stimulus, long_sq_ind)
            response = filtr(response, long_sq_ind)
        if 'Short Square - Triple' in stim_kind:
                if sweep_number == 81:
                    stimulus = filtr(stimulus, short_sq_tri_81_ind)
                    response = filtr(response, short_sq_tri_81_ind)
                if sweep_number == 82:
                    stimulus = filtr(stimulus, short_sq_tri_82_ind)
                    response = filtr(response, short_sq_tri_82_ind)
                if sweep_number == 83:
                    stimulus = filtr(stimulus, short_sq_tri_83_ind)
                    response = filtr(response, short_sq_tri_83_ind)
                if sweep_number == 84:
                    stimulus = filtr(stimulus, short_sq_tri_84_ind)
                    response = filtr(response, short_sq_tri_84_ind)
                if sweep_number == 85:
                    stimulus = filtr(stimulus, short_sq_tri_85_ind)
                    response = filtr(response, short_sq_tri_85_ind)
                else:
                    stimulus = filtr(stimulus, short_sq_tri_82_ind)
                    response = filtr(response, short_sq_tri_82_ind)
                    
        if 'Short Square' in stim_kind and not ' - Triple' in stim_kind:
            stimulus = filtr(stimulus, short_sq_ind)
            response = filtr(response, short_sq_ind)
        if 'Ramp to Rheobase' in stim_kind:
            stimulus = filtr(stimulus, ramp_to_rheo_ind)
            response = filtr(response, ramp_to_rheo_ind)
        if 'Ramp' in stim_kind and not 'Rheobase' in stim_kind:
            stimulus = filtr(stimulus, ramp_ind)
            response = filtr(response, ramp_ind)
        if 'Square - 0.5ms Subthreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_0_5_ind)
            response = filtr(response, sq_0_5_ind)
        if 'Square - 2s Suprathreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_2_ind)
            response = filtr(response, sq_2_ind)
        if not 'Noise' in stim_kind and not 'Test' in stim_kind:
            sampled_stim, sample_dt = sample(stimulus, final_len, True, sampling_rate)
            sampled_response, sample_dt = sample(response, final_len, False, sampling_rate)
            plot_sampled(sweep_number, stimulus, response, sampled_stim, sampled_response)
        if 'Noise' in stim_kind:
            stimulus1 = filtr(stimulus, noise_ind_1)
            response1 = filtr(response, noise_ind_1)
            stimulus2 = filtr(stimulus, noise_ind_2)
            response2 = filtr(response, noise_ind_2)
            stimulus3 = filtr(stimulus, noise_ind_3)
            response3 = filtr(response, noise_ind_3)
            sampled_stim1, sample_dt1 = sample(stimulus1, final_len, True, sampling_rate)
            sampled_response1, sample_dt1 = sample(response1, final_len, False, sampling_rate)
            sampled_stim2, sample_dt2 = sample(stimulus2, final_len, True, sampling_rate)
            sampled_response2, sample_dt2 = sample(response2, final_len, False, sampling_rate)
            sampled_stim3, sample_dt3 = sample(stimulus3, final_len, True, sampling_rate)
            sampled_response3, sample_dt3 = sample(response3, final_len, False, sampling_rate)
            plot_sampled(sweep_number, stimulus1, response1, sampled_stim1, sampled_response1)
            plot_sampled(sweep_number, stimulus2, response2, sampled_stim2, sampled_response2)
            plot_sampled(sweep_number, stimulus3, response3, sampled_stim3, sampled_response3)


def process_and_save_data(data_set, cell_id, inputs, sweeps_to_keep, sweeps_to_drop, 
                        long_sq_ind, short_sq_tri_81_ind, short_sq_tri_82_ind, short_sq_tri_83_ind, 
                        short_sq_tri_84_ind, short_sq_tri_85_ind, short_sq_ind, ramp_to_rheo_ind, 
                        ramp_ind, sq_0_5_ind, sq_2_ind, noise_ind_1, noise_ind_2, noise_ind_3, 
                        full_noise_ind, final_len):
    # Create directories if they don't exist
    os.makedirs(f'{inputs["data_dir"]}/stims', exist_ok=True)
    os.makedirs(f'{inputs["data_dir"]}/target_volts', exist_ok=True)
    
    # Define file paths
    stim_file_path = f'{inputs["data_dir"]}/stims/allen_data_stims_{cell_id}.hdf5'
    volts_file_path = f'{inputs["data_dir"]}/target_volts/allen_data_target_volts_{cell_id}.hdf5'
           
    # Remove existing files if they exist
    if os.path.isfile(stim_file_path):
        os.remove(stim_file_path)
        
    if os.path.isfile(volts_file_path):
        os.remove(volts_file_path)
    
    # Create HDF5 files    
    stims_hdf5 = h5py.File(stim_file_path, 'w')
    volts_hdf5 = h5py.File(volts_file_path, 'w')
    keys = []
    stim_kinds = []
    
    sweep_numbers = sorted(data_set.get_experiment_sweep_numbers())
    
    for sweep_number in sweeps_to_keep:
        sweep_data = data_set.get_sweep(sweep_number)
        stimulus = sweep_data['stimulus']*10**9
        response = sweep_data['response']*10**3
        sampling_rate = sweep_data['sampling_rate']
        meta_data = data_set.get_sweep_metadata(sweep_number)
        stim_kind = meta_data['aibs_stimulus_name']
        if type(stim_kind) == bytes: 
            stim_kind = stim_kind.decode('ASCII')
        print('Stim kind: '+ stim_kind)
            
        if stim_kind in sweeps_to_drop or len(response) == 0:
            if sweep_number in sweep_numbers:
                sweep_numbers.remove(int(sweep_number))
            continue
            
        if 'Test' in stim_kind:
            print(sweep_number)
            
        # Apply filters based on stimulus type
        if 'Short Square - Triple' in stim_kind:
            if sweep_number == 81:
                stimulus = filtr(stimulus, short_sq_tri_81_ind)
                response = filtr(response, short_sq_tri_81_ind)
            elif sweep_number == 82:
                stimulus = filtr(stimulus, short_sq_tri_82_ind)
                response = filtr(response, short_sq_tri_82_ind)
            elif sweep_number == 83:
                stimulus = filtr(stimulus, short_sq_tri_83_ind)
                response = filtr(response, short_sq_tri_83_ind)
            elif sweep_number == 84:
                stimulus = filtr(stimulus, short_sq_tri_84_ind)
                response = filtr(response, short_sq_tri_84_ind)
            elif sweep_number == 85:
                stimulus = filtr(stimulus, short_sq_tri_85_ind)
                response = filtr(response, short_sq_tri_85_ind)
            else:
                stimulus = filtr(stimulus, short_sq_tri_82_ind)
                response = filtr(response, short_sq_tri_82_ind)
                
        if 'Short Square' in stim_kind and not '- Triple' in stim_kind:
            stimulus = filtr(stimulus, short_sq_ind)
            response = filtr(response, short_sq_ind)
            
        if 'Long Square' in stim_kind:
            stimulus = filtr(stimulus, long_sq_ind)
            response = filtr(response, long_sq_ind)
            
        if 'Ramp to Rheobase' in stim_kind:
            stimulus = filtr(stimulus, ramp_to_rheo_ind)
            response = filtr(response, ramp_to_rheo_ind)
            
        if 'Ramp' in stim_kind and not 'Rheobase' in stim_kind:
            stimulus = filtr(stimulus, ramp_ind)
            response = filtr(response, ramp_ind)
            
        if 'Square - 0.5ms Subthreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_0_5_ind)
            response = filtr(response, sq_0_5_ind)
            
        if 'Square - 2s Suprathreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_2_ind)
            response = filtr(response, sq_2_ind)
            
        # Process non-noise stimuli
        if not 'Noise' in stim_kind and not 'Test' in stim_kind:
            sampled_stim, sample_dt_stim = sample(stimulus, final_len, True, sampling_rate)
            sampled_response, sample_dt = sample(response, final_len, False, sampling_rate)
            plot_sampled(sweep_number, stimulus, response, sampled_stim, sampled_response)
            
            try:
                assert len(sampled_response) > 1
                assert len(sampled_stim) > 1
                assert sample_dt_stim == sample_dt
            except:
                print(f"Error with sweep {sweep_number}: sampled data issue")
                continue
                
            stims_hdf5.create_dataset(str(sweep_number), data=sampled_stim)
            stims_hdf5.create_dataset(str(sweep_number)+'_dt', data=np.array([sample_dt]))
            volts_hdf5.create_dataset(str(sweep_number), data=sampled_response)
            volts_hdf5.create_dataset(str(sweep_number)+'_dt', data=np.array([sample_dt]))
            keys.append(str(sweep_number))
            stim_kinds.append(stim_kind)
            
        # Process noise stimuli
        if 'Noise' in stim_kind:
            stimulus1 = filtr(stimulus, noise_ind_1)
            response1 = filtr(response, noise_ind_1)
            stimulus2 = filtr(stimulus, noise_ind_2)
            response2 = filtr(response, noise_ind_2)
            stimulus3 = filtr(stimulus, noise_ind_3)
            response3 = filtr(response, noise_ind_3)
            
            sampled_stim1, sample_dt1 = sample(stimulus1, final_len, True, sampling_rate)
            sampled_response1, sample_dt1 = sample(response1, final_len, False, sampling_rate)
            sampled_stim2, sample_dt2 = sample(stimulus2, final_len, True, sampling_rate)
            sampled_response2, sample_dt2 = sample(response2, final_len, False, sampling_rate)
            sampled_stim3, sample_dt3 = sample(stimulus3, final_len, True, sampling_rate)
            sampled_response3, sample_dt3 = sample(response3, final_len, False, sampling_rate)
            
            plot_sampled(sweep_number, stimulus1, response1, sampled_stim1, sampled_response1)
            plot_sampled(sweep_number, stimulus2, response2, sampled_stim2, sampled_response2)
            plot_sampled(sweep_number, stimulus3, response3, sampled_stim3, sampled_response3)
            
            stims_hdf5.create_dataset(str(sweep_number)+'1', data=sampled_stim1)
            stims_hdf5.create_dataset(str(sweep_number)+'1_dt', data=np.array([sample_dt1]))
            volts_hdf5.create_dataset(str(sweep_number)+'1', data=sampled_response1)
            volts_hdf5.create_dataset(str(sweep_number)+'1_dt', data=np.array([sample_dt1]))
            
            stims_hdf5.create_dataset(str(sweep_number)+'2', data=sampled_stim2)
            stims_hdf5.create_dataset(str(sweep_number)+'2_dt', data=np.array([sample_dt2]))
            volts_hdf5.create_dataset(str(sweep_number)+'2', data=sampled_response2)
            volts_hdf5.create_dataset(str(sweep_number)+'2_dt', data=np.array([sample_dt2]))
            
            stims_hdf5.create_dataset(str(sweep_number)+'3', data=sampled_stim3)
            stims_hdf5.create_dataset(str(sweep_number)+'3_dt', data=np.array([sample_dt3]))
            volts_hdf5.create_dataset(str(sweep_number)+'3', data=sampled_response3)
            volts_hdf5.create_dataset(str(sweep_number)+'3_dt', data=np.array([sample_dt3]))
            
            keys.append(str(sweep_number)+'1')
            keys.append(str(sweep_number)+'2')
            keys.append(str(sweep_number)+'3')
            stim_kinds.append(stim_kind+'1')
            stim_kinds.append(stim_kind+'2')
            stim_kinds.append(stim_kind+'3')
    
    # Save metadata to HDF5 files
    stims_hdf5.create_dataset('sweep_numbers', data=sweep_numbers)
    stims_hdf5.create_dataset('sweep_keys', data=np.string_(keys))
    volts_hdf5.create_dataset('sweep_numbers', data=sweep_numbers)
    volts_hdf5.create_dataset('sweep_keys', data=np.string_(keys))
    volts_hdf5.create_dataset('stim_types', data=np.string_(stim_kinds))
    
    # Close HDF5 files
    stims_hdf5.close()
    volts_hdf5.close()


def check_and_remove_fn(fn, force=False):
    if os.path.isfile(fn) and force:
        os.remove(fn)
        return 0
    elif os.path.isfile(fn):
        return 1
    else:
        return 0

def get_target_volts(target_path, model_number):
    target_file_str = target_path+'allen_data_target_volts_{}.hdf5'
    target_volts = h5py.File(target_file_str.format(model_number), 'r')
    return target_volts

def get_stims(stims_path, model_number):
    stims_file_str = stims_path+'allen_data_stims_{}.hdf5'
    stims = h5py.File(stims_file_str.format(model_number), 'r')
    return stims
    
def get_sweep_keys(stims):
    sweep_keys = [e.decode('ascii') for e in stims['sweep_keys']] 
    return sweep_keys
    
def plot_sweep_keys(sweep_keys,stims,sweep_filter):
    for key in sweep_keys:
        if not key in sweep_filter:
            print("Stim sweep number", key)
            print("Max val:", np.max(stims[key]))
            print("Min val:", np.min(stims[key]))
            plt.plot(stims[key])
            plt.show()
            
def match_sweeps(stim,stims_to_match, sweep_keys, sweeep_index_to_match, timesteps, allow_dup = False, verbose=True, show=True):
    """
    Args:
        stim: stim we are going to match
        stims_to_match: reference to match with
        sweep_keys: sweep keys of stim we have
        sweeep_index_to_match: ?
        show: plot to check?
    """
    matched_sweep_numbers = []
    sweep_map_to_original = []
    seen = []
    for stim_num in sweep_index_to_match:
        match_stim = stims_to_match[stim_num]
        if verbose:
            print("Original sweep number:", stim_num)
        if show:
            plt.plot(match_stim, color='red')
            plt.show()
        for key in sweep_keys:
            if key in seen and not allow_dup:
                continue
            if not key in sweep_filter_1+sweep_filter_2:
                curr_stim = stims[key]
                len_to_div = len(curr_stim) / timesteps
                curr_start = np.argmax(np.array(curr_stim) > 0) / len_to_div
                match_start = np.argmax(np.array(match_stim) > 0) 
                same_start = np.abs(curr_start - match_start) <  300
                diff_max = abs(np.max(match_stim)-np.max(curr_stim))
                diff_min = abs(np.min(match_stim)-np.min(curr_stim))
                #if diff_max < 0.0005 and diff_min < 0.0005:
                if diff_max < 0.0005 and diff_min < 0.0005:
                    if verbose:
                        print("Matched sweep number:", key)
                    matched_sweep_numbers.append(key)
                    sweep_map_to_original.append(stim_num)
                    seen.append(key)
                    if show:
                        plt.plot(curr_stim)
                        plt.show()
    return matched_sweep_numbers, sweep_map_to_original

# def sample(input_vec, target_len, curr_dt=None):
#     vec_len = len(input_vec)
#     scale_factor = int(vec_len/target_len)
#     dt =  (vec_len/target_len) * curr_dt
#     sampled_vec = []
#     scale_factor = max(scale_factor,1)
#     for i in range(0, vec_len, scale_factor):
#         window = input_vec[i:i+scale_factor]
#         sampled_vec.append(np.max(window))
#     return sampled_vec, dt

def extend(input_vec, target_len, curr_dt=None):
    vec_len = len(input_vec)
    extend_len = target_len - len(input_vec)
    assert extend_len > 0
    extend_vec = np.append(input_vec, np.repeat(input_vec[-1], extend_len))
  
    return extend_vec

def downsample(stims, target_volts, matched_sweep_numbers, resolution=10000, show=True):
    volt_res = {}
    stim_res = {}
    dts = {}
    for sweep_num in matched_sweep_numbers:
        curr_volt = target_volts[sweep_num][:]
        curr_stim = stims[sweep_num][:]
        dts[sweep_num] = resolution/len(curr_volt)
        curr_volt_sampled,_ = sample(curr_volt, resolution, dts[sweep_num])#target_volts['{}_dt'.format(sweep_num)][0])
        curr_stim_sampled,_ = sample(curr_stim, resolution, dts[sweep_num])#target_volts['{}_dt'.format(sweep_num)][0])
        assert False, "fix this function"
        if len(curr_stim_sampled) != 10000:
            # I don't think this is an issue, since nothing important is getting cutoff
            # see comment
            #print("Warning...stim for {} is not len 10k, .... truncating (TODO: Recalc DT)".format(sweep_num))
            #print("truncated by : ", len(curr_stim_sampled) - 10000)
            if len(curr_stim_sampled) - 10000 > 1000 and show:
                plt.figure()
                plt.plot(curr_stim_sampled)
                plt.plot(curr_stim_sampled[:10000])
            curr_stim_sampled = curr_stim_sampled[:10000]
            curr_volt_sampled = curr_volt_sampled[:10000]
            
        volt_res[sweep_num] = curr_volt_sampled
        stim_res[sweep_num] = curr_stim_sampled
        assert len(curr_stim_sampled) == 10000
        if show:
            plt.plot(curr_volt)
            plt.show()
            plt.plot(curr_volt_sampled, color='red')
            #plt.plot(curr_stim_sampled, color='green')

            plt.show()
    return volt_res, stim_res, dts

        
#### MAIN FUNCTIONALITIES ##########

def check_stims(model_number, timesteps):
    f = h5py.File(f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}.hdf5", "r")
    all_keys = list(f.keys())
    sweep_keys = [key.decode('ascii') for key in f['sweep_keys'][:]]
    for key in sweep_keys:
        assert "{}_dt".format(key) in all_keys
        #print(int(f["{}_dt".format(key)]))
        assert f["{}_dt".format(key)][0] < 1, "Dt is weird"
        assert len(f[key][:]) == timesteps, "stim is wrong length:  {}".format(len(f[key][:]))
        
def check_volts(model_number, timesteps):
    f = h5py.File(f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}.hdf5", "r")
    for key in f.keys():
        print(f[key][0], "Start volt")
        assert len(f[key][:]) == timesteps, "Voltage is wrong length: {}".format(len(f[key][:]))

        
def save_results_hdf5(model_number, stims, target_volts, dts, \
                      sweep_index_to_match, matched_sweep_numbers, \
                      sweep_map_to_original, timesteps):
    if not os.path.isdir(f"{inputs['data_dir']}/results/{model_number}"):
        os.mkdir(f"{inputs['data_dir']}/results/{model_number}")
    if os.path.isfile(f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}.hdf5"):
        os.remove(f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}.hdf5")
    if os.path.isfile(f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}.hdf5"):
        os.remove(f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}.hdf5")
    stim_file = h5py.File(f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}.hdf5", "w")
    target_file = h5py.File(f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}.hdf5", "w")
    correspondance = []
    seen = []
    skipped =0
    ct = 0
    for sweep_num,mapped_sweep in zip(matched_sweep_numbers, sweep_map_to_original):
        assert len(stims[sweep_num]) == timesteps, "not saving stim, it's wrong len"
        ct += 1
        if sweep_num in seen:
            skipped += 1
            continue
        else:
            seen.append(sweep_num)
        stim_file.create_dataset(sweep_num,data= stims[sweep_num])
        stim_file.create_dataset("{}_dt".format(sweep_num),data=[dts[sweep_num]])
        target_file.create_dataset(sweep_num, data=target_volts[str(sweep_num)])
        correspondance.append(mapped_sweep)
    correspondance = [n.encode("ascii", "ignore") for n in correspondance]
    matched_sweep_numbers = [n.encode("ascii", "ignore") for n in matched_sweep_numbers]
    matched_sweep_numbers_int = [int(num) for num in matched_sweep_numbers]
    stim_file.create_dataset("corresponding_original", data=correspondance)
    stim_file.create_dataset("sweep_keys", data=matched_sweep_numbers)
    stim_file.create_dataset("sweep_nums", data=matched_sweep_numbers_int)

    target_file.close()
    target_file.close()
    print("then SKIPPED", skipped, " and saved: ", ct)
    print("---------- Saved data for {}  ---------".format(model_number))
    print("--------------------------------------")
    
        
def match_sweep_and_save(model_numbers):
    for model_number in model_numbers:
        target_volts = get_target_volts(target_path, model_number )
        stims = get_stims(stims_path,model_number)
        sweep_keys = get_sweep_keys(stims)
        matched_sweep_numbers, sweep_map_to_original = match_sweeps(stims, stims_to_match, sweep_keys, \
                                             sweep_index_to_match, allow_dup=False, verbose=False,show=False)
        print("--------------------------------------")
        filler = sweep_map_to_original[-1]
        sweep_map_to_original.extend([filler,filler,filler,filler,filler])
        sweepNum2volt, sweepNum2Stim, dts = downsample(stims, target_volts, matched_sweep_numbers, show=False)

        save_results_hdf5(model_number, sweepNum2Stim, sweepNum2volt, dts, \
                          sweep_index_to_match, matched_sweep_numbers, sweep_map_to_original)
        
    

def select_stims(all_target_volts, all_stims, num_stims=0, passive=False):
    all_viable = []
    if num_stims == 0:
        num_stims = len(list(all_target_volts.keys()))
    for key in all_target_volts.keys():
        if "stim_types" in key: continue
        curr_stim = all_stims[key]
        curr_targV = all_target_volts[key]
        if "dt" not in key and 'sweep' not in key and '63' not in key:
            try:
                currently_passive = np.max(curr_targV) < 0 
            except:
                import pdb; pdb.set_trace()
            # or (not passive and currently_passive)
            if (passive and not currently_passive)  \
             or np.allclose(curr_stim,0) or key in all_viable:
                continue
            else:
                all_viable.append(key)
    all_viable = np.unique(all_viable)
    choices = np.random.choice(all_viable, min(len(all_viable),num_stims), replace=False)
    choices = np.unique(choices)
    return choices

def save_stims(inputs, stims_path, target_path, model_number, passive, timesteps, show=False, pdf=None, force=False):
    all_target_volts = get_target_volts(target_path, model_number )
    all_stims = get_stims(stims_path,model_number)
    # TODO: store these files in var and delete them in case they exist
    if passive:
        stim_path = f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}_passive.hdf5"
        volt_path = f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}_passive.hdf5"
        obj_path = f"{inputs['data_dir']}/results/{model_number}/allen{model_number}_objectives_passive.hdf5"
    else:
        stim_path = f"{inputs['data_dir']}/results/{model_number}/stims_{model_number}.hdf5"
        volt_path = f"{inputs['data_dir']}/results/{model_number}/target_volts_{model_number}.hdf5"
        obj_path = f"{inputs['data_dir']}/results/{model_number}/allen{model_number}_objectives.hdf5"
        
    e1, e2, e3 = check_and_remove_fn(stim_path, force), check_and_remove_fn(volt_path, force), check_and_remove_fn(obj_path, force)
    
    
    if not (e1 == 0 and e2 == 0 and e3 == 0):
        print("Attempting to overwrite files without force set to true {}".format(stim_path))
        return

    stim_f = h5py.File(stim_path, "w")
    volt_f = h5py.File(volt_path, "w")
    new_obj_f = h5py.File(obj_path, "w")
    opt_stim_name_list = []
    stim_types = []
    # TODO: move filter to be right before choosing and only here
    # you should filter passive here as well
    stim_names = select_stims(all_target_volts, all_stims, num_stims=0, passive=passive)
    for key in stim_names:
        print("processing :", key)
        curr_stim = all_stims[key]
        stim_idx = np.where(stim_names == key)[0][0]
        stim_type = all_target_volts['stim_types'][stim_idx]
        curr_targV = all_target_volts[key]
        curr_dt = all_stims[key+'_dt'][0]
        prev_len = len(curr_stim)
        if curr_dt > 1:
            print(f'skipped : {key} because dt is {dt}')
            continue
        if len(curr_stim) > timesteps + 500 : # allow for the stim to be a little longer
            curr_stim,dt = sample(curr_stim, timesteps, curr_dt = curr_dt)
            curr_targV,_ = sample(curr_targV, timesteps, curr_dt = curr_dt) 
        elif timesteps > len(curr_stim):
            curr_stim = extend(curr_stim, timesteps, curr_dt = curr_dt)
            curr_targV = extend(curr_targV, timesteps, curr_dt = curr_dt) # add to targv 
            dt = curr_dt # dt stays the same
            
        else:
            dt = all_stims[key+'_dt'][:][0]
            print(f'dt : {dt}') 
        
        # cut off from front!
        cutoff = max(0,len(curr_targV) - timesteps )
        curr_targV = curr_targV[cutoff:]
        curr_stim = curr_stim[cutoff:]
        # dt doesn't give us exact cut off, this is regrettable
        print(f'stim: {key}, start dt: {curr_dt}, end dt: {dt}, prev len: {prev_len}, curr len: {len(curr_stim)}"')
        stim_types.append(stim_type)
        stim_f.create_dataset(key, data=curr_stim)
        stim_f.create_dataset(key+"_dt", data=[dt])
        volt_f.create_dataset(key, data=curr_targV)
        opt_stim_name_list.append(key)

        if show:
            fig = plt.figure()
            plt.title("added targ v " + key + " with DT " + str(dt))
            plt.plot(np.array(all_target_volts[key]))
            
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)
                
            fig = plt.figure()
            plt.plot(curr_targV)
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)
            
            fig = plt.figure()
            plt.plot(curr_stim)
            plt.title(key + " stim")
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)

            
            

    new_weights = np.ones(3000)
    print(opt_stim_name_list, 'opt sim name')
    dt = h5py.special_dtype(vlen=str) 
    if passive:
        opt_stim_name_list = np.array(opt_stim_name_list, dtype=dt) 
        new_obj_f.create_dataset('opt_stim_name_list', data=opt_stim_name_list)
        new_obj_f.create_dataset('opt_weight_list', data=new_weights)
    #     new_obj_f.create_dataset('ordered_score_function_list', data=obj_f['ordered_score_function_list'])
        new_obj_f.create_dataset('ordered_score_function_list', data=[b'chi_square_normal'])

    new_obj_f.close()
    volt_f.close()
    stim_f.create_dataset('stim_types', data=stim_types)
    stim_f.close()
    
    print(len(opt_stim_name_list), "stims")


def copy_files_to_run(model_num, passive, dest):
    """
    Copy model files to the run directory
    
    Args:
        model_num: Model number
        passive: Whether to use passive stims ('True' or 'False')
        dest: Destination directory
    """
    print(f"Copying files for model {model_num} to {dest}")
    print(f"Passive mode: {passive}")
    
    # Read input.txt to get data_dir
    data_dir = None
    with open("./input.txt", "r") as input_file:
        for line in input_file:
            if "=" in line:
                name, value = line.strip().split("=", 1)
                if name.strip() == "data_dir":
                    data_dir = value.strip()
                    break
    
    if not data_dir:
        raise ValueError("Could not find data_dir in input.txt")
    
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(dest, "target_volts"), exist_ok=True)
    os.makedirs(os.path.join(dest, "stims"), exist_ok=True)
    os.makedirs(os.path.join(dest, "objectives"), exist_ok=True)
    
    # Copy files based on passive flag
    if passive == "True":
        # Copy target volts
        src = f"{data_dir}/results/{model_num}/target_volts_{model_num}_passive.hdf5"
        dst = f"{dest}/target_volts/target_volts_{model_num}_passive.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy stims to run directory
        src = f"{data_dir}/results/{model_num}/stims_{model_num}_passive.hdf5"
        dst = f"{dest}/stims/stims_{model_num}_passive.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy stims to main stims directory
        dst = f"{data_dir}/stims/stims_{model_num}_passive.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy objectives
        src = f"{data_dir}/results/{model_num}/allen{model_num}_objectives_passive.hdf5"
        dst = f"{dest}/objectives/allen{model_num}_objectives_passive.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
    else:
        # Copy target volts
        src = f"{data_dir}/results/{model_num}/target_volts_{model_num}.hdf5"
        dst = f"{dest}/target_volts/target_volts_{model_num}.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy stims to run directory
        src = f"{data_dir}/results/{model_num}/stims_{model_num}.hdf5"
        dst = f"{dest}/stims/stims_{model_num}.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy stims to main stims directory
        dst = f"{data_dir}/stims/stims_{model_num}.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
        
        # Copy objectives
        src = f"{data_dir}/results/{model_num}/allen{model_num}_objectives.hdf5"
        dst = f"{dest}/objectives/allen{model_num}_objectives.hdf5"
        print(f"Copying {src} to {dst}")
        shutil.copy2(src, dst)
    
    print("Files copied successfully")
    
def read_inputfile(input_path=''):
    if not len(input_path):
        input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../input.txt')
    # Get parameters from input file
    input_file = open(input_path, "r")
    inputs = {}
    input_lines = input_file.readlines()
    for line in input_lines:
        vals = line.split("=")
        if len(vals) != 2 and "\n" not in vals:
            raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
        if "\n" not in vals:
            inputs[vals[0]] = vals[1][:len(vals[1])-1]
    return inputs
    
def main():
    # Create the main parser
    parser = argparse.ArgumentParser(description='Allen SDK Data Processing')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create subparser for data download and processing
    process_parser = subparsers.add_parser('pull', help='Download and process cell data')
    process_parser.add_argument('--cell_id', type=int, required=True, help='Cell ID to process')
    
    # Create subparser for stim processing
    stim_parser = subparsers.add_parser('assemble', help='Filter and save stimulus data')
    stim_parser.add_argument('--model', type=int, required=True, help='Model number')
    stim_parser.add_argument('--timesteps', type=int, required=True, help='Number of timesteps to use')
    stim_parser.add_argument('--passive', action="store_true", help='Process only passive stims')
    stim_parser.add_argument('--pdf', action="store_true", help='Save plots to PDF')
    stim_parser.add_argument('--force', action="store_true", help='Force overwrite existing files')

    # Create subparser for copying files to run directory
    copy_parser = subparsers.add_parser('copy_to_run', help='Copy files to run directory')
    copy_parser.add_argument('--model', type=int, required=True, help='Model number')
    copy_parser.add_argument('--passive', type=str, choices=['True', 'False'], default='False', 
                            help='Use passive stims (True/False)')
    copy_parser.add_argument('--dest', type=str, required=True, 
                            help='Destination directory (e.g., runs/model_peeling_date_custom)')

    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.command == 'pull':
        pull_cell_data(args)
    elif args.command == 'assemble':
        save_stim_data(args)
    elif args.command == 'copy_to_run':
        copy_files_to_run(args.model, args.passive, args.dest)
    else:
        parser.print_help()

        
def pull_cell_data(args):
    inputs = read_inputfile()
    # Initialize Allen SDK APIs
    biophysical_api = lib_biophysical_api.BiophysicalApi()
    cell_api = lib_cell_api.CellTypesApi()
    
    # Get cell data
    cell_info = cell_api.list_cells_api()
    cell_df = pd.DataFrame(cell_info)
    
    # Define selection criteria
    selection_criteria = (cell_df['donor__species'] == 'Mus musculus') & \
                         (cell_df['structure__name'] == '"Primary visual area, layer 5"') & \
                         (cell_df['ef__f_i_curve_slope'] > .2)  & \
                         (cell_df['m__biophys'] == 1) 
    
    # Use the cell_id from arguments
    cell_id = args.cell_id 
    
    # Create directories if they don't exist
    os.makedirs(f'{inputs["data_dir"]}/nwb_files', exist_ok=True)
    
    # Download and save the cell's electrophysiology data
    cell_api.save_ephys_data(cell_id, f"{inputs['data_dir']}/nwb_files/{cell_id}.nwb")
    
    # Load the NWB file
    data_set = NwbDataSet(f"{inputs['data_dir']}/nwb_files/{cell_id}.nwb")
    
    # Define indices for different stimulus types
    initial_ind = 150000
    final_len = 30000
    
    short_sq_ind = [150000, 270000]
    short_sq_tri_81_ind = [360000, 480000]
    short_sq_tri_82_ind = [370000, 460000]
    short_sq_tri_83_ind = [370000, 460000]
    short_sq_tri_84_ind = [380000, 440000]
    short_sq_tri_85_ind = [380000, 440000]
    long_sq_ind = [120000, 480000]
    ramp_ind = [110000, 6200000]
    ramp_to_rheo_ind = [initial_ind, 4550000]
    noise_ind_1 = [initial_ind, 1300000]
    noise_ind_2 = [1800000, 2800000]
    noise_ind_3 = [3400000, 4400000]
    full_noise_ind = [initial_ind, 4400000]
    sq_0_5_ind = [initial_ind, 450000]
    sq_2_ind = [130000, 700000]
    
    # Define sweeps to drop
    long_square_drop = [49, 40, 51, 52, 64]
    noise_1_drop = [55, 57, 59]
    noise_2_drop = [58, 60]
    ramp_drop = [5, 6]
    ramp_to_rheobase_drop = [102, 103]
    short_square_drop = [22, 23, 24, 25, 78]
    short_square_triple_drop = [79, 80, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    sq_0_5ms_drop = [63]
    sq_2s_drop = [66, 68, 69, 72, 73, 75, 76, 77]
    
    sweeps_to_drop = ['noise_1_drop', 'noise_2_drop', 'long_square_drop', 'Ramp to Rheobase',
                      'ramp_to_rheobase_drop', 'short_square_drop', 'short_square_triple_drop' + 'sq_0_5ms_drop', 'sq_2s_drop']
    
    # Get list of sweeps to keep
    sweep_numbers = sorted(data_set.get_experiment_sweep_numbers())
    sweeps_to_keep = []
    for sweep in sweep_numbers:
        sweeps_to_keep.append(sweep)
    
    # Print sweep information
    sweep_data = data_set.get_sweep(sweep_numbers[0])
    sampling_rate = sweep_data['sampling_rate']
    print(f"Sampling interval: {1/sampling_rate * 1000} ms")
    
    # Process and save the data
    process_and_save_data(data_set, cell_id, inputs, sweeps_to_keep, sweeps_to_drop, 
                        long_sq_ind, short_sq_tri_81_ind, short_sq_tri_82_ind, short_sq_tri_83_ind, 
                        short_sq_tri_84_ind, short_sq_tri_85_ind, short_sq_ind, ramp_to_rheo_ind, 
                        ramp_ind, sq_0_5_ind, sq_2_ind, noise_ind_1, noise_ind_2, noise_ind_3, 
                        full_noise_ind, final_len)
    
    # Display a random sweep for verification
    sweeps = data_set.get_sweep_numbers()
    sweep = np.random.choice(sweeps)
    voltage_trace = data_set.get_sweep(sweep)
    start, end = voltage_trace['index_range'][0], 150000 + 290800
    stim = voltage_trace['stimulus'][start:end]
    stim_unit = voltage_trace['stimulus_unit']
    response = voltage_trace['response'][start:end] * 1000  # volts to mV
    dt = 1 / voltage_trace['sampling_rate']
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title(f"Random sweep {sweep} - Stimulus")
    plt.plot(stim)
    plt.subplot(2, 1, 2)
    plt.title("Voltage Response")
    plt.plot(response)
    plt.tight_layout()
    plt.show()


def save_stim_data(args):
    inputs = read_inputfile()
    stims_path = f'{inputs["data_dir"]}/stims/'
    target_path = f'{inputs["data_dir"]}/target_volts/'
    sweep_filter_1 = [str(e) for e in range(79, 100)]
    sweep_filter_2 = ['101', '102', '103']
    sweep_filter = sweep_filter_1 + sweep_filter_2
    pdf = None
    os.makedirs(os.path.join(f'{inputs["data_dir"]}/results', str(args.model)), exist_ok=True)
    if args.pdf:
        if args.passive:
            pdf = PdfPages(os.path.join(f'{inputs["data_dir"]}/results', str(args.model), 'passive_stims.pdf'))
        else:
            pdf = PdfPages(os.path.join(f'{inputs["data_dir"]}/results', str(args.model), 'full_stims.pdf'))

    save_stims(inputs, stims_path, target_path, args.model, passive=args.passive, timesteps=args.timesteps, force=args.force, show=True, pdf=pdf)
    
    if args.pdf:
        pdf.close()



if __name__ == "__main__":
    main()