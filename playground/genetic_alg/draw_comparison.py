from allensdk.core.nwb_data_set import NwbDataSet
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py 
import os
from matplotlib.backends.backend_pdf import PdfPages
import pickle as pkl
import cfg

if not os.path.isfile('compare_response2.hdf5'):
    f = h5py.File('compare_response2.hdf5', 'w')
    for sweep in os.listdir('compare_responses'):
        curr_response = np.genfromtxt(f'compare_responses/{sweep}', delimiter=',')
        sweep = sweep.replace('.csv','')

        print(sweep)
        f.create_dataset(name=f'{sweep}_response', data = curr_response)
        f.create_dataset(name=f'{sweep}_dt', data = 1)
    f.close()



np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['agg.path.chunksize'] = 10000

# if you ran the examples above, you will have a NWB file here
cell_file_name = f'./allen_model/{cfg.model_num}_ephys.nwb'
cell_data_set = NwbDataSet(cell_file_name)


allen_file_name = [os.path.join('./allen_model/work', file) for file in os.listdir('./allen_model/work')][0]
allen_data_set = NwbDataSet(allen_file_name)

sota_file_name = './compare_response2.hdf5'
sota_data_set = h5py.File(sota_file_name, 'r')

sweep_numbers = sorted(cell_data_set.get_experiment_sweep_numbers())
# # sampling rate is in Hz
#sampling_rate = sweep_data['sampling_rate']
#print(sampling_rate)


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
        
        
import h5py

initial_ind = 150000

short_sq_ind = [170000, 240000]
long_sq_ind = [initial_ind, 460000]
ramp_ind = [initial_ind, 1710000]
noise_ind_1 = [initial_ind, 1200000]
noise_ind_2 = [1900000, 2800000]
noise_ind_3 = [3500000, 4400000]
sq_0_5_ind = [initial_ind, 450000]
sq_2_ind = [initial_ind, 700000]

def filtr(lis, ind_lis):
    return lis[ind_lis[0]:ind_lis[1]]

def plot_sampled(sweep_number, stimulus, cell_response, allen_response, sota_response, dt):
    allen_response = allen_response
    sota_response = sota_response
    fig = plt.figure(figsize=(cm_to_in(40), cm_to_in(10)))
    plt.subplot(1, 3, 1)
    plt.title('Stim number '+str(sweep_number))
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nA)')
    plt.plot(np.arange(len(stimulus))*dt, stimulus, color='black', linewidth=0.7)
    plt.subplot(1, 3, 2)
    plt.title('Allen\'s Model Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.plot(np.arange(len(stimulus))*dt, cell_response, color='black')
    plt.plot(np.arange(len(stimulus))*dt, allen_response, color='crimson')
    plt.subplot(1, 3, 3)
    plt.title('CoMParE Model Response')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    plt.plot(np.arange(len(stimulus))*dt, cell_response, color='black')
    plt.plot(np.arange(len(stimulus))*dt, sota_response, color='crimson')
    plt.tight_layout(pad=1)
    plt.show()
    print('\n \n')
    return fig

def adjust_v_init_failure(response, target_reponse):
    diff = response[0] - target_reponse[0]
    return response - diff


def read_and_plot(sweep_list, save_path=None, pdf=None):
    if save_path:
        if os.path.isfile(save_path):
            os.remove(save_path)
        save_file = h5py.File(save_path, 'w')
        sweep_keys = []
    for sweep_number in sweep_list:
        sweep_data = cell_data_set.get_sweep(sweep_number)
        allen_sweep_data = allen_data_set.get_sweep(sweep_number)
        stimulus = sweep_data['stimulus']*10**9
        cell_response = sweep_data['response'][4000:]*10**3
        sampling_rate = sweep_data['sampling_rate']
        meta_data = cell_data_set.get_sweep_metadata(sweep_number)
        stim_kind = meta_data['aibs_stimulus_name']
        allen_response = allen_sweep_data['response'][4000:]*10**3
        
        try:
            sota_response = sota_data_set[str(sweep_number)+'_response'][:]
        except KeyError:
            print("skipped :", sweep_number)
            continue
            
        
        sota_dt = sota_data_set[str(sweep_number)+'_dt']
        print('sota dt :', sota_dt, 'sampling rate:', sampling_rate)
        if not type(stim_kind) == str: 
            stim_kind = stim_kind.decode('ASCII')
        print('Stim kind: '+stim_kind)
        if 'Test' in stim_kind:
            print('Test:', sweep_number)
        if 'Short Square' in stim_kind:
            stimulus = filtr(stimulus, short_sq_ind)
            cell_response = filtr(cell_response, short_sq_ind)
            allen_response = filtr(allen_response, short_sq_ind)
            sota_response = filtr(sota_response, short_sq_ind)
        if 'Long Square' in stim_kind:
            stimulus = filtr(stimulus, long_sq_ind)
            cell_response = filtr(cell_response, long_sq_ind)
            allen_response = filtr(allen_response, long_sq_ind)
            sota_response = filtr(sota_response, long_sq_ind)
        if 'Ramp' in stim_kind:
            stimulus = filtr(stimulus, ramp_ind)
            cell_response = filtr(cell_response, ramp_ind)
            allen_response = filtr(allen_response, ramp_ind)
            sota_response = filtr(sota_response, ramp_ind)
        if 'Square - 0.5ms Subthreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_0_5_ind)
            cell_response = filtr(cell_response, sq_0_5_ind)
            allen_response = filtr(allen_response, sq_0_5_ind)
            sota_response = filtr(sota_response, sq_0_5_ind)
        if 'Square - 2s Suprathreshold' in stim_kind:
            stimulus = filtr(stimulus, sq_2_ind)
            cell_response = filtr(cell_response, sq_2_ind)
            allen_response = filtr(allen_response, sq_2_ind)
            sota_response = filtr(sota_response, sq_2_ind)
            
        if not 'Noise' in stim_kind and not 'Test' in stim_kind:
            allen_response = adjust_v_init_failure(allen_response,cell_response) 
            sota_response = adjust_v_init_failure(sota_response,cell_response) 
            fig = plot_sampled(sweep_number, stimulus, cell_response, allen_response, sota_response, sota_dt)
            if save_path:
                save_file.create_dataset(str(sweep_number)+'_stimulus', data=stimulus)
                save_file.create_dataset(str(sweep_number)+'_cell_response', data=cell_response)
                save_file.create_dataset(str(sweep_number)+'_allen_model_response', data=allen_response)
                save_file.create_dataset(str(sweep_number)+'_compare_model_response', data=sota_response)
                save_file.create_dataset(str(sweep_number)+'_dt', data=np.array([1/sampling_rate*1000]))
                sweep_keys.append(str(sweep_number))
            if pdf:
                pdf.savefig(fig)
                plt.close(fig)
        if 'Noise' in stim_kind:
            stimulus1 = filtr(stimulus, noise_ind_1)
            cell_response1 = filtr(cell_response, noise_ind_1)
            allen_response1 = filtr(allen_response, noise_ind_1)
            sota_response1 = filtr(sota_response, noise_ind_1)
            stimulus2 = filtr(stimulus, noise_ind_2)
            cell_response2 = filtr(cell_response, noise_ind_2)
            allen_response2 = filtr(allen_response, noise_ind_2)
            sota_response2 = filtr(sota_response, noise_ind_2)
            stimulus3 = filtr(stimulus, noise_ind_3)
            cell_response3 = filtr(cell_response, noise_ind_3)
            allen_response3 = filtr(allen_response, noise_ind_3)
            sota_response3 = filtr(sota_response, noise_ind_3)
            
            allen_response1 = adjust_v_init_failure(allen_response1,cell_response1) 
            sota_response1 = adjust_v_init_failure(sota_response1,cell_response1) 
            allen_response2 = adjust_v_init_failure(allen_response2,cell_response2) 
            sota_response2 = adjust_v_init_failure(sota_response2,cell_response2) 
            allen_response3 = adjust_v_init_failure(allen_response3,cell_response3) 
            sota_response3 = adjust_v_init_failure(sota_response3,cell_response3) 
        
            fig1 = plot_sampled(sweep_number, stimulus1, cell_response1, allen_response1, sota_response1, sota_dt)
            fig2 = plot_sampled(sweep_number, stimulus2, cell_response2, allen_response2, sota_response2, sota_dt)
            fig3 = plot_sampled(sweep_number, stimulus3, cell_response3, allen_response3, sota_response3, sota_dt)
            if save_path:
                save_file.create_dataset(str(sweep_number)+'_1_stimulus', data=stimulus1)
                save_file.create_dataset(str(sweep_number)+'_1_cell_response', data=cell_response1)
                save_file.create_dataset(str(sweep_number)+'_1_allen_model_response', data=allen_response1)
                save_file.create_dataset(str(sweep_number)+'_1_compare_model_response', data=sota_response1)
                save_file.create_dataset(str(sweep_number)+'_1_dt', data=np.array([1/sampling_rate*1000]))
                save_file.create_dataset(str(sweep_number)+'_2_stimulus', data=stimulus2)
                save_file.create_dataset(str(sweep_number)+'_2_cell_response', data=cell_response2)
                save_file.create_dataset(str(sweep_number)+'_2_allen_model_response', data=allen_response2)
                save_file.create_dataset(str(sweep_number)+'_2_compare_model_response', data=sota_response2)
                save_file.create_dataset(str(sweep_number)+'_2_dt', data=np.array([1/sampling_rate*1000]))
                save_file.create_dataset(str(sweep_number)+'_3_stimulus', data=stimulus3)
                save_file.create_dataset(str(sweep_number)+'_3_cell_response', data=cell_response3)
                save_file.create_dataset(str(sweep_number)+'_3_allen_model_response', data=allen_response3)
                save_file.create_dataset(str(sweep_number)+'_3_compare_model_response', data=sota_response3)
                save_file.create_dataset(str(sweep_number)+'_3_dt', data=np.array([1/sampling_rate*1000]))
                sweep_keys.append(str(sweep_number)+'_1')
                sweep_keys.append(str(sweep_number)+'_2')
                sweep_keys.append(str(sweep_number)+'_3')
            
            if False:
                pdf.savefig(fig1)
                plt.close(fig1)
                pdf.savefig(fig2)
                plt.close(fig2)
                pdf.savefig(fig2)
                plt.close(fig2)
    if save_path:
        save_file.create_dataset('sweep_numbers', data=np.array(sweep_list))
        save_file.create_dataset('sweep_keys', data=np.array([np.string_(e) for e in sweep_keys]))
        save_file.close()
        
        
# # Save parsed stimulus and responses
pdf = PdfPages('comparison.pdf')
parsed_stim_response_path = f'./allen_model_sota_model_parsed_cell_{cfg.model_num}.hdf5'
read_and_plot(sweep_numbers, parsed_stim_response_path, pdf)
pdf.close()