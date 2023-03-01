import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import h5py
import pickle
import os
from allensdk.core.nwb_data_set import NwbDataSet
import concurrent.futures
os.chdir("neuron_genetic_alg/neuron_files/allen")
from neuron import h
os.chdir("../../../")
print(os.getcwd())
import sys
import pandas as pd
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
import multiprocessing
from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print(rank)
# print(1/0)
# size = comm.Get_size()
rank = int(os.environ['SLURM_PROCID'])


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


def plot_stim_volts_pair(stim, volts, title_stim, title_volts, file_path_to_save=None):
    plt.figure(figsize=(cm_to_in(28), cm_to_in(8)))
    plt.subplot(1, 2, 1)
    plt.title(title_stim)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (nA)')
    plt.plot(stim, color='black', linewidth=0.7)
    plt.subplot(1, 2, 2)
    plt.title('Voltage Response '+title_volts)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (mV)')
    #plt.plot(volts[0], label='target', color='black')
    #plt.plot(volts[1], label='best individual', color='crimson')
    plt.plot(volts, color='crimson')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.tight_layout(pad=1)
    plt.show()
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")

def cm_to_in(cm):
    return cm/2.54

# Code for optimization results analysis
def read_best_params(opt_result_path, base_path, opt_ind):
    
    df = pd.read_csv(base_path, skipinitialspace=True, usecols=['Base value'])
    base_full = df.values.T[0]
    
    with open(opt_result_path, 'rb') as f:
        best_indvs = pickle.load(f, encoding = "latin1")
        
    best_full = list(base_full)
    for i in range(len(opt_ind)):
        best_full[opt_ind[i]] = best_indvs[-1][i]
        
    return best_full

# Running a single volt
def run_single_volts(param_set, stim_data, dt):

    print('running volts')
    run_file = 'neuron_genetic_alg/neuron_files/allen/run_model_cori.hoc'
    h.load_file(run_file)

    total_params_num = len(param_set)
    ntimestep = len(stim_data)
    timestamps = np.array([dt for i in range(ntimestep)])
    h.curr_stim = h.Vector().from_python(stim_data)
    h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
    h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
    h.ntimestep = ntimestep
    h.runStim()
    out = h.vecOut.to_python()
    return np.array(out)

def sample_stim(raw_stim, orig_rate, stim_kind, sample_rate=100):
    sampled_stim = []
    if stim_kind == 'Ramp':
        raw_stim = raw_stim[4000:int(len(raw_stim)/3)]
    else:
        raw_stim = raw_stim[4000:]
    orig_len = len(raw_stim)
    for i in range(0, orig_len, sample_rate):
        sampled_stim.append(raw_stim[i])
    return np.array(sampled_stim)*10**9, 1/orig_rate*sample_rate*1000

def convert_scale(stim, sample_rate):
    stim = stim[4000:]
    return np.array(stim)*10**9, 1/sample_rate*1000



GA_result_path = 'neuron_genetic_alg/best_indv_logs/best_indvs_gen_6.pkl'
# base_params_path = './params/params_bbp_full_gpu_tuned_10_based.csv'
base_params_path = './params/params_allen_full.csv'

# opt_ind=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
opt_ind = np.arange(16)
best_params = read_best_params(GA_result_path, base_params_path, opt_ind)
best_params[1] = - best_params[1]
model_num = str(488683423)
os.makedirs('./model_responses',exist_ok=True)
os.makedirs('./compare_responses',exist_ok=True)
original_file_name = f'./model_responses/{str(488683423)}.nwb'
orig_dataset = NwbDataSet(original_file_name)
sweep_numbers = sorted(orig_dataset.get_experiment_sweep_numbers())


# if os.path.isfile(output_path):
#     output_file = h5py.File(output_path, 'a')
# else:
#     os.remove(output_path)
#     output_file = h5py.File(output_path, 'w')


try:
    sweep_number = sweep_numbers[rank]
except:
    exit()
    
output_path = f'./compare_responses/{sweep_number}.csv'

sweep_data = orig_dataset.get_sweep(sweep_number)
stim_kind = orig_dataset.get_sweep_metadata(sweep_number)['aibs_stimulus_name']
stimulus = sweep_data['stimulus']
sampling_rate = sweep_data['sampling_rate']
sampled_stim, dt = convert_scale(stimulus, sampling_rate)
# sampled_stim, dt = sample_stim(stimulus, sampling_rate, stim_kind)
print(f'running {sweep_number}')
volts = run_single_volts(best_params, sampled_stim, dt)
np.savetxt(output_path, volts, delimiter=',')
# output_file.create_dataset(str(sweep_number)+'_stimulus', data=np.array(stimulus))
# output_file.create_dataset(str(sweep_number)+'_response', data=np.array(volts))
# output_file.create_dataset(str(sweep_number)+'_dt', data=np.array([dt]))
# output_file.close()
