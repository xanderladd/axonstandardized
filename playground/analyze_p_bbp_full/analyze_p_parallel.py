from mpi4py import MPI
import numpy as np
import h5py
import scipy.stats as stat
import os
import sys
import argparse
from noisyopt import minimizeCompass
import config
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Analyze P Parllel')
parser.add_argument('--model', type=str, required=True, help='specifies model for AnalyzeP')
parser.add_argument('--peeling', type=str, required=True, help='specifies peeling for AnalyzeP')
parser.add_argument('--CURRENTDATE', type=str, required=True, help='specifies date')
parser.add_argument('--custom', type=str, required=False, help='specifies custom postfix')

args = parser.parse_args()
model = args.model
peeling = args.peeling
currentdate = args.CURRENTDATE
custom = args.custom


if 'runs' in os.getcwd():
    wrkDir = os.getcwd()
elif custom is not None:
    wrkDir = 'runs/' + model + '_' + peeling + '_' + currentdate + '_' + custom
else:
    wrkDir = 'runs/' + model + '_' + peeling + '_' + currentdate
score_path =  wrkDir + '/scores/'
optimization_results_path = wrkDir + '/genetic_alg/optimization_results/'
if not os.path.isdir(optimization_results_path):
    os.makedirs(optimization_results_path, exist_ok=True)
params_file_path = 'params/params_' + model + '_' + peeling + '.hdf5'

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.

    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def construct_stim_score_function_list(score_path):
    stim_list = []
    file_list = os.listdir(score_path)
    for file_name in file_list:
        if '.hdf5' in file_name:
            curr_scores = h5py.File(score_path + file_name, 'r')
            stim_list.append(curr_scores['stim_name'][0].decode('ascii'))
            score_function_list = [e.decode('ascii') for e in curr_scores['score_function_names'][:]]
    return sorted(stim_list), sorted(score_function_list)

def construct_score(ordered_stim_list, ordered_score_function_list):
    norm_score_prefix = 'norm_pin_scores_'
    pin_score_dict = {}
    for stim in ordered_stim_list:
        score_file_name = stim + '_scores.hdf5'
        score_file = h5py.File(score_path + score_file_name, 'r')
        curr_pin_list = []
        for function_name in ordered_score_function_list:
            curr_pin_score = score_file[norm_score_prefix+function_name][:].T[0]
            pin_len = len(curr_pin_score)
            curr_pin_list.append(curr_pin_score)
        pin_score_dict[stim] = curr_pin_list
    return pin_score_dict


def construct_objective(score_mat):
    perfect_distribution = np.cumsum(np.ones(np.shape(score_mat)[1]))
    def obj(x):
    # Vector x is the weight vector where dot procuct with score vector and
    # argmin of objective will be the optimal linear combination
        x = x[np.newaxis]
        spearman = (stat.spearmanr(np.dot(x, score_mat).reshape((len(perfect_distribution))), perfect_distribution))[0]
#         print('Spearman: ', spearman/8)
#         print('Mean: ', mean)
#         print('Standard Deviation: ', 3*std)
#         print('Total: ', -spearman/8 - mean + 3*std, '\n')
        # if np.isnan(spearman):
        #     print("There is NAN")
        return -spearman
    return obj

def optimize(stim_name_list, pin_score_dict):
    score_mat = np.concatenate(tuple([pin_score_dict[n] for n in stim_name_list]), axis=0)
    
    optimizer_len = np.shape(score_mat)[0]
    bound = [[0, 100] for _ in range(optimizer_len)]
    initial_guess = np.array([np.random.random_sample()*100 for _ in range(optimizer_len)])
    obj = construct_objective(score_mat)
    print("starting min compas")
    return minimizeCompass(obj, bounds=bound, x0=initial_guess, deltainit = 1000, deltatol=0.001, paired=False)

# Use default communicator. No need to complicate things.
COMM = MPI.COMM_WORLD

if COMM.rank == 0:
    # Each job should contain params_name, a single param set index
    # and number of total params as a list: [params_name, param_ind, n]
    jobs = []
    ordered_stim_list, ordered_score_function_list = construct_stim_score_function_list(score_path)
    pin_score_dict = construct_score(ordered_stim_list, ordered_score_function_list)

    for stim_name in ordered_stim_list:
        jobs.append([stim_name])
    # Split into however many cores are available.
    jobs = split(jobs, COMM.size)
else:
    pin_score_dict = None
    ordered_score_function_list = None
    jobs = None

pin_score_dict = COMM.bcast(pin_score_dict, root=0)
ordered_score_function_list = COMM.bcast(ordered_score_function_list, root=0)
jobs = COMM.scatter(jobs, root=0)

# Now each rank just does its jobs and collects everything in a results list.
# Make sure to not use super big objects in there as they will be pickled to be
# exchanged over MPI.
results = {}
for job in jobs:
    opt_result = optimize(job, pin_score_dict)
    print('Stimulation: ', job[0])
    print('Optimization result: \n', opt_result)
    results[job[0]] = opt_result

results = MPI.COMM_WORLD.gather(results, root=0)

if COMM.rank == 0:
    optimized_result_dict = {}
    for d in results:
        k = d.keys()
        for key in k:
            optimized_result_dict[key] = d[key]

    optimized_result_sorted = sorted(list(optimized_result_dict.items()), key=lambda x : x[1]['fun'])
    opt_result_hdf5 = h5py.File(optimization_results_path+'opt_result_single_stim_' + model + \
    '_' + peeling + '_full.hdf5', 'w')
    ordered_score_function_list_as_binary = np.array([np.string_(e) for e in ordered_score_function_list])
    opt_result_hdf5.create_dataset('ordered_score_function_list', data=ordered_score_function_list_as_binary)
    stims_optimal_order = []
    for opt_result in optimized_result_sorted:
        opt_result_hdf5.create_dataset(opt_result[0], data=opt_result[1]['x'])
        stims_optimal_order.append(np.string_(opt_result[0]))
    opt_result_hdf5.create_dataset('stims_optimal_order', data=np.array(stims_optimal_order))
    opt_result_hdf5.close()
