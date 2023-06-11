import pandas as pd
import math
import numpy as np
import h5py
import score_functions as sf
import config

def get_param_bounds(params_csv, params_opt_ind):
    param_df = pd.read_csv(params_csv)
    orig_params = param_df['Base value'].values
    names = param_df['Param name'].values

    param_df = param_df.loc[params_opt_ind]
    mins, maxs = param_df['Lower bound'].values, param_df['Upper bound'].values
    return names, orig_params, mins, maxs 
    
def log_params(maxs, mins, orig_params):
    bases = maxs / mins
    # log bound orig params
    orig_params = np.array([math.log(orig_params[i],bases[i]) for i in range(len(mins))])
    mins = np.array([math.log(mins[i],bases[i]) for i in range(len(mins))])
    maxs = np.array([math.log(maxs[i],bases[i]) for i in range(len(maxs))])
    return bases, orig_params, mins, maxs

    
def passive_chisq(target, data):
    return np.linalg.norm(target-data)**2   / np.linalg.norm(target)

def decode_list(stim_name_list):
    res = []
    for stim_name in stim_name_list:
        if type(stim_name) != str:
            stim_name = stim_name.decode('ASCII')
        res.append(stim_name)
    return res
    
def score_passive(target_volt, data_volt):
    psv_score = config.PASSIVE_SCALAR * \
                len(config.score_function_ordered_list) * \
                passive_chisq(curr_target_volt, curr_data_volt)
    return psv_score

def eval_function(target, data, function, dt):
    if function in config.custom_score_functions:
        score = getattr(sf, function)(target, data, dt)
    else:
        score = sf.eval_efel(function, target, data, dt)
    return score

    
def normalize_score(curr_score, stim_name, curr_sf):
    if not np.isfinite(curr_score):
        curr_score = 100000
        print(1/0)
    else:
        norm_score = curr_score
        norm_score = config.normalizers[stim_name.encode("ASCII")][curr_sf].transform(np.array(curr_score).reshape(-1,1))[0]  # load and use a saved sklean normalizer from previous step
        norm_score = min(max(norm_score,-2),2) # allow a little lee-way
    return norm_score

def retrieve_dt(curr_stim_name, stims_hdf5, dt=None):
    if type(curr_stim_name) ==  bytes or type(curr_stim_name) ==  np.bytes_: 
        curr_stim_name = curr_stim_name.decode('ASCII')
    if not dt:
        dt = stims_hdf5[curr_stim_name + '_dt']
        
    assert dt, "DT not specified"
    assert dt < .1, "DT is too high"
    
    return dt

def evaluate_score_function(stim_names, target_volts, data_volts, weights, dt=None):
    stim_names = decode_list(stim_names)
    stims_hdf5 = h5py.File(config.stims_path, 'r')
    # start all scores at 0
    total_score, psv_scores, actv_scores= 0, 0, 0
    
    for stim_idx in range(len(stim_names)):
        curr_data_volt, curr_target_volt = data_volts[stim_idx], target_volts[stim_idx] 
        stim_name = stim_names[stim_idx]
        curr_stim = stims_hdf5[stim_name][:]
        dt = retrieve_dt(stim_names[stim_idx], stims_hdf5, dt=config.dt)
        
        # HANDLE PASSIVE STIM
        if np.max(curr_target_volt) < 0:
            psv_scores += score_passive(target_volt, data_volt)
            # continue since active evalaution is not needed
            continue
            
        # HANDLE ACTIVE STIM
        for sf_idx in range(len(config.score_function_ordered_list)):
            curr_sf = config.score_function_ordered_list[sf_idx].decode('ascii')
            curr_weight = weights[len(config.score_function_ordered_list)*stim_idx + sf_idx]
            # nothing to do if weight is 0
            if curr_weight == 0:
                continue
            else:
                curr_score = eval_function(curr_target_volt, curr_data_volt, curr_sf, dt)
                norm_score = normalize_score(curr_score, stim_name, curr_sf)
            
            actv_scores += norm_score * curr_weight
            
    total_score = psv_scores + actv_scores
    # print('ACTIVE :', actv_scores, "PASV:", psv_scores)
    return total_score


def un_nest_score(score):
    res = []
    
    if score == None:
        return []
    if (type(score) == list or type(score) == np.array) and not len(score):
        return []

    if type(score) == np.float64:
        return [score]

    for elem in score:
        if type(elem) == np.float64:
            res.append(elem)
        else:
            res += un_nest_score(elem)
    return res

def unest_mpi_arr(lst):
    if lst == None: return 
    

    assert type(lst[0][0]) != list
    assert type(lst[0][0]) !=  np.ndarray

    step = len(lst[0])
    size = len(lst)
    max_arr = np.sum([len(elem) for elem in lst] ) #len(lst) * step - step % len(lst[-1])
    res = np.empty(max_arr)
    
    for idx, elem in enumerate(lst):
        insert_inds = np.arange(idx, max_arr, size)
        
        try:
            res[insert_inds] = elem
        except:
            import pdb; pdb.set_trace()
        
    return res

# def ea_debug_function_ill_probably_never_use_but_is_helpful():
    # if global_rank == argmin or global_rank == 776:
    #     import matplotlib.pyplot as plt
    #     plt.plot(data_volts_list[0], color='red')
    #     plt.plot(self.target_volts_list[0], color='black')
    #     plt.savefig(f'volts_{global_rank}.png')
    #     plt.close()
    #     np.save(f'volts_{global_rank}.npy', np.array(data_volts_list[0]))
