import numpy as np
import pandas as pd
import math
import h5py

def vector_log(param_list, base):
    return np.array([math.log(x, base) for x in param_list])

def apply_log_transform(params,bases):
    res = np.empty(params.shape)
    for param_ind,base in enumerate(bases):
        res[:,param_ind] = vector_log(params[:,param_ind],base)
    return res

param_df = pd.read_csv('../params/params_allen_full.csv')
base_params = param_df['Base value'].values
log_bases =  (param_df['Upper bound'] - param_df['Lower bound']).values
population = np.genfromtxt('population.csv').reshape(-1,16)
ea_ranking = np.genfromtxt('fitness.csv')

import pdb; pdb.set_trace()
# take log(x/t) where t is base value and x is sampled param
log_diff = np.abs(apply_log_transform(population/base_params,log_bases))
log_diff_rankings = np.argsort(np.sum(log_diff,axis=1))
ea_ranking =  np.argsort(ea_ranking)

f_ea = h5py.File('params_allen_full_ea_2.hdf5','w')
f_ranked = h5py.File('params_allen_full_ranked_2.hdf5','w')
pin_sample_ind = np.arange(len(log_diff_rankings))

f_ea.create_dataset('orig_full', data=np.array(base_params).reshape(1,-1))
f_ea.create_dataset('pin_'+str(len(log_diff_rankings))+'_full', data=population[ea_ranking])
f_ea.create_dataset('sample_ind', data=np.array(pin_sample_ind))
# doesn't matter ... can I remove this without breaking other code?
f_ea.create_dataset('param_num', data=np.array([1000]))
f_ea.create_dataset('dx', data=np.array([.1]))


f_ranked.create_dataset('orig_full', data=np.array(base_params).reshape(1,-1))
f_ranked.create_dataset('pin_'+str(len(log_diff_rankings))+'_full', data=population[log_diff_rankings])
f_ranked.create_dataset('sample_ind', data=np.array(pin_sample_ind))
# doesn't matter ... can I remove this without breaking other code?
f_ranked.create_dataset('param_num', data=np.array([1000]))
f_ranked.create_dataset('dx', data=np.array([.1]))