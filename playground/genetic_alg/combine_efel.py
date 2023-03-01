import pickle
import os
os.chdir('neuron_genetic_alg')
from config import *
os.chdir('../')

prefix ='efel_data/subsets'
files = os.listdir(prefix)

res = {}

for file in files:
    with open(os.path.join(prefix, file), 'rb') as f:
        data = pickle.load(f)
    for key in data.keys():
        res[key] = data[key]

pickle.dump(res, open(f'./efel_data/efel_dataset_model_{model_num}.pkl', 'wb'))