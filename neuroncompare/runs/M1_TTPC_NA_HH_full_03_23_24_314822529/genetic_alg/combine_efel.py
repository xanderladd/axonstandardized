import pickle
import os
import cfg

prefix ='efel_data/subsets'
files = os.listdir(prefix)

res = {}

for file in files:
    with open(os.path.join(prefix, file), 'rb') as f:
        data = pickle.load(f)
    for key in data.keys():
        res[key] = data[key]

pickle.dump(res, open(f'./efel_data/efel_dataset_model_{cfg.model_num}.pkl', 'wb'))