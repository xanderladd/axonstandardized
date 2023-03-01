# Import statements
import makeParamSetHelpers as helper
import numpy as np
import h5py
import os
import sys

test_text = input("Are you SURE you want to create a new param set \
(only do this if you are at the start of a peeling step) (y/n) :  ")
while test_text != "y" and test_text != "n":
    test_text = input("please type 'y' or 'n' :  ")
    
if test_text == "n":
    sys.exit(0)

FILEPATH =  'input.txt'

input_file = open(FILEPATH, "r")
inputs = {}
input_lines = input_file.readlines()
for line in input_lines:
    vals = line.split("=")
    if len(vals) != 2 and "\n" not in vals:
        raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
    if "\n" not in vals:
        inputs[vals[0]] = vals[1][:len(vals[1])-1]

assert 'params' in inputs, "No params specificed"
assert 'user' in inputs, "No user specified"
assert 'model' in inputs, "No model specificed"
assert 'peeling' in inputs, "No peeling specificed"
assert 'seed' in inputs, "No seed specificed"

assert inputs['model'] in ['mainen', 'bbp', 'allen', 'compare_allen'], "Model must be from: \'allen\' \'mainen\', \'bbp\' or \'compare_allen\'. Do not include quotes."


assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"

inputs['params'] = inputs['params'].split(',')

# CHANGED
#inputs['params'] = [int(p) for p in inputs['params']]

# Set these values accordingly to get the desired number of output rows. outputRows = nSubZones * nPerSubZone
nSubZones = int(inputs['nSubZones']) if 'nSubZones' in inputs else 100
nPerSubZone = int(inputs['nPerSubZone']) if 'nPerSubZone' in inputs else 10
# Set the norm value here.
norm = float(inputs['norm']) if 'norm' in inputs else 100
# Set a seed value here for generating the samples between -4 and 4.
seed = int(inputs['seed'])
# Set the value of dx here.
dx = float(inputs['dx']) if 'dx' in inputs else 0.001

# Set params to sample here. Other params will be kept at base value. The default below is to sample all params.

model = inputs['model']
peeling = inputs['peeling']
user = inputs['user']
params = [int(p) for p in inputs['params']]
opt_ind = np.array(params) - 1
print(params)

# Sample pdx to keep the size not too large. The list must be indicies of pin.
pin_sample_ind = list(range(1000))

# Set the file path for the params csv here.
if inputs['usePrevParams'] == "True":
    file_path = 'param_stim_generator/params_reference/params_' + model + '_' + peeling + '_prev.csv'
else:
    file_path = 'param_stim_generator/params_reference/params_' + model + '_' + peeling + '.csv'


# data is the parsed csv, orig is a row vector of base values for each param (1 x 12)
data, orig, ch_names = helper.parse_csv(file_path)




#xander 6/26
count = 0
for best, lb, ub, name in zip(orig.reshape(-1,1), data[:,1], data[:,2], ch_names):
    best = best[0]
    if count in opt_ind:
        print(count, "(optimized)", "name :", name, " | best: ", np.round(best,9) , " | lb: ", np.round(lb,8), " | ub: ", round(ub,8))
        

    else:
        print(count, "name :", name, " | best: ", np.round(best,9) , " | lb: ", np.round(lb,8), " | ub: ", np.round(ub,8))
    print("------------------------------------------------------------")
    if name == 'e_pas_all':
        data[count,:3] = - np.abs(data[count,:3])
        print("turned {} negative".format(name) )
        orig[:,count] =  - np.abs(orig[:,count])
        print(data[count], orig[:,count])
    count += 1

   
    
    
pMatx, pSortedMatx, pSetsN, pSortedSetsN = helper.calculate_pmatx(data, nSubZones, nPerSubZone, params, norm, seed)
'''
pMatx is the final pin data after sampling in a range of [-4, 4] and applying the uniform function. Each param not in the params list will be kept at 0 and not sampled.
pSortedMatx is calculated by sorting by the summed squared differences from the original param value after sampling in the range [-4, 4] and then applying the uniform function. This matrix was mainly used for plotting the sorted values vs unsorted values to look at the general relationship between the two.
pSetsN is the samples from [-4, 4] for each param.
pSortedSetsN is the sorted squared differences from the original param value after sampling in the range [-4, 4]. This does not have the uniform function applied, because we use this matrix for the OAT analysis.

'''
# Save matrices as hdf5 files.
#LOCAL config
params_nwb = h5py.File('params/params_' +model + '_' + peeling +'.hdf5', 'w') # check correctness
params_nwb.create_dataset('orig_' + peeling, data=orig)


params_nwb.create_dataset('pin_'+str(len(pSortedMatx))+'_'+peeling, data=pSortedMatx)
# np.savetxt('./test_pin.csv', pSortedMatx)



dx_matrix = helper.shift_by_dx(pSortedSetsN, dx, params)
final_p = helper.calculate_pmatx_dx(data, dx_matrix)
param_num = len(params)
sampled_dx_matrix = []
#for i in pin_sample_ind:
    #for j in range(param_num):
        #sampled_dx_matrix.append(final_p[i*param_num+j])

#params_nwb.create_dataset('pdx_'+str(len(sampled_dx_matrix))+'_'+str(dx)+'_'+peeling, data=np.array(sampled_dx_matrix))
params_nwb.create_dataset('sample_ind', data=np.array(pin_sample_ind))
params_nwb.create_dataset('param_num', data=np.array([param_num]))
params_nwb.create_dataset('dx', data=np.array([dx]))
# np.savetxt('./test_pdx.csv', sampled_dx_matrix)

params_nwb.close()
# Plots final pMatx against pSortedMatx
