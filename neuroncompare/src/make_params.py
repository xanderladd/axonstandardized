import csv
import numpy as np
import h5py
import os
import sys

# Helper functions (originally from makeParamSetHelpers.py)
# Patch for neuroncompare/src/make_params.py
# Add at the beginning of the script, before the main code

import argparse

# Add argument parsing
parser = argparse.ArgumentParser(description='Generate parameters for neuroncompare')
parser.add_argument('--auto-confirm', action='store_true', 
                   help='Automatically confirm parameter generation without prompting')
args = parser.parse_args()

# Uniform function for sampling
def uniform(normDiff, currbase, lower_bound, upper_bound, numRows):
    values = np.zeros((numRows, ))
    sign = np.sign(normDiff)
    i = 0
    for x in np.nditer(normDiff):
        val = np.interp(x, [-4, 4], [lower_bound, upper_bound])
        values[i] = val
        i += 1
    return values

# The follow function helps parse csv files and generates a 2d matrix for running calculations on.
# Returns complete matrix of parsed csv and also a row vector (1 x 12) of base values for each param.
def parse_csv(file_name):
    func_names = ['Uniform'] # We are now using Uniform only
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        numCols = len(next(reader)) # changed this line from reader.next() for python3
        numRows = sum(1 for row in reader) + 1
        csvfile.seek(0)
        next(reader)
        m = np.zeros((numRows - 1, 7)) # numRows - 1 because ignoring header
        i = 0
        ch_names = []

        for row in reader:
            ch_names.append(row[0])
            for k in range(3):
                m[i, k] = float(row[k+1])
            m[i, 3] = float(row[7])

            if row[5] in func_names:
                m[i, 4] = func_names.index(row[5]) + 1
            else:
                raise Exception('Bad function name')

            m[i, 5] = row[6] == 'Open'

            if row[10] == "":
                m[i, 6] = 0.1
            else:
                m[i, 6] = float(row[10])
            i += 1
    return m, m[:,0].reshape((1, m.shape[0])), ch_names

def calculate_pmatx(data, nSubZones, nPerSubZone, sample_params, norm, seed):
    ''' 
    This function takes in __data__, which is the parsed CSV matrix after calling the above function *parse_csv*, and returns three things:
    - pMatx, which is the final pin data after sampling in a range of [-4, 4] and applying the uniform function. Each param not in sample_params will be kept at 0 and not sampled.
    - pSortedMatx, which is calculated by sorting by the summed squared differences from the original param value after sampling in the range [-4, 4] and then applying the uniform function. This matrix was mainly used for plotting the sorted values vs unsorted values to look at the general relationship between the two.
    - pSetsN is the samples from [-4, 4] for each param.
    - pSortedSetsN, which is just the sorted squared differences from the original param value after sampling in the range [-4, 4]. This does not have the uniform function applied, because we use this matrix for the OAT analysis. 
    '''
    np.random.seed(seed) # seed for consistent sampling
    outputRows = nSubZones * nPerSubZone
    nParams = (data[:, 5] == 1).sum() # 12
    freeParams = len(data) - nParams 
    pSetsN = np.zeros((outputRows, nParams))
    counter = 0

    # map the base values to a value in the range [-4, 4]
    mapped_base_values = np.zeros(nParams)
    lower_bounds = np.zeros(nParams)
    upper_bounds = np.zeros(nParams)
    base_values = np.zeros(nParams)
    for i in range(nParams):
        base_value = data[freeParams + i, 0]
        lower_bound = data[freeParams + i, 1]
        upper_bound = data[freeParams + i, 2]
        base_values[i] = base_value
        lower_bounds[i] = lower_bound
        upper_bounds[i] = upper_bound
        mapped_base_values[i] = np.interp(base_value, [lower_bound, upper_bound], [-4, 4])
    for i in range(nSubZones):
        # sample_params are the params to be sampled. Other params are set to the mapped base value to [-4, 4].
        for p in range(1, nParams+1):
            if p in sample_params:
                #pSetsN[counter:counter + nPerSubZone, p - 1] = (np.random.rand(nPerSubZone) * 2.0 - 1.0) * (4.0 * (i + 1.0) / float(nSubZones))
                raw_sample = np.random.rand(nPerSubZone) * [1 if np.random.rand() >= 0.5 else -1 for i in range(nPerSubZone)]
                for j in range(len(raw_sample)):
                    sample = raw_sample[j]
                    if sample < 0:
                        raw_sample[j] = np.interp(sample, [-1, 0], [(-4-mapped_base_values[p-1])/nSubZones*(i+1), 0]) + mapped_base_values[p-1]
                    else:
                        raw_sample[j] = np.interp(sample, [0, 1], [0, (4-mapped_base_values[p-1])/nSubZones*(i+1)]) + mapped_base_values[p-1]
                pSetsN[counter:counter + nPerSubZone, p - 1] = raw_sample
            else:
                pSetsN[counter:counter + nPerSubZone, p - 1] = mapped_base_values[p-1]
        counter += nPerSubZone
        
    # 100 norm of difference between param value and the mapped base value to [-4, 4]
    adjustedPSetsN = np.zeros((outputRows, nParams))
    
    for j in range(nParams):
        for i in range(pSetsN.shape[0]):
            adjustedPSetsN[i, j] = (pSetsN[i, j] - mapped_base_values[j]) ** norm

    # Sorting
    pSum = np.sum(adjustedPSetsN, axis=1)
    sortedList = [j[0] for j in sorted(enumerate(pSum), key=lambda i: i[1])] # sort in ascending order
    pSortedSetsN = np.zeros((outputRows, nParams))
    for j in range(nParams):
        for i in range(len(sortedList)):
            pSortedSetsN[i, j] = pSetsN[sortedList[i], j]

    # Calculate pMatx and pSortedMatx
    pMatx = np.zeros((outputRows, nParams))
    pSortedMatx = np.zeros((outputRows, nParams))
    pCurrND = np.zeros((outputRows, 1))
    pSortedCurrND = np.zeros((outputRows, 1))

    for i in range(nParams):
        pCurrND = pSetsN[:, i]               # unsorted params
        pSortedCurrND = pSortedSetsN[:, i]   # sorted params
        currBase = data[freeParams + i, 0]
        currBaseVar = data[freeParams + i, 3] 
        func = data[freeParams + i, 4]
        lastParam = data[freeParams + i, 6]
        lb = data[freeParams + i, 1]
        ub = data[freeParams + i, 2]

        adjustedND = pCurrND
        adjustedSortedND = pSortedCurrND
        
        if func == 1:
            pMatx[:, i] = uniform(adjustedND, currBase, lb, ub, outputRows)
            pSortedMatx[:, i] = uniform(adjustedSortedND, currBase, lb, ub, outputRows)
        else:
            raise Exception('function name error') 
    return pMatx, pSortedMatx, pSetsN, pSortedSetsN

# make 120 12x12 matrices and concatenate row-wise, creating 1440 x 12 matrix (is able to generalize for different dimensions)
# permute each row of original matrix one at a time. The diagonal will be original values shifted by dx
# For OAT analysis (one at a time)
def shift_by_dx(data, dx, sample_params):
    ''' 
    This function shifts each param one at a time by a given dx value.
    The function takes in two parameters, __data__ and __dx__. __data__ should be the __pSortedSetsN__ matrix returned from the above function, and __dx__ should be the appropriate dx to shift the params. 
    Returns __augmented__, which will be a 1440 * 12 matrix if the original input has 120 rows. 
    '''
    dataRows = data.shape[0]
    nParams = data.shape[1]
    num_sample_params = len(sample_params)                            # number of params to sample
    augmented = np.zeros((num_sample_params * dataRows, nParams))   # 1440 x 12 matrix
    curr_data_row = 0
    for row in range(0, augmented.shape[0], num_sample_params):     # iterate through rows, step size of sample_params at a time
        for ind in range(num_sample_params):                           # iterate through columns, where total columns is 12
            curr_col = sample_params[ind] - 1
            augmented[row + ind, :] = data[curr_data_row, :]
            augmented[row + ind, curr_col] += dx                    # increment one param at a time per row
        curr_data_row += 1
    return augmented

def calculate_pmatx_dx(data, augmented):
    ''' 
    This function takes in the shifted dx matrix __augmented__ and calculates the final pMatx after applying the multiplicative or exponent function. The first argument __data__ should be the parsed csv from before. 
    Returns the new pMatx after applying the appropriate function. 
    '''
    nParams = (data[:, 5] == 1).sum() # 12
    freeParams = len(data) - nParams
    pMatx = np.zeros((augmented.shape[0], augmented.shape[1]))
    pCurrND = np.zeros((augmented.shape[0], 1))
    for i in range(nParams):
        pCurrND = augmented[:, i]
        currBase = data[freeParams + i, 0]
        currBaseVar = data[freeParams + i, 3] 
        func = data[freeParams + i, 4]
        lastParam = data[freeParams + i, 6]
        lb = data[freeParams + i, 1]
        ub = data[freeParams + i, 2]
        adjustedND = pCurrND 
        if func == 1:
            pMatx[:, i] = uniform(adjustedND, currBase, lb, ub, pMatx.shape[0])
    return pMatx



# Then modify the user confirmation check (find the section that looks like this)
if __name__ == "__main__":
    if not args.auto_confirm:
        test_text = input("Are you SURE you want to create a new param set \
        (only do this if you are at the start of a peeling step) (y/n) :  ")
        while test_text != "y" and test_text != "n":
            test_text = input("please type 'y' or 'n' :  ")
            
        if test_text == "n":
            sys.exit(0)
    else:
        print("Auto-confirming parameter generation")
 
    FILEPATH = 'input.txt'
    
    input_file = open(FILEPATH, "r")
    inputs = {}
    input_lines = input_file.readlines()
    for line in input_lines:
        vals = line.split("=")
        if len(vals) != 2 and "\n" not in vals:
            raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
        if "\n" not in vals:
            inputs[vals[0]] = vals[1][:len(vals[1])-1]
    
    assert 'params' in inputs, "No params specified"
    assert 'data_dir' in inputs, "No data directory specified"
    assert 'user' in inputs, "No user specified"
    assert 'model' in inputs, "No model specified"
    assert 'peeling' in inputs, "No peeling specified"
    assert 'seed' in inputs, "No seed specified"
    
    assert inputs['model'] in ['mainen', 'bbp', 'allen', 'compare_bbp', 'M1_TTPC_NA_HH'], "Model must be from: \'allen\' \'mainen\', \'bbp\' or \'compare_allen\'. Do not include quotes."
    assert inputs['peeling'] in ['passive', 'potassium', 'sodium', 'calcium', 'full'], "Model must be from: \'passive\', \'potassium\', \'sodium\', \'calcium\', \'full\'. Do not include quotes."
    assert "stim_file" in inputs, "provide stims file to use, neg_stims or stims_full?"
    
    inputs['params'] = inputs['params'].split(',')
    
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
    data_dir = inputs['data_dir']
    params = [int(p) for p in inputs['params']]
    opt_ind = np.array(params) - 1
    print(params)
    
    # Sample pdx to keep the size not too large. The list must be indicies of pin.
    pin_sample_ind = list(range(1000))
    
    # Set the file path for the params csv here.
    if inputs['usePrevParams'] == "True":
        file_path = f'{data_dir}/params/params_' + model + '_' + peeling + '_prev.csv'
    else:
        file_path = f'{data_dir}/params/params_' + model + '_' + peeling + '.csv'
    
    # data is the parsed csv, orig is a row vector of base values for each param (1 x 12)
    data, orig, ch_names = parse_csv(file_path)
    
    #xander 6/26
    count = 0
    for best, lb, ub, name in zip(orig.reshape(-1,1), data[:,1], data[:,2], ch_names):
        best = best[0]
        if count in opt_ind:
            print(count, "(optimized)", "name :", name, " | best: ", np.round(best,9) , " | lb: ", np.round(lb,8), " | ub: ", round(ub,8))
        else:
            print(count, "name :", name, " | best: ", np.round(best,9) , " | lb: ", np.round(lb,8), " | ub: ", np.round(ub,8))
        print("------------------------------------------------------------")
        if 'e_pas' in name:
            data[count,:3] = - np.abs(data[count,:3])
            print("turned {} negative".format(name))
            orig[:,count] = - np.abs(orig[:,count])
            print(data[count], orig[:,count])
        count += 1
    
    pMatx, pSortedMatx, pSetsN, pSortedSetsN = calculate_pmatx(data, nSubZones, nPerSubZone, params, norm, seed)
    '''
    pMatx is the final pin data after sampling in a range of [-4, 4] and applying the uniform function. Each param not in the params list will be kept at 0 and not sampled.
    pSortedMatx is calculated by sorting by the summed squared differences from the original param value after sampling in the range [-4, 4] and then applying the uniform function. This matrix was mainly used for plotting the sorted values vs unsorted values to look at the general relationship between the two.
    pSetsN is the samples from [-4, 4] for each param.
    pSortedSetsN is the sorted squared differences from the original param value after sampling in the range [-4, 4]. This does not have the uniform function applied, because we use this matrix for the OAT analysis.
    '''
    
    # Save matrices as hdf5 files.
    #LOCAL config
    os.makedirs(f'{data_dir}/params', exist_ok=True)
    params_nwb = h5py.File(f'{data_dir}/params/params_' + model + '_' + peeling + '.hdf5', 'w') # check correctness
    params_nwb.create_dataset('orig_' + peeling, data=orig)
    
    params_nwb.create_dataset('pin_'+str(len(pSortedMatx))+'_'+peeling, data=pSortedMatx)
    # np.savetxt('./test_pin.csv', pSortedMatx)
    
    dx_matrix = shift_by_dx(pSortedSetsN, dx, params)
    final_p = calculate_pmatx_dx(data, dx_matrix)
    param_num = len(params)
    sampled_dx_matrix = []
    #for i in pin_sample_ind:
    #    for j in range(param_num):
    #        sampled_dx_matrix.append(final_p[i*param_num+j])
    
    #params_nwb.create_dataset('pdx_'+str(len(sampled_dx_matrix))+'_'+str(dx)+'_'+peeling, data=np.array(sampled_dx_matrix))
    params_nwb.create_dataset('sample_ind', data=np.array(pin_sample_ind))
    params_nwb.create_dataset('param_num', data=np.array([param_num]))
    params_nwb.create_dataset('dx', data=np.array([dx]))
    # np.savetxt('./test_pdx.csv', sampled_dx_matrix)
    
    params_nwb.close()
    # Plots final pMatx against pSortedMatx

'''
def plot_sorted(pMatx, pSortedMatx):
    for i in range(pMatx.shape[1]):
        plt.figure(figsize=(15,8))
        plt.xlabel('Distance')
        plt.ylabel('Param value [log10]')
        plt.title('Param ' + str(i + 1))
        plt.scatter([j for j in range(pMatx.shape[0])], pMatx[:,i], c = 'red', label = 'Unsorted') 
        plt.scatter([j for j in range(pSortedMatx.shape[0])], pSortedMatx[:,i], c = 'blue', label = 'Sorted') 
        plt.legend()
        plt.show()

def plot_pSetsN(pSetsN):
    for i in range(pSetsN.shape[1]):
        plt.xlabel('Sampled value before uniform')
        plt.title('Param ' + str(i + 1) + ' Distribution Before Applying Uniform')
        sns.distplot(pSetsN[:,i])
        plt.show()

def plot_pMatx(pMatx):
    for i in range(pMatx.shape[1]):
        plt.xlabel('Sampled value after uniform')
        plt.title('Param ' + str(i + 1) + ' Distribution After Applying Uniform')
        sns.distplot(pMatx[:,i])
        plt.show()
'''