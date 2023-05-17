from mpi4py import MPI
import numpy as np
import h5py
from neuron import h
import os, sys
import math
from extractModel_mappings import   allparams_from_mapping
#from bbpConfig import *
import subprocess
import time
import csv
import struct
 

##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]
nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
params_file_path = '../../../params/params_bbp_' + peeling+ '.hdf5'
stims_file_path = '../../../stims/stims_full.hdf5'


# Number of timesteps for the output volt.
ntimestep = 10000
vs_fn = '/tmp/Data/VHotP'

# Value of dt in miliseconds
dt = 0.02
if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")

# Output destination.
volts_path = '../volts/'

# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
params_hdf5 = h5py.File(params_file_path, 'r')
params_name_list = list(params_hdf5.keys())
stims_name_list = sorted(list(stims_hdf5.keys()))

curr_stim_name_list = stims_name_list
print("params names list",params_name_list)
print("stim name list", len(curr_stim_name_list))


pin_set_size = None
pdx_set_size = None

##########################
# Utility Functions      #
##########################
def split(container, count):
    """
    Simple function splitting a container into equal length chunks.

    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)


def run_model_gpu(stim_ind):
    """
    Parameters
    -------------------------------------------------------
    stim_ind: index to send as arg to neuroGPU 
    params: DEPRECATED remove

    Returns
    ---------------------------------------------------------
    p_object: process object that stops when neuroGPU done
    """
    volts_fn = vs_fn + str(stim_ind) + '.dat'
    if os.path.exists(volts_fn):
        os.remove(volts_fn)
    p_object = subprocess.Popen(['./bin/neuroGPU',str(stim_ind)])
    return p_object
    
def convert_allen_data():
    """
    Function that sets up our new allen data every run. It reads and writes every stimi
    and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
    """
    for i in range(len(curr_stim_name_list)):
        old_stim = "../Data/Stim_raw{}.csv".format(i)
        old_time = "../Data/times{}.csv".format(i)
        if os.path.exists(old_stim) :
            os.remove(old_stim)
            os.remove(old_time)
    for i in range(len(curr_stim_name_list)):
        stim = curr_stim_name_list[i]
        dt = .02 # refactor this later to be read or set to .02 if not configured
        #self.dts.append(dt)
        f = open ("../Data/times{}.csv".format(i), 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        current_times = [dt for i in range(ntimestep)]
        wtr.writerow(current_times)
        writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
        writer.writerow(stims_hdf5[stim][:])
        
def stim_swap(idx, i):
    """
    Stim swap takes 'idx' which is the stim index % 8 and 'i' which is the actual stim idx
    and then deletes the one at 'idx' and replaces it with the stim at i so that 
    neuroGPU reads stims like 13 as stim_raw5 (13 % 8)
    """
    old_stim = '../Data/Stim_raw' + str(idx) + '.csv'
    old_time = '../Data/times' + str(idx) + '.csv'
    if os.path.exists(old_stim):
        os.remove(old_stim)
        os.remove(old_time)
    os.rename(r'../Data/Stim_raw' + str(i) + '.csv', r'../Data/Stim_raw' + str(idx) + '.csv')
    os.rename(r'../Data/times' + str(i) + '.csv', r'../Data/times' + str(idx) + '.csv')

def getVolts(idx):
    '''Helper function that gets volts from data and shapes them for a given stim index'''
    fn = vs_fn + str(idx) +  '.dat'    #'.h5' 
    #curr_volts =  nrnMreadH5(fn)
    #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts
    
# # Use default communicator. No need to complicate things.
# COMM = MPI.COMM_WORLD
convert_allen_data()

for first_stim_ind in range(0,len(curr_stim_name_list), nGpus):
    
    curr_stim_names = [curr_stim_name_list[stim_ind] \
                       for stim_ind in range(first_stim_ind, first_stim_ind + nGpus) if stim_ind < len(curr_stim_name_list)]
    volts_hdf5s = []
    for stim_name in curr_stim_names: 
        if not os.path.exists(volts_path+stim_name+'_volts.hdf5'):
            volts_hdf5s.append(h5py.File(volts_path+stim_name+'_volts.hdf5', 'w'))
        else:
            vFile = h5py.File(volts_path+stim_name+'_volts.hdf5', 'r')
            if len(vFile.keys()) < 2:
                print("removing EMPTY VOLTS")
                os.remove(volts_path+stim_name+'_volts.hdf5')
                volts_hdf5s.append(h5py.File(volts_path+stim_name+'_volts.hdf5', 'w'))
            else:
                print("not empy volts")
               
    start_time_sim = time.time()

    if first_stim_ind > 0:
        for i in range(len(volts_hdf5s)):
            stim_swap(i % nGpus, first_stim_ind + i)
    if len(volts_hdf5s) > 0:
        for params_name in params_name_list:
            if 'orig' in params_name:
                params_data = params_hdf5[params_name]
                print("updating all params...")
                allparams = allparams_from_mapping(list(params_data)) 
                print("updated")
                processObjs = []
                for i in range(len(volts_hdf5s)):
                    processObjs.append(run_model_gpu(i))
                for i in range(len(volts_hdf5s)):
                    processObjs[i].wait()
                    currOrigVolts = getVolts(i)[0]
                    name_to_write = 'orig' + '_' + curr_stim_names[i]
                    print("writing :" , name_to_write, "VOLTAGE SHAPE : ", currOrigVolts.shape)
                    volts_hdf5s[i].create_dataset(name_to_write, data=currOrigVolts)

            elif 'pin' in params_name:
                params_data = params_hdf5[params_name]
                print("updating all params...")
                allparams = allparams_from_mapping(list(params_data)) 
                print("updated")
                processObjs = []
                for i in range(len(volts_hdf5s)):
                    processObjs.append(run_model_gpu(i))
                for i in range(len(volts_hdf5s)):
                    processObjs[i].wait()
                    currPinVolts = getVolts(i)
                    name_to_write = 'pin' + '_' + curr_stim_names[i]
                    print("writing :" , name_to_write, "VOLTAGE SHAPE : ", currPinVolts.shape)
                    volts_hdf5s[i].create_dataset(name_to_write, data=currPinVolts)
            else:
                continue
            # Split into however many cores are available.

        end_time_stim = time.time()
        print("STIM TOOK :", end_time_stim - start_time_sim)
        for i in range(len(volts_hdf5s)):
            volts_hdf5s[i].close()
   