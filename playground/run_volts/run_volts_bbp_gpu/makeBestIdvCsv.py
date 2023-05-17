import numpy as np
import h5py
import bluepyopt as bpop
import nrnUtils
import efel
import pandas as pd
import os
import subprocess
import time
import shutil
import struct
import glob
import ctypes
import matplotlib.pyplot as plt
from extractModel_mappings import   allparams_from_mapping

import multiprocessing
import csv
import ap_tuner as tuner
import pickle

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MPICH_GNI_FORK_MODE"] = "FULLCOPY" # export MPICH_GNI_FORK_MODE=FULLCOPY

inputFile = open("../../../../../input.txt","r") 
for line in inputFile.readlines():
    if "bbp" in line:
        from config.bbp19_config import *
    elif "allen" in line:
        from config.allen_config import *

nGpus = len([devicenum for devicenum in os.environ['CUDA_VISIBLE_DEVICES'] if devicenum != ","])
nCpus =  multiprocessing.cpu_count()



print("USING nGPUS: ", nGpus, " and USING nCPUS: ", nCpus)

custom_score_functions = [
                    'chi_square_normal',\
                    'traj_score_1',\
                    'traj_score_2',\
                    'traj_score_3',\
                    'isi',\
                    'rev_dot_product',\
                    'KL_divergence']

# Number of timesteps for the output volt.
ntimestep = 10000

if not os.path.isdir("/tmp/Data"):
    os.mkdir("/tmp/Data")

def nrnMread(fileName):
    f = open(fileName, "rb")
    nparam = struct.unpack('i', f.read(4))[0]
    typeFlg = struct.unpack('i', f.read(4))[0]
    return np.fromfile(f,np.double)

def nrnMreadH5(fileName):
    f = h5py.File(fileName,'r')
    dat = f['Data'][:][0]
    return np.array(dat)

def run_model(stim_ind, params):
    """
    Parameters
    -------------------------------------------------------
    stim_ind: index to send as arg to neuroGPU 
    params: DEPRECATED remove

    Returns
    ---------------------------------------------------------
    p_object: process object that stops when neuroGPU done
    """
    #volts_fn = vs_fn + str(stim_ind) + '.dat'
    volts_fn = vs_fn + str(stim_ind) + '.h5'
    if os.path.exists(volts_fn):
        os.remove(volts_fn)
    p_object = subprocess.Popen(['../bin/neuroGPU',str(stim_ind)])
    return p_object
    
    

# convert the allen data and save as csv
def convert_allen_data():
    """
    Function that sets up our new allen data every run. It reads and writes every stimi
    and timesi and removes previous ones. Using csv writer to write timesi so it reads well.
    """
    for i in range(len(opt_stim_name_list)):
        old_stim = "../Data/Stim_raw{}.csv".format(i)
        old_time = "../Data/times{}.csv".format(i)
        if os.path.exists(old_stim) :
            os.remove(old_stim)
            os.remove(old_time)
    for i in range(len(opt_stim_name_list)):
        stim = opt_stim_name_list[i].decode("utf-8")
        dt = .02 # refactor this later to be read or set to .02 if not configured
        f = open ("../Data/times{}.csv".format(i), 'w')
        wtr = csv.writer(f, delimiter=',', lineterminator='\n')
        current_times = [dt for i in range(ntimestep)]
        wtr.writerow(current_times)
        writer = csv.writer(open("../Data/Stim_raw{}.csv".format(i), 'w'))
        writer.writerow(stim_file[stim][:])
            

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
    """
    Helper function that gets volts from data and shapes them for a given stim index
    """
    fn = vs_fn + str(idx) +  '.h5'    #'.h5' 
    curr_volts =  nrnMreadH5(fn)
    #fn = vs_fn + str(idx) +  '.dat'    #'.h5'
    #curr_volts =  nrnMread(fn)
    Nt = int(len(curr_volts)/ntimestep)
    shaped_volts = np.reshape(curr_volts, [Nt,ntimestep])
    return shaped_volts


def make_best_volts(best_params, opt_stim_list):
    convert_allen_data()
    print(best_params)
    params = np.array(best_params).reshape(-1,1).T
    #params = np.repeat(params, 5 ,axis=0)
    data_volts_list = np.array([])
    allparams = allparams_from_mapping(list(params)) 
    for stimset in range(0,len(opt_stim_list), nGpus):
        p_objects = []
        for gpuId in range(nGpus): 
            if  (gpuId + stimset) >= len(opt_stim_list):
                break
            if stimset != 0:
                print("Swapping ", gpuId, gpuId + stimset)
                stim_swap(gpuId, gpuId + stimset)
            p_objects.append(run_model(gpuId, []))
        for gpuId in range(nGpus):
            if  (gpuId + stimset) >= len(opt_stim_list):
                break 
            p_objects[gpuId].wait()
            if len(data_volts_list) < 1:
                data_volts_list  = getVolts(gpuId)
            else:
                data_volts_list = np.append(data_volts_list, getVolts(gpuId),axis=0)
            print(data_volts_list.shape)
    np.savetxt("resultVolts.csv", data_volts_list, delimiter=",")
    return data_volts_list


if __name__ == "__main__":
    data = nrnUtils.readParamsCSV(paramsCSV)
    opt_ind = np.array(params_opt_ind)
    data = np.array([data[i] for i in opt_ind])
    GA_result_path = './best_indv_logs/best_indvs_gen_9.pkl'
    #GA_result_path = './best_indv_logs/best_indv_bbp_potassium.pkl'
    
    with open(GA_result_path, 'rb') as f:
        best_indv = pickle.load(f)
        #best_indv[-1][-2:] = [0.00008, 3.00E-05]
        optParams = best_indv[-1]
        
    best_indv = list(orig_params)
    for i in range(len(opt_ind)):
        best_indv[opt_ind[i]] = optParams[i]
    print(best_indv)
    print(len(best_indv), "LEN BEST INDV")
    opt_stim_list = [e.decode('ascii') for e in opt_stim_name_list]
    make_best_volts(best_indv, opt_stim_list)
    
  
    


