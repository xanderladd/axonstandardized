import csv
import pandas as pd
import os
import numpy as np
import h5py
import pickle
import warnings
warnings.filterwarnings('ignore')


input_file = open('./input.txt', "r")
    
inputs = {}
input_lines = input_file.readlines()
for line in input_lines:
    vals = line.split("=")
    if len(vals) != 2 and "\n" not in vals:
        raise Exception("Error in line:\n" + line + "\nPlease include only one = per line.")
    if "\n" not in vals:
        inputs[vals[0]] = vals[1][:len(vals[1])-1]

if 'opt_stims' in inputs:
    opt_stims = inputs['opt_stims']
else:
    opt_stims = []

print("opt_stims:", opt_stims)