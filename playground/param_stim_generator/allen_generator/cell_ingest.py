import allensdk.api.queries.biophysical_api as lib_biophysical_api
import allensdk.api.queries.cell_types_api as lib_cell_api
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
from allensdk.core.nwb_data_set import NwbDataSet
import matplotlib.pyplot as plt
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cell_id', type=int, required=True)
args = parser.parse_args()


# Code start
biophysical_api = lib_biophysical_api.BiophysicalApi()
cell_api = lib_cell_api.CellTypesApi()

cell_info = cell_api.list_cells_api()
cell_df = pd.DataFrame(cell_info)

# cell_id overrides this
# mouse v5, fi curve and has biophysical model
selection_criteria = (cell_df['donor__species'] == 'Mus musculus') & \
                     (cell_df['structure__name'] == '"Primary visual area, layer 5"') & \
                     (cell_df['ef__f_i_curve_slope'] > .2)  & \
                     (cell_df['m__biophys'] == 1) 

# retrieve cell ID for a random cell that follows above criteria
# np.random.choice(cell_df.loc[selection_criteria,'specimen__id'])
#cell_id = 333785935
cell_id = args.cell_id 

# NOTE: saves data to file
os.makedirs('../../../../axonstandardized_data/nwb_files', exist_ok=True)
cell_api.save_ephys_data(cell_id, f"../../../../axonstandardized_data/nwb_filesnwb_files/{cell_id}.nwb")

# read NWB file that we just downloaded
os.makedirs('../../../../axonstandardized_data/nwb_files', exist_ok=True)
cell_data = NwbDataSet(f"../../../../axonstandardized_data/nwb_files/{cell_id}.nwb")

sweeps = cell_data.get_sweep_numbers()
sweep = 58 #np.random.choice(sweeps)
voltage_trace = cell_data.get_sweep(sweep)

start, end = voltage_trace['index_range'][0], 150000 + 290800 #voltage_trace['index_range'][1]
stim = voltage_trace['stimulus'][start:end]
stim_unit = voltage_trace['stimulus_unit']
response = voltage_trace['response'][start:end] * 1000 # volts to mV
dt = 1 / voltage_trace['sampling_rate']
