import sys
import os
import h5py
import copy
# changed 12/18 ^^


##########################
# Script PARAMETERS      #
##########################

# Relative path common to all other paths.
peeling=sys.argv[2]

# modification to run stims 10 Nodes X 30 stims = 300stims
with open('../../../input.txt', 'r') as f:
    for line in f:
        if "=" in line:
            name, value = line.split("=")
            if name == "num_nodes":
                num_nodes = int(value)
            if name == "num_volts":
                num_volts = int(value)
            if name == "stim_file":
                stim_file = value.replace('\n','')
            if name == "model":
                model = value.replace('\n','')
            if name== "params":
                param_opt_inds = value.split(",")
            if name == "timesteps":
                ntimestep = int(value)
           
         
if model == 'compare_allen':
    print("******TURNING E PAS NEGATIVE HACK*******")
    if "2" in param_opt_inds:
        neg_index = 1
    os.chdir('neuron_files/compare_allen')	
    from neuron import h	
    os.chdir('../../')
    run_file = 'neuron_files/compare_allen/run_model_cori.hoc'
elif model == 'bbp':
    os.chdir('neuron_files/bbp')	
    from neuron import h	
    os.chdir('../../')
    run_file = 'neuron_files/bbp/run_model_cori.hoc'
elif model == 'allen':
    os.chdir('neuron_files/allen_modfiles')	
    from neuron import h	
    os.chdir('../../')
    run_file = None



params_file_path = '../../../../../params/params_' + model + '_' + peeling+ '.hdf5'
stims_file_path = '../../../../../stims/' + stim_file + '.hdf5'
# Number of timesteps for the output volt.
# ntimestep = 10000

# Output destination.
volts_path = '../../../volts/'

# Required variables. Some should be updated at rank 0
prefix_list = ['orig', 'pin', 'pdx']
stims_hdf5 = h5py.File(stims_file_path, 'r')
params_hdf5 = h5py.File(params_file_path, 'r')
params_name_list = list(params_hdf5.keys())
neg_index = None

stims_name_list = sorted(list(stims_hdf5.keys()))
stims_name_list = [elem for elem in stims_name_list if "dt" not in elem]

num_stims_to_run = 1
i=int(sys.argv[1])
if i == 0 and num_nodes == 1:
    curr_stim_name_list = stims_name_list
elif num_nodes > 1 and num_volts == 0:
    num_stims_to_run = math.ceil(len(stims_name_list) / num_nodes)
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]
    print(len(curr_stim_name_list))
else:
    curr_stim_name_list = stims_name_list[(i-1)*num_stims_to_run:(i)*num_stims_to_run]



curr_stim_name_list.reverse()
curr_stim_name_list = curr_stim_name_list[:1]
print(ntimestep)
print("params names list",params_name_list)
print("stim name list", curr_stim_name_list)

curr_stim_name_list_copy = copy.deepcopy(curr_stim_name_list)
for curr_stim_name in curr_stim_name_list_copy:
    filepath = volts_path+curr_stim_name+'_volts.hdf5'
    if os.path.isfile(filepath):
        curr_stim_name_list.remove(curr_stim_name)
        
if len(curr_stim_name_list) < 1:
    print("STIM NAME LIST is EMPTY CUS ITS COMPLETE, EXITING")
    exit()

    


pin_set_size = None
pdx_set_size = None


if model == 'compare_allen' or model == 'bbp':
    def run_model(param_set, stim_name, ntimestep):
        h.load_file(run_file)
        total_params_num = len(param_set)
        dt = stims_hdf5[stim_name+'_dt']
        stim_data = stims_hdf5[stim_name]
        curr_ntimestep = len(stim_data)
        timestamps = np.array([dt for i in range(curr_ntimestep)])
        h.curr_stim = h.Vector().from_python(stim_data)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))

        h.ntimestep = curr_ntimestep
        h.runStim()
        out = h.vecOut.to_python()
        return np.array(out)
elif model == 'allen':
    
    def run_model(param_set,  sweep, ntimestep):
        args = {'manifest_file': './manifest.json','axon_type': 'truncated'}
        description = runner.load_description(args)
        utils = runner.create_utils(description)
        h = utils.h
        # configure model
        manifest = description.manifest
        morphology_path = description.manifest.get_path('MORPHOLOGY').encode('ascii', 'ignore')
        morphology_path = morphology_path.decode("utf-8")
        utils.generate_morphology(morphology_path)
        utils.load_parameters(param_set)
        # utils.load_cell_parameters()
        responses = []
        dt = stims_hdf5[str(sweep) + "_dt"][:][0]
        stim = stims_hdf5[str(sweep) ][:]
        sweep = int(str(sweep))
        # configure stimulus and recording
        stimulus_path = description.manifest.get_path('stimulus_path')
        run_params = description.data['runs'][0]
        # utils.setup_iclamp(stimulus_path, sweep=sweep)
        # change this so they don't change our dt
        v_init = -88.4 # HARDCODED
        utils.setup_iclamp2(stimulus_path, sweep=sweep, stim=stim, dt=dt, v_init=v_init)
        vec = utils.record_values()
        tstart = time.time()

        if abs(dt*h.nstep_steprun*h.steps_per_ms - 1)  != 0:
            h.steps_per_ms = 1/(dt * h.nstep_steprun)

        h.finitialize()
        h.run()
        tstop = time.time()
        res =  utils.get_recorded_data(vec)
        final_voltage = res['v']*1000

        return final_voltage