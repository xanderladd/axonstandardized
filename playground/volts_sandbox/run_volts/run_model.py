import numpy as np
import os
from hoc_utils import decode_list, retrieve_dt
import config
from NeuronModelClass import NeuronModel

os.chdir(config.neuron_path) 
from neuron import h
os.chdir("../../")


import h5py

try:
    
    import allensdk.core.json_utilities as ju
    # from biophys_optimize.utils import Utils
    from allensdk.model.biophysical.utils import create_utils
    import allensdk.model.biophysical.runner as runner
except ImportError:
    print('could not import AllenSDK')
    
if 'bbp' in config.model:
    def run_model(param_set, stim_name_list, input_dt=None):
        h.load_file(config.run_file)
        volts_list = []
        stims = h5py.File(config.stims_path, 'r')
        for curr_stim_name in stim_name_list:
            total_params_num = len(param_set)
            curr_stim = stims[curr_stim_name][:]
            dt = retrieve_dt(curr_stim_name, stims, dt=input_dt)
            timestamps = np.array([dt for i in range(config.ntimestep)])
            h.curr_stim = h.Vector().from_python(curr_stim)
            h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
            h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
            h.ntimestep = config.ntimestep
            h.runStim()
            out = h.vecOut.to_python()        
            volts_list.append(out)
        return np.array(volts_list)
    
    # Running a single volt
    def run_model_stim(param_set, stim_data, dt, mod_dir=''):
        print('running volts')
        run_file = 'neuron_genetic_alg/neuron_files/compare_bbp/run_model_cori.hoc'
        h.load_file(run_file)

        total_params_num = len(param_set)
        ntimestep = len(stim_data)
        timestamps = np.array([dt for i in range(ntimestep)])
        h.curr_stim = h.Vector().from_python(stim_data)
        h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
        h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
        h.ntimestep = ntimestep
        h.runStim()
        out = h.vecOut.to_python()
        return np.array(out)
    
elif config.model == 'allen':
    def run_allen_model(param_set, stim_name_list):
        description = runner.load_description(args)
        utils = runner.create_utils(description)
        h = utils.h

        # configure model
        manifest = description.manifest
        morphology_path = description.manifest.get_path('MORPHOLOGY').encode('ascii', 'ignore')
        morphology_path = morphology_path.decode("utf-8")
        utils.generate_morphology(morphology_path)
        utils.load_parameters(param_set)
        responses = []
        for sweep in stim_name_list:
            dt = stim_file[str(sweep.decode('ascii')) + "_dt"][:][0]
            stim = stim_file[str(sweep.decode('ascii')) ][:]

            sweep = int(str(sweep.decode('ascii')))
            # configure stimulus and recording
            stimulus_path = description.manifest.get_path('stimulus_path')
            run_params = description.data['runs'][0]
            # change this so they don't change our dt
            v_init = target_volts_hdf5[str(sweep)][0] # - 14
            utils.setup_iclamp2(stimulus_path, sweep=sweep, stim=stim, dt=dt, v_init=v_init)
            vec = utils.record_values()
            tstart = time.time()
            # ensure they don't change dt during the sim
            if abs(dt*h.nstep_steprun*h.steps_per_ms - 1)  != 0:
                h.steps_per_ms = 1/(dt * h.nstep_steprun)

            h.finitialize()
            h.run()
            tstop = time.time()
            res =  utils.get_recorded_data(vec)
            # rescale recorded data to mV
            res['v'] = res['v']*1000
            responses.append(res['v'] )

        return responses
elif config.model == 'M1_TTPC_NA_HH':
    def run_model(param_set, stim_name_list, dt, start_Vm=-72):
        model = NeuronModel(mod_dir = './neuron_files/M1_TTPC_NA_HH/')
        model.update_params(param_set)
        volts_list = []
        stims = h5py.File(config.stims_path, 'r')
        
        if type(stim_name_list) != list and type(stim_name_list) != np.ndarray:
            stim_name_list = [stim_name_list]
            
        for curr_stim_name in stim_name_list:
            stim = stims[curr_stim_name][:]
            curr_dt = retrieve_dt(curr_stim_name, stims, dt=dt)
            Vm, I, t, stim = model.run_model_compare(stim, dt=curr_dt, start_Vm=start_Vm)
            volts_list.append(Vm)
        return np.array(volts_list)
    
    def run_model_stim(param_set, stim, dt, mod_dir='./neuron_files/M1_TTPC_NA_HH/', start_Vm=-72):
        model = NeuronModel(mod_dir = mod_dir )
        model.update_params(param_set)
        volts_list = []
        Vm, I, t, stim = model.run_model_compare(stim, dt=dt, start_Vm=start_Vm)
        volts_list.append(Vm)
        return np.array(volts_list)




