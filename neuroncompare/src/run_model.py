import numpy as np
import os
from neuroncompare.src.hoc_utils import decode_list, retrieve_dt
try:
    import neuroncompare.src.run_stim_config as config
except:
    import neuroncompare.src.optim_config as config

os.chdir(config.neuron_path) 
from neuron import h
os.chdir("../../")
from neuroncompare.src.NeuronModelClass import NeuronModel


import h5py

try:
    
    import allensdk.core.json_utilities as ju
    # from biophys_optimize.utils import Utils
    from allensdk.model.biophysical.utils import create_utils
    import allensdk.model.biophysical.runner as runner
except ImportError:
    print('could not import AllenSDK')
    
if 'bbp' in config.model:
    def run_model(param_set, stim_name_list, input_dt=None, start_Vm=None):
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
        run_file = 'neuron_genetic_alg/cell_models/compare_bbp/run_model_cori.hoc'
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
        model = NeuronModel(mod_dir = './cell_models/M1_TTPC_NA_HH/')
        model.update_params(param_set)
        # plot_dendritic_ih(model, save_path='plots/dend.png')
        # import pdb; pdb.set_trace()
        # print(1/0)
        # pps = test_passive_properties(model)
        # import pdb; pdb.set_trace()
        # results = debug_passive_properties(model)
        # plot_passive_responses(results)
        # import pdb; pdb.set_trace()

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
    
    def run_model_stim(param_set, stim, dt, mod_dir='./cell_models/M1_TTPC_NA_HH/', start_Vm=-72):
        model = NeuronModel(mod_dir = mod_dir )
        model.update_params(param_set)
        volts_list = []
        Vm, I, t, stim = model.run_model_compare(stim, dt=dt, start_Vm=start_Vm)
        volts_list.append(Vm)
        return np.array(volts_list)





def test_passive_properties(model):
    """Test passive membrane properties of the model."""
    
    # Dictionary to store results
    results = {
        'resting_vm': {},
        'input_resistance': {},
        'time_constant': {}
    }
    
    # Test sections to analyze
    sections = {
        'soma': model.soma_ref,
        'ais': model.ais,
        'dist_dend': model.dist_dend
    }
    
    for sec_name, sec in sections.items():
        # Record resting Vm
        model.h.finitialize(-70)  # Initialize to -70 mV
        results['resting_vm'][sec_name] = sec(0.5).v
        
        # Calculate input resistance with small hyperpolarizing pulse
        model.init_stim(stim_start=100, stim_dur=100, amp=-0.1)  # -0.1 nA pulse
        vm_base = sec(0.5).v
        model.h.finitialize(-70)
        model.h.continuerun(300)
        vm_step = sec(0.5).v
        
        # Input resistance (MOhm) = ΔV (mV) / I (nA)
        delta_v = vm_step - vm_base
        results['input_resistance'][sec_name] = abs(delta_v / -0.1)
        
        # Calculate time constant with same pulse
        # Time to reach 63% of steady state
        t_start = 100  # ms
        target_v = vm_base + (0.63 * delta_v)
        
        t = t_start
        while t < 200 and sec(0.5).v > target_v:
            model.h.continuerun(t + 0.1)
            t += 0.1
        
        results['time_constant'][sec_name] = t - t_start
    
    return results

    

def debug_passive_properties(model):
    """Enhanced testing of passive properties with detailed debugging."""
    
    h = model.h
    results = {}
    
    # Test sections
    sections = {
        'soma': model.soma_ref,
        'ais': model.ais,
        'dist_dend': model.dist_dend
    }
    
    # Record voltage traces for analysis
    recordings = {}
    for sec_name, sec in sections.items():
        # Create recording vectors
        v_vec = h.Vector()
        t_vec = h.Vector()
        v_vec.record(sec(0.5)._ref_v)
        t_vec.record(h._ref_t)
        recordings[sec_name] = {'v': v_vec, 't': t_vec}
        
        # Print section properties
        print(f"\n{sec_name} properties:")
        print(f"L = {sec.L:.2f} µm")
        print(f"diam = {sec.diam:.2f} µm")
        print(f"cm = {sec.cm:.3f} µF/cm²")
        print(f"g_pas = {sec(0.5).pas.g:.3e} S/cm²")
        print(f"e_pas = {sec(0.5).pas.e:.1f} mV")
        print(f"Ra = {sec.Ra:.1f} Ω·cm")
    
    # Run test with hyperpolarizing pulse
    stim = h.IClamp(model.soma_ref(0.5))
    stim.delay = 100  # ms
    stim.dur = 100    # ms
    stim.amp = -0.1   # nA
    
    h.tstop = 400    # ms
    h.dt = 0.1       # ms
    h.finitialize(-70)
    
    print("\nRunning simulation...")
    while h.t < h.tstop:
        h.fadvance()
        if h.t % 100 == 0:
            print(f"t = {h.t:.1f} ms")
    
    # Analyze results for each section
    for sec_name in sections:
        v = np.array(recordings[sec_name]['v'])
        t = np.array(recordings[sec_name]['t'])
        
        # Find baseline and steady-state
        baseline = np.mean(v[int(90/h.dt):int(100/h.dt)])
        steady_state = np.mean(v[int(190/h.dt):int(200/h.dt)])
        delta_v = steady_state - baseline
        
        # Calculate input resistance
        r_in = abs(delta_v / stim.amp)
        
        # Find time constant (time to reach 63% of steady state)
        target_v = baseline + (0.63 * delta_v)
        stim_start_idx = int(100/h.dt)
        idx_63 = stim_start_idx + np.where(v[stim_start_idx:] <= target_v)[0][0]
        tau = t[idx_63] - 100
        
        results[sec_name] = {
            'baseline': baseline,
            'steady_state': steady_state,
            'delta_v': delta_v,
            'input_resistance': r_in,
            'time_constant': tau,
            'voltage_trace': v,
            'time': t
        }
    
    return results

def plot_passive_responses(results):
    """Plot the voltage responses from passive property testing."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    for sec_name, data in results.items():
        plt.plot(data['time'], data['voltage_trace'], label=sec_name)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('Passive Response to -0.1 nA Current Step')
    plt.grid(True)
    plt.legend()

    
    plt.savefig('tst.png')


def adjust_passive_parameters(params):
    """
    Adjust passive parameters to achieve physiological responses:
    - ~5 mV response to -0.1 nA (input resistance ~50 MΩ)
    - Faster dendritic response
    - Clean exponential decay
    """
    param_set = params.copy()
    
    # Increase gpas to reduce voltage response
    param_set[17] = 2e-4  # gpas_soma
    param_set[18] = 2e-4  # gpas_apical
    param_set[19] = 2e-4  # gpas_basal
    param_set[20] = 2e-4  # gpas_axon
    
    # Set uniform cm
    param_set[21] = 1.0   # cm_all
    param_set[22] = 1.0   # cm_axon
    
    # Set uniform e_pas
    param_set[23] = -70.0  # e_pas_all
    param_set[24] = -70.0  # e_pas_axon
    
    # Additional parameters to check/modify in the model:
    # Ra should be ~100 Ω·cm
    # nseg should be adjusted based on the d_lambda rule
    
    return param_set

    
def adjust_spatial_discretization(model):
    """
    Adjust spatial discretization using d_lambda rule.
    """
    for sec in model.h.allsec():
        # Set Ra to 100 Ω·cm
        sec.Ra = 100
        
        # Apply d_lambda rule
        freq = 100 # Hz, frequency at which AC length constant will be computed
        d_lambda = 0.1 # fraction of length constant for spatial discretization
        
        # Calculate appropriate number of segments
        sec.nseg = int((sec.L/(d_lambda * model.h.lambda_f(freq)) + 0.9)/2)*2 + 1


def print_dendritic_ih(model):
    """
    Prints the Ih conductances for all dendritic compartments of a NeuronModel instance.
    
    Args:
        model: An instance of NeuronModel class
        
    Prints:
        Tables showing Ih conductance values for apical and basal dendrites,
        including section names, positions, distances from soma, and conductance values.
    """
    h = model.h  # Get NEURON h object from the model
    
    print("\nDendritic Ih Conductances:")
    print("-" * 50)
    
    # Print apical dendrite conductances
    print("\nApical Dendrites:")
    print(f"{'Section':<20} {'Position':<10} {'Distance (μm)':<15} {'gIh (S/cm²)':<15}")
    print("-" * 60)
    
    for sec in h.cell.apical:
        for seg in sec:
            # Calculate distance from soma
            h.distance(0, 0.5, sec=h.cell.soma[0])
            dist = h.distance(seg.x, sec=sec)
            
            # Get Ih conductance
            gih = seg.gIhbar_Ih
            
            print(f"{sec.name():<20} {seg.x:<10.2f} {dist:<15.2f} {gih:<15.2e}")
    
    # Print basal dendrite conductances
    print("\nBasal Dendrites:")
    print(f"{'Section':<20} {'Position':<10} {'Distance (μm)':<15} {'gIh (S/cm²)':<15}")
    print("-" * 60)
    
    for sec in h.cell.basal:
        for seg in sec:
            # Calculate distance from soma
            h.distance(0, 0.5, sec=h.cell.soma[0])
            dist = h.distance(seg.x, sec=sec)
            
            # Get Ih conductance
            gih = seg.gIhbar_Ih
            
            print(f"{sec.name():<20} {seg.x:<10.2f} {dist:<15.2f} {gih:<15.2e}")

def plot_dendritic_ih(model, save_path=None):
    """
    Analyzes and plots Ih conductances in dendritic compartments for a given NeuronModel.
    
    Args:
        model: Instance of NeuronModel
        save_path (str, optional): Path to save the plots. If None, plots are displayed.
    
    Returns:
        tuple: (apical_data, basal_data) as numpy arrays with columns [distance, conductance]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    h = model.h
    
    # Collect data
    apical_data = []
    basal_data = []
    
    # Get apical data
    for sec in h.cell.apical:
        for seg in sec:
            # Calculate distance from soma
            h.distance(0, 0.5, sec=h.cell.soma[0])
            dist = h.distance(seg.x, sec=sec)
            gih = seg.gIhbar_Ih
            apical_data.append((dist, gih))
    
    # Get basal data
    for sec in h.cell.basal:
        for seg in sec:
            # Calculate distance from soma
            h.distance(0, 0.5, sec=h.cell.soma[0])
            dist = h.distance(seg.x, sec=sec)
            gih = seg.gIhbar_Ih
            basal_data.append((dist, gih))
    
    # Convert to numpy arrays
    apical_data = np.array(apical_data)
    basal_data = np.array(basal_data)
    
    # Sort by distance for cleaner plotting
    apical_data = apical_data[apical_data[:, 0].argsort()]
    basal_data = basal_data[basal_data[:, 0].argsort()]
    
    # Calculate theoretical curve
    def theoretical_ih(distance, dend_Ih=0.00008):
        """Formula from biophysics.hoc: (-0.869600 + 2.087000*exp((%g-0.000000)*0.003100))*%g"""
        return (-0.8696 + 2.087 * np.exp((distance - 0) * 0.0031)) * dend_Ih
    
    # Generate x values for smooth curve starting from first observed value
    x_theoretical = np.linspace(min(apical_data[:, 0]), max(apical_data[:, 0])*1.1, 1000)
    
    # Calculate shift needed to match first observed point
    first_point_dist = apical_data[0, 0]
    first_point_cond = apical_data[0, 1]
    theo_offset = first_point_cond / theoretical_ih(first_point_dist)
    
    # Adjust theoretical curve to start at first observed point
    y_theoretical = theoretical_ih(x_theoretical) * theo_offset
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Apical dendrites
    ax1.scatter(apical_data[:, 0], apical_data[:, 1], color='blue', alpha=0.6, 
               label='Measured Values', s=30)
    ax1.plot(x_theoretical, y_theoretical, 'r-', label='Theoretical Curve', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Distance from Soma (μm)')
    ax1.set_ylabel('Conductance (S/cm²)')
    ax1.set_title('Apical Dendrite Ih Conductance')
    ax1.legend()
    
    # Plot 2: Basal dendrites
    ax2.scatter(basal_data[:, 0], basal_data[:, 1], color='green', alpha=0.6, 
               label='Measured Values', s=30)
    ax2.axhline(y=basal_data[0, 1], color='r', linestyle='--', 
                label='Constant Value', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Distance from Soma (μm)')
    ax2.set_ylabel('Conductance (S/cm²)')
    ax2.set_title('Basal Dendrite Ih Conductance')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print statistics
    print("\nDendritic Ih Analysis:")
    print("\nApical Dendrites:")
    print(f"Number of compartments: {len(apical_data)}")
    print(f"Distance range: {min(apical_data[:, 0]):.1f} to {max(apical_data[:, 0]):.1f} μm")
    print(f"Conductance range: {min(apical_data[:, 1]):.2e} to {max(apical_data[:, 1]):.2e} S/cm²")
    
    print("\nBasal Dendrites:")
    print(f"Number of compartments: {len(basal_data)}")
    print(f"Distance range: {min(basal_data[:, 0]):.1f} to {max(basal_data[:, 0]):.1f} μm")
    print(f"Conductance: {basal_data[0, 1]:.2e} S/cm² (uniform)")
    
    return apical_data, basal_data