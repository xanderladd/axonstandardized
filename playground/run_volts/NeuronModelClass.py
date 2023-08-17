# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""
import argparse
import numpy as np
# from vm_plotter import *
from neuron import h
import os
import sys
import pdb

class NeuronModel:
    def __init__(self, mod_dir = './neuron_files/M1_TTPC_NA_HH/',#'./Neuron_Model_HH/', 
    
    nav12=1,
                      nav16=1,
                      dend_nav12=1,
                      soma_nav12=1,
                      ais_nav12=1,
                      dend_nav16=1,
                      soma_nav16=1,
                      ais_nav16=1,
                      ais_ca = 1,
                      ais_KCa = 1,
                      axon_Kp=1,
                      axon_Kt =1,
                      axon_K=1,
                      axon_Kca =1,
                      axon_HVA = 1,
                      axon_LVA = 1,
                      node_na = 1,
                      soma_K=1,
                      dend_K=1,
                      gpas_all=1):
        # print(f"nav12={nav12}, nav16={nav16}, dend_nav12={dend_nav12}, soma_nav12={soma_nav12}, ais_nav12={ais_nav12}, "
        # f"dend_nav16={dend_nav16}, soma_nav16={soma_nav16}, ais_nav16={ais_nav16}, ais_ca={ais_ca}, ais_KCa={ais_KCa}, "
        # f"axon_Kp={axon_Kp}, axon_Kt={axon_Kt}, axon_K={axon_K}, axon_Kca={axon_Kca}, axon_HVA={axon_HVA}, "
        # f"axon_LVA={axon_LVA}, node_na={node_na}, soma_K={soma_K}, dend_K={dend_K}, gpas_all={gpas_all}")
        run_dir = os.getcwd()

        os.chdir(mod_dir)
        self.h = h  # NEURON h
        print(f'running model at {os.getcwd()} run dir is {run_dir}')
        #pdb.set_trace()
        h.load_file("runModel.hoc")
        self.soma_ref = h.root.sec
        self.soma = h.secname(sec=self.soma_ref)
        self.sl = h.SectionList()
        self.sl.wholetree(sec=self.soma_ref)
        self.nexus = h.cell.apic[66]
        self.dist_dend = h.cell.apic[91]
        self.ais = h.cell.axon[0]
        self.axon_proper = h.cell.axon[1]
        h.dend_na12 = 0.012/2
        h.dend_na16 = h.dend_na12
        h.dend_k = 0.004226 * soma_K


        h.soma_na12 = 0.983955/10
        h.soma_na16 = h.soma_na12
        h.soma_K = 8.396194779331378477e-02 * soma_K

        h.ais_na16 = 4
        h.ais_na12 = 4
        h.ais_ca = 0.00990*4*ais_ca
        h.ais_KCa = 0.007104*ais_KCa

        h.node_na = 2 * node_na

        h.axon_KP = 0.973538 * axon_Kp
        h.axon_KT = 1.7 * axon_Kt
        h.axon_K = 1.021945 * axon_K
        h.axon_LVA = 0.0014 * axon_LVA
        h.axon_HVA = 0.00012 * axon_HVA
        h.axon_KCA = 1.8 * axon_Kca


        #h.cell.axon[0].gCa_LVAstbar_Ca_LVAst = 0.001376286159287454

        #h.soma_na12 = h.soma_na12/2
        h.naked_axon_na = h.soma_na16/5
        h.navshift = -10
        h.myelin_na = h.naked_axon_na
        h.myelin_K = 0.303472
        h.myelin_scale = 10
        h.gpas_all = 3e-5 * gpas_all
        h.cm_all = 1


        h.dend_na12 = h.dend_na12 * nav12 * dend_nav12
        h.soma_na12 = h.soma_na12 * nav12 * soma_nav12
        h.ais_na12 = h.ais_na12 * nav12 * ais_nav12

        h.dend_na16 = h.dend_na16 * nav16 * dend_nav16
        h.soma_na16 = h.soma_na16 * nav16 * soma_nav16
        h.ais_na16 = h.ais_na16 * nav16 * ais_nav16
        h.working()
        os.chdir(run_dir)


    def init_stim(self, sweep_len = 800, stim_start = 100, stim_dur = 500, amp = 0.3, dt = 0.1):
        # updates the stimulation params used by the model
        # time values are in ms
        # amp values are in nA

        h("st.del = " + str(stim_start))
        h("st.dur = " + str(stim_dur))
        h("st.amp = " + str(amp))
        h.tstop = sweep_len
        h.dt = dt
        
    def update_params(self, params):
        if type(params) == dict:
            h.dend_na12 = params['dend_na12']
            h.soma_na12 = params['soma_na12']
            h.ais_na12 = params['ais_na12']
            h.dend_na16 = params['dend_na16']
            h.soma_na16 = params['soma_na16']
            h.ais_na16 = params['ais_na16']
            h.ais_ca = params['ais_ca']
            h.ais_KCa = params['ais_KCa']
            h.axon_KP = params['axon_KP']
            h.axon_KT = params['axon_KT']
            h.axon_K = params['axon_K']
            h.axon_KCA = params['axon_KCa']
            h.axon_HVA = params['axon_HVA']
            h.axon_LVA = params['axon_LVA']
            h.node_na = params['node_na']
            h.soma_K = params['soma_K']
            h.dend_k = params['dend_K']
            h.gpas_all = params['gpas_all']
            h.cm_all = params['cm_all']
        elif type(params) == list or type(params) == np.ndarray:
            h.dend_na12 = params[0]
            h.soma_na12 = params[1]
            h.ais_na12 = params[2]
            h.dend_na16 = params[3]
            h.soma_na16 = params[4]
            h.ais_na16 = params[5]
            h.ais_ca = params[6]
            h.ais_KCa = params[7]
            h.axon_KP = params[8]
            h.axon_KT = params[9]
            h.axon_K = params[10]
            h.axon_KCA = params[11]
            h.axon_HVA = params[12]
            h.axon_LVA = params[13]
            h.node_na = params[14]
            h.soma_K = params[15]
            h.dend_k = params[16]
            h.gpas_all = params[17]
            h.cm_all = params[18]
        h.working()

    def run_model_compare(self, stim, dt, start_Vm=-72):
        # updates the stimulation params used by the model
        # time values are in ms
        # amp values are in nA
        # clamp = h.st
        h.dt = dt
        h.finitialize(start_Vm)
        clamp = h.IClamp(h.cell.soma[0](0.5))
        # h.st = clamp
        # stim_vec = h.Vector(stim)
        # stim_vec.play(clamp._ref_amp, dt)
        # # stim_vec.play(clamp.amp, dt)
        clamp.delay = 0
        clamp.dur = 1e9
#         h.tstop = len(stim) / (h.steps_per_ms * dt)
        
#         v = h.Vector().record(h.cell.soma[0](.5)._ref_v)             # Membrane potential vector
#         t = h.Vector().record(h._ref_t)                     # Time stamp vector
#         i = h.Vector().record(clamp._ref_amp)
        
        v = []
        t = []
        for timestep in range(len(stim)):
            h.dt = dt
            clamp.amp = stim[timestep]
            h.fadvance()
            v.append(h.cell.soma[0].v)
            t.append(dt)
            # print(h.cell.soma[0].v, timestep)
            
        return v, stim, t, stim
        

        
    def run_model(self, start_Vm = -72, dt= 0.1,rec_extra = False):
        h.dt=dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop/h.dt)
        Vm = np.zeros(timesteps)
        I = {}
        I['Na'] = np.zeros(timesteps)
        I['Ca'] = np.zeros(timesteps)
        I['K'] = np.zeros(timesteps)
        stim = np.zeros(timesteps)
        t = np.zeros(timesteps)
        if rec_extra:
            
            extra_Vms = {}
            extra_Vms['ais'] = np.zeros(timesteps)
            extra_Vms['nexus'] = np.zeros(timesteps)
            extra_Vms['dist_dend'] = np.zeros(timesteps)
            extra_Vms['axon'] = np.zeros(timesteps)
            
        for i in range(timesteps):
            Vm[i] = h.cell.soma[0].v
            I['Na'][i] = h.cell.soma[0](0.5).ina
            I['Ca'][i] = h.cell.soma[0](0.5).ica
            I['K'][i] = h.cell.soma[0](0.5).ik
            stim[i] = h.st.amp
            t[i] = i*h.dt / 1000
            if rec_extra:
                nseg = int(self.h.L/10)*2 +1  # create 19 segments from this axon section
                ais_end = 10/nseg # specify the end of the AIS as halfway down this section
                ais_mid = 4/nseg # specify the middle of the AIS as 1/5 of this section 
                extra_Vms['ais'][i] = self.ais(ais_mid).v
                extra_Vms['nexus'][i] = self.nexus(0.5).v
                extra_Vms['dist_dend'][i] = self.dist_dend(0.5).v
                extra_Vms['axon'][i]=self.axon_proper(0.5).v
            h.fadvance()
        if rec_extra:
            return Vm, I, t, stim,extra_Vms
        else:
            return Vm, I, t, stim

    def run_sim_model(self,start_Vm = -72, dt= 0.1,sim_config = {
                'section' : 'soma',
                'section_num' : 0,
                'segment' : 0.5,
                'currents'  :['ina','ica','ik'],
                'ionic_concentrations' :["cai", "ki", "nai"]
            }):
         
        """
        Runs a simulation model and returns voltage, current, time, and stimulation data.

        Args:
            start_Vm (float): Initial membrane potential (default: -72 mV).
            dt (float): Time step size for the simulation (default: 0.1 ms).
            sim_config (dict): Configuration dictionary for simulation parameters (default: see below).

        Returns:
            Vm (ndarray): Recorded membrane voltages over time.
            I (dict): Current traces for different current types.
            t (ndarray): Time points corresponding to the recorded data.
            stim (ndarray): Stimulation amplitudes over time.

        Description:
            This function runs a simulation model and records the membrane voltage, current traces, time points,
            and stimulation amplitudes over time. The simulation model is configured using the provided parameters.

        Default Simulation Configuration:
            'section': 'soma'
            'segment': 0.5
            'section_num' : 0
            'currents'  :['ina','ica','ik'],
            'ionic_concentrations' :["cai", "ki", "nai"]

        Example Usage:
            Vm, I, t, stim = run_sim_model(start_Vm=-70, dt=0.05, sim_config={
                'section': 'soma',
                'section_num' : 0,
                'segment': 0.5,
                'currents'  :['ina','ica','ik'],
                'ionic_concentrations' :["cai", "ki", "nai"]
            })
        """
        
        h.dt=dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop/h.dt)
        #initialise to zeros,
        #current_types = list(set(sim_config['inward'] + sim_config['outward']))
        current_types = sim_config['currents']
        ionic_types = sim_config['ionic_concentrations']
        Vm = np.zeros(timesteps, dtype=np.float64)
        I = {current_type: np.zeros(timesteps, dtype=np.float64) for current_type in current_types}
        ionic = {ionic_type : np.zeros(timesteps,dtype=np.float64) for ionic_type in ionic_types}
        #print(f"I : {I}")
        stim = np.zeros(timesteps, dtype=np.float64)
        t = np.zeros(timesteps, dtype=np.float64)
        section = sim_config['section']
        section_number = sim_config['section_num']
        segment = sim_config['segment']
        volt_var  = "h.cell.{section}[{section_number}]({segment}).v".format(section=section, section_number=section_number,segment=segment)
        #print(eval("h.psection()"))
        #print(h("topology()"))
        #val = eval("h.cADpyr232_L5_TTPC1_0fb1ca4724[0].soma[0](0.5).na12mut.ina_ina")
        #print(f"na16 mut {val}")
        curr_vars={}
        # for current_type in current_types:
        #     if current_type == 'ina_ina_na12':
        #         curr_vars[current_type] =  "h.cell.{section}[0].{current_type}".format(section=section, segment=segment, current_type=current_type) 
        #     else:
        #         curr_vars[current_type] = "h.cell.{section}[0]({segment}).{current_type}".format(section=section, segment=segment, current_type=current_type) 
        curr_vars = {current_type : "h.cell.{section}[{section_number}]({segment}).{current_type}".format(section=section, section_number=section_number, segment=segment, current_type=current_type) for current_type in current_types}
        print(f"current_vars : {curr_vars}")
        ionic_vars = {ionic_type : "h.cell.{section}[{section_number}]({segment}).{ionic_type}".format(section=section , section_number=section_number, segment=segment, ionic_type=ionic_type) for ionic_type in ionic_types}
        #print(f"ionic_vars : {ionic_vars}")
        for i in range(timesteps):
            Vm[i] =eval(volt_var)
            try :
                for current_type in current_types:
                    I[current_type][i] = eval(curr_vars[current_type])

                #getting the ionic concentrations
                for ionic_type in ionic_types:
                    ionic[ionic_type][i] = eval(ionic_vars[ionic_type])
            except Exception as e:
                print(e)
                print("Check the config files for the correct Attribute")
                sys.exit(1)

            stim[i] = h.st.amp
            t[i] = i*h.dt / 1000
            h.fadvance()
        #print(f"I : {I}")
        return Vm, I, t, stim, ionic
    
  
#######################
# MAIN
#######################



"""

  def run_sim_model(self, start_Vm=-72, dt=0.1, sim_config=None):     
        if sim_config is None:
            sim_config = {
                'section': 'soma',
                'segment': 0.5,
                'inward': ['ina', 'ica'],
                'outward': ['ik']
            }

        h.dt = dt
        h.finitialize(start_Vm)
        timesteps = int(h.tstop / h.dt)
        current_types = list(set(sim_config['inward'] + sim_config['outward']))
        Vm_vec = h.Vector()
        I_vecs = {current_type: h.Vector() for current_type in current_types}
        stim_vec = h.Vector()
        t_vec = h.Vector()

        section = sim_config['section']
        segment = sim_config['segment']
        compt_sec = getattr(h.cell, section)[0](segment)

        Vm_vec.record(compt_sec._ref_v)
        for current_type in current_types:
            curr_var = getattr(compt_sec, "_ref_{current_type}".format(current_type=current_type))
            I_vecs[current_type].record(curr_var)

        stim_vec.record(h.st._ref_amp)
        t_vec.record(h._ref_t)

        h.frecord_init()  # Enable recording of state variables
        h.continuerun(timesteps)  # Run the simulation

        Vm = np.array(Vm_vec)
        I = {current_type: np.array(I_vecs[current_type]) for current_type in current_types}
        stim = np.array(stim_vec)
        t = np.array(t_vec) / 1000.0

        return Vm, I, t, stim


"""