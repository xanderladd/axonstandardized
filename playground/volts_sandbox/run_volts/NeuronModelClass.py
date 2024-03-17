# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:07:44 2021

@author: bensr
"""
import argparse
import numpy as np
# from vm_plotter import *
# from neuron import h
import config
import os
os.chdir(config.neuron_path) 
from neuron import h
os.chdir("../../")
import os
import sys
import pdb
import currentscape


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
        
#     def update_params(self, params):
#         if type(params) == dict:
#             h.dend_na12 = params['dend_na12']
#             h.soma_na12 = params['soma_na12']
#             h.ais_na12 = params['ais_na12']
#             h.dend_na16 = params['dend_na16']
#             h.soma_na16 = params['soma_na16']
#             h.ais_na16 = params['ais_na16']
#             h.ais_ca = params['ais_ca']
#             h.ais_KCa = params['ais_KCa']
#             h.axon_KP = params['axon_KP']
#             h.axon_KT = params['axon_KT']
#             h.axon_K = params['axon_K']
#             h.axon_KCA = params['axon_KCa']
#             h.axon_HVA = params['axon_HVA']
#             h.axon_LVA = params['axon_LVA']
#             h.node_na = params['node_na']
#             h.soma_K = params['soma_K']
#             h.dend_k = params['dend_K']
#             h.gpas_all = params['gpas_all']
#             h.cm_all = params['cm_all']
#         elif type(params) == list or type(params) == np.ndarray:
#             h.dend_na12 = params[0]
#             h.soma_na12 = params[1]
#             h.ais_na12 = params[2]
#             h.dend_na16 = params[3]
#             h.soma_na16 = params[4]
#             h.ais_na16 = params[5]
#             h.ais_ca = params[6]
#             h.ais_KCa = params[7]
#             h.axon_KP = params[8]
#             h.axon_KT = params[9]
#             h.axon_K = params[10]
#             h.axon_KCA = params[11]
#             h.axon_HVA = params[12]
#             h.axon_LVA = params[13]
#             h.node_na = params[14]
#             h.soma_K = params[15]
#             h.dend_k = params[16]
#             h.gpas_all = params[17]
#             # h.cm_all = params[18]
            
#             h.cell.soma[0].cm = params[18]
#             h.cell.axon[0].cm = params[18]
#             h.cell.axon[1].cm = params[18]
#             for i in range(len(h.cell.dend)):
#                 h.cell.dend[i].cm = params[18]
                
#             if len(params) > 19:
#                 h.cell.soma[0].e_pas = params[19]
#                 h.cell.axon[0].e_pas = params[19]
#                 h.cell.axon[1].e_pas = params[19]
#                 for i in range(len(h.cell.dend)):
#                     h.cell.dend[i].e_pas = params[19]
                    
#                 h.cell.soma[0].g_pas = params[17]
#                 h.cell.axon[0].g_pas = params[17]
#                 h.cell.axon[1].g_pas = params[17]
#                 for i in range(len(h.cell.dend)):
#                     h.cell.dend[i].g_pas = params[17]
                    
#                 h.cell.soma[0].gIhbar_Ih = params[20]
#                 for i in range(len(h.cell.dend)):
#                     h.cell.dend[i].gIhbar_Ih = params[20]
                    
#         h.working()
        
    def update_params(self, params):
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
        
        # SWITCH between to current model
        if len(params) > 19:
            
            ############ GPAS
            for sec in h.cell.soma:
                for seg in sec:
                    seg.g_pas = params[17]
                    seg.cm = params[18]
                    seg.e_pas = params[19]
                    seg.gIhbar_Ih = params[20]

            for sec in h.cell.apical:
                for seg in sec:
                    seg.g_pas = params[17]
                    seg.cm = params[18]
                    seg.e_pas = params[19]
                    seg.gIhbar_Ih = params[20]

            for sec in h.cell.basal:
                for seg in sec:
                    seg.g_pas = params[17]
                    seg.cm = params[18]
                    seg.e_pas = params[19]
                    seg.gIhbar_Ih = params[20]
                    

            for sec in h.cell.axon:
                for seg in sec:
                    seg.g_pas = params[17]
                    seg.cm = params[18]
                    seg.e_pas = params[19]
            
        h.working()
        # import pdb; pdb.set_trace()
        for sec in h.cell.axon:
            for seg in sec:
                seg.g_pas =params[17]
                
        for sec in h.cell.axon:
            for seg in sec:
                seg.cm = params[18]
        
                
    def run_model_compare(self, stim, dt, start_Vm=-72):
        # updates the stimulation params used by the model
        # time values are in ms
        # amp values are in nA
        # clamp = h.st
        h.dt = dt
        h.finitialize(start_Vm)
        clamp = h.IClamp(h.cell.soma[0](0.5))
        clamp.delay = 0
        clamp.dur = 1e9
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
        # print( "P SECTION", eval("h.psection()"))
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
    
    
    def run_model_compare_cs(self, stim, dt, start_Vm=-72, sim_config = {
                'section' : 'soma',
                'section_num' : 0,
                'segment' : 0.5,
                'currents'  :['ina','ica','ik'],
                'ionic_concentrations' :["cai", "ki", "nai"]
            }):
        # updates the stimulation params used by the model
        # time values are in ms
        # amp values are in nA
        # clamp = h.st        
        h.dt = dt
        h.finitialize(start_Vm)
        clamp = h.IClamp(h.cell.soma[0](0.5))
        clamp.delay = 0
        clamp.dur = 1e9
        timesteps = len(stim)
        v = []
        t = []
        current_types = sim_config['currents']
        ionic_types = sim_config['ionic_concentrations']
        
        I = {current_type: np.zeros(timesteps, dtype=np.float64) for current_type in current_types}
        
        ionic = {ionic_type : np.zeros(timesteps,dtype=np.float64) for ionic_type in ionic_types}
        
        section = sim_config['section']
        section_number = sim_config['section_num']
        segment = sim_config['segment']
                
        volt_var  = "h.cell.{section}[{section_number}]({segment}).v".format(section=section, section_number=section_number,segment=segment)
        curr_vars={}
        curr_vars = {current_type : "h.cell.{section}[{section_number}]({segment}).{current_type}".format(section=section, section_number=section_number, segment=segment, current_type=current_type) for current_type in current_types}
        print(f"current_vars : {curr_vars}")
        ionic_vars = {ionic_type : "h.cell.{section}[{section_number}]({segment}).{ionic_type}".format(section=section , section_number=section_number, segment=segment, ionic_type=ionic_type) for ionic_type in ionic_types}
        
        
        for timestep in range(len(stim)):
            h.dt = dt
            clamp.amp = stim[timestep]
            h.fadvance()
            v.append(h.cell.soma[0].v)
            t.append(dt)
            
            for current_type in current_types:

                I[current_type][timestep] = eval(curr_vars[current_type])
            
            for ionic_type in ionic_types:
                ionic[ionic_type][timestep] = eval(ionic_vars[ionic_type])
            # print(h.cell.soma[0].v, timestep)
        return v, I, t, stim, ionic
        
    
    def make_currentscape_plot(self,amp,time1,time2,stim_start =100,sweep_len=800,sim_config = {
                    'section' : 'soma',
                    'segment' : 0.5, ##0.5 should be half way down AIS
                    'section_num': 0,
                    'currents'  : ['ihcn_Ih','ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'], ##Currents must be present in .mod files
                    #'currents'  :['ina','ica','ik'], ##Example if you have fewer currents
                    'ionic_concentrations' :["cai", "ki", "nai"]
            }):

            current_names = ['Ih','Ca_HVA','Ca_LVAst','SKv3_1','SK_E2','Na16 WT','Na16 WT','Na12','Na12 MUT','pas'] ##Current names in order of 'currents'in sim_config. Or if you don't want different curent names, use- current_names = sim_config['currents']


            #amp = 0.5 ##Modify stimulus current
            #sweep_len = 800 ##Modify this to change total length of recording
            self.init_stim(stim_start =stim_start,amp=amp,sweep_len = sweep_len) ##Modify stim_start to change with the stimulus starts. Helpful when looking at single APs
            Vm, I, t, stim, ionic = self.run_sim_model(dt=0.01,sim_config=sim_config) ##un the model, dt is timesteps


            ##### Below for plotting user-specified time steps
            dt = 0.01 ##dt used for calculating time steps
            # time1 = 51 ##start time. Must be between 0 < x < sweep_len
            # time2 = 60 ##end time. Must be between 0 < x < sweep_len
            step1 = int((time1/dt))
            step2 = int((time2/dt))
            Vmsteplist = Vm[step1:step2] ##assign new list for range selected between two steps
            maxvm = max(Vm[step1:step2]) ##gets max voltage
            indexmax = Vmsteplist.argmax() ##gets index (time point in Vmsteplist) where max voltage is
            #####
            self.plot_folder = 'plots'
            self.pfx = 'pfx'


            plot_config = {
                "output": {
                    "savefig": True,
                    #"dir": "./Plots/12HMM16HH_TF/SynthMuts_120523/Currentscape/", ##can hardcode output directory path
                    "dir": f"{self.plot_folder}",
                    #"fname": "Na12_mut22_1nA_800ms", ##Change file name here
                    "fname":f"{self.pfx}_{amp}_{sweep_len}",
                    "extension": "pdf", ##choose pdf or other image extension
                    #"extension": "jpg",
                    "dpi": 600,
                    "transparent": False},

                "show":{#"total_contribution":True, ##adds pie charts for overall contribution of currents over full recording
                        #"all_currents":True, ##adds line plots below currents to to show currents over time (rather than just percentage of total)
                        "currentscape": True}, ##Shows currentscape

                "colormap": {"name":"colorbrewer.qualitative.Paired_10"}, ##Can change color pallets. The _# means how many colors in that pallette. If you don't have enough colors for each current to have unique color, some currents will not be displayed
                #"colormap": {"name":"cartocolors.qualitative.Prism_10"},
                #"colormap": {"name":"cmocean.diverging.Balance_10"},

                "xaxis":{"xticks":[25,50,75],
                         "gridline_width":0.2,},

                "current": {"names": current_names,
                            "reorder":False,
                            # "autoscale_ticks_and_ylim":False,
                            # "ticks":[0.00001, 0.001, 0.1], 
                            # "ylim":[0.00001,0.01] #yaxis lims[min,max]
                            },

                "ions":{"names": ["ca", "k", "na"], ##Ionic currents to be displayed at bottom of plot
                        "reorder": False},

                "voltage": {"ylim": [-90, 50]},
                "legendtextsize": 5,
                "adjust": {
                    "left": 0.15,
                    "right": 0.8,
                    "top": 1.0,
                    "bottom": 0.0
                    }
                }


            print(f"The max voltage value is {maxvm}")        
            print(f"The index at which the max voltage happens is {indexmax}")

            #fig = plot_currentscape(Vm, [I[x] for x in I.keys()], plot_config,[ionic[x] for x in ionic.keys()]) ##Default version that plots full sweep_len (full simulation)
            fig = currentscape.plot(Vm[step1:step2], [I[x][step1:step2] for x in I.keys()], plot_config,[ionic[x][step1:step2] for x in ionic.keys()]) ##Use this version to add time steps, must include time1 and time2 above
            #fig = plot_currentscape(Vm[step1:step2], [I[x][step1:step2] for x in I.keys()], plot_config) ##Plots time step version but removes ionic currents plot at bottom




            ###### Writing all raw data to csv######
            # with open("./Plots/12HH16HMM_TF/111423/Currentscape/Na16_WT_1na_75ms_rawdata.csv",'w',newline ='') as csvfile:
            #     writer = csv.writer(csvfile, delimiter = ',')
            #     #writer.writerow(current_names)
            #     writer.writerow(I.keys())

            #     writer.writerows(I[x] for x in I) ##This and line below for writing data from entire sweep_len
            #     writer.writerow(Vm)

                # writer.writerows(I[x][step1:step2] for x in I) ##This and below line are used when time steps are used
                # writer.writerow(Vm[step1:step2])

    def make_currentscape_plot_stim(self,stim, dt, start_Vm=-72, sim_config = {
                    'section' : 'soma',
                    'segment' : 0.5, ##0.5 should be half way down AIS
                    'section_num': 0,
                    'currents'  : 
        ['cm','gCa_HVAbar_Ca_HVA','gSKv3_1bar_SKv3_1','gSK_E2bar_SK_E2','gCa_LVAstbar_Ca_LVAst','gIhbar_Ih','gamma_CaDynamics_E2','gbar_na12','gbar_na12mut','gbar_na16', 'gbar_na16mut', 'g_pas', 'e_pas'],
        # ['ihcn_Ih','ica_Ca_HVA','ica_Ca_LVAst','ik_SKv3_1','ik_SK_E2','na16.ina_ina','na16mut.ina_ina','na12.ina_ina','na12mut.ina_ina','i_pas'], ##Currents must be present in .mod files
                    #'currents'  :['ina','ica','ik'], ##Example if you have fewer currents
                    'ionic_concentrations' :["cai", "ki", "nai"]
            }):
            
            current_names = sim_config['currents'] ##Current names in order of 'currents'in sim_config. Or if you don't want different curent names, use- current_names = sim_config['currents']


            #amp = 0.5 ##Modify stimulus current
            #sweep_len = 800 ##Modify this to change total length of recording
            # self.init_stim(stim_start =stim_start,amp=amp,sweep_len = sweep_len) ##Modify stim_start to change with the stimulus starts. Helpful when looking at single APs
            Vm, I, t, stim, ionic = self.run_model_compare_cs(stim, dt=dt, start_Vm=start_Vm, sim_config=sim_config)
            
            Vm = np.array(Vm)
            

            ##### Below for plotting user-specified time steps
            # time1 = 51 ##start time. Must be between 0 < x < sweep_len
            # time2 = 60 ##end time. Must be between 0 < x < sweep_len
            time1  = 100
            time2  = len(stim) - 100
            step1 = int((time1/dt))
            step2 = int((time2/dt))
            Vmsteplist = Vm[step1:step2] ##assign new list for range selected between two steps
            maxvm = max(Vm[step1:step2]) ##gets max voltage
            indexmax = Vmsteplist.argmax() ##gets index (time point in Vmsteplist) where max voltage is
            #####
            self.plot_folder = 'plots'
            self.pfx = 'pfx'


            plot_config = {
                "output": {
                    "savefig": True,
                    #"dir": "./Plots/12HMM16HH_TF/SynthMuts_120523/Currentscape/", ##can hardcode output directory path
                    "dir": f"{self.plot_folder}",
                    #"fname": "Na12_mut22_1nA_800ms", ##Change file name here
                    "fname":f"{self.pfx}_34_{len(stim)}",
                    "extension": "pdf", ##choose pdf or other image extension
                    #"extension": "jpg",
                    "dpi": 600,
                    "transparent": False},

                "show":{#"total_contribution":True, ##adds pie charts for overall contribution of currents over full recording
                        #"all_currents":True, ##adds line plots below currents to to show currents over time (rather than just percentage of total)
                        "currentscape": True}, ##Shows currentscape

                "colormap": {"name":"colorbrewer.qualitative.Paired_12"}, ##Can change color pallets. The _# means how many colors in that pallette. If you don't have enough colors for each current to have unique color, some currents will not be displayed
                #"colormap": {"name":"cartocolors.qualitative.Prism_10"},
                #"colormap": {"name":"cmocean.diverging.Balance_10"},

                "xaxis":{"xticks":[25,50,75],
                         "gridline_width":0.2,},

                "current": {"names": current_names,
                            "reorder":False,
                            # "autoscale_ticks_and_ylim":False,
                            # "ticks":[0.00001, 0.001, 0.1], 
                            # "ylim":[0.00001,0.01] #yaxis lims[min,max]
                            },

                "ions":{"names": ["ca", "k", "na"], ##Ionic currents to be displayed at bottom of plot
                        "reorder": False},

                "voltage": {"ylim": [-90, 50]},
                "legendtextsize": 5,
                "adjust": {
                    "left": 0.15,
                    "right": 0.8,
                    "top": 1.0,
                    "bottom": 0.0
                    }
                }


            print(f"The max voltage value is {maxvm}")        
            print(f"The index at which the max voltage happens is {indexmax}")

            #fig = plot_currentscape(Vm, [I[x] for x in I.keys()], plot_config,[ionic[x] for x in ionic.keys()]) ##Default version that plots full sweep_len (full simulation)
            fig = currentscape.plot(Vm[step1:step2], [I[x][step1:step2] for x in I.keys()], plot_config,[ionic[x][step1:step2] for x in ionic.keys()]) ##Use this version to add time steps, must include time1 and time2 above






    
