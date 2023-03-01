import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import h5py
import pickle
import bluepyopt
from neuron import h
from scalebary import add_scalebar



my_dpi = 96

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

tick_major = 6
tick_minor = 4
plt.rcParams["xtick.major.size"] = tick_major
plt.rcParams["xtick.minor.size"] = tick_minor
plt.rcParams["ytick.major.size"] = tick_major
plt.rcParams["ytick.minor.size"] = tick_minor

font_small = 12
font_medium = 13
font_large = 14
plt.rc('font', size=font_small)          # controls default text sizes
plt.rc('axes', titlesize=font_medium)    # fontsize of the axes title
plt.rc('axes', labelsize=font_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_small)    # fontsize of the tick labels
plt.rc('legend', fontsize=font_small)    # legend fontsize
plt.rc('figure', titlesize=font_large)   # fontsize of the figure title

ntimestep = 10000
dt = 0.02
def_times = np.array([dt for i in range(ntimestep)])
def_times = np.cumsum(def_times)
def readParamsCSV(fileName):
    fields = ['Param name', 'Base value']
    df = pd.read_csv(fileName, skipinitialspace=True, usecols=fields)
    paramsList = [tuple(x) for x in df.values]
    return paramsList

def cm_to_in(cm):
    return cm/2.54
def plot_indv_section(axs, param_names,params_dist,sect_name,bar_size,xlim=[0,1]):
    y_pos = np.arange(len(param_names))
    axs.barh(y_pos,params_dist, align='center', linestyle='-', color='black',height=bar_size)
    axs.set_yticks(y_pos)
    axs.set_yticklabels(param_names)
    axs.invert_yaxis()  # labels read top-to-bottom
    #plt.axvline(x=0, color='black', linewidth=0.4 ,linestyle='--')
    axs.set_xlim(xlim)
    #axs.set_xticks([0, 1])
    axs.set_title(sect_name,loc='left',fontweight="bold")
    
def final_indv_plot_by_section(param_names, final_best_indv, title, file_path_to_save=None, max_xtic=1, vert_size=10,dend_inds = range(0,4),axon_inds = range(4,12),soma_inds = range(12,19),xlim = [0,1]):
    ax_axon = None
    param_names = replace_param_names(param_names)
    fig_ga = plt.figure(figsize=(cm_to_in(12), cm_to_in(7)))
    fig_ga.subplots_adjust(wspace=0.5)
    if(axon_inds is not None):
        gs = fig_ga.add_gridspec(2, 2,height_ratios=[len(soma_inds),len(dend_inds)])
        ax_axon = fig_ga.add_subplot(gs[:, 1])
        ax_soma = fig_ga.add_subplot(gs[0, 0])
        ax_dend = fig_ga.add_subplot(gs[1, 0])
        axon_param_names = [param_names[i] for i in axon_inds]
        axon_dist_params = [final_best_indv[i] for i in axon_inds]
        plot_indv_section(ax_axon,axon_param_names,axon_dist_params,'Axonal',0.5)
    else:
        gs = fig_ga.add_gridspec(2, 1,height_ratios=[len(soma_inds),len(dend_inds)])
        ax_soma = fig_ga.add_subplot(gs[0])
        ax_dend = fig_ga.add_subplot(gs[1])
    #print(f' soma_inds are {somatic_inds} param_names are {param_names}')
    somatic_param_names = [param_names[i] for i in soma_inds]
    somatic_dist_params = [final_best_indv[i] for i in soma_inds]
    plot_indv_section(ax_soma,somatic_param_names,somatic_dist_params,'Somatic',0.8,xlim)
    #ax_soma.set_frame_on(False)
    ax_soma.axes.get_xaxis().set_visible(False)
    ax_soma.spines['bottom'].set_visible(False)
    
    #ax_soma.set_frame_on(False)
    dend_param_names = [param_names[i] for i in dend_inds]
    dend_dist_params = [final_best_indv[i] for i in dend_inds]
    plot_indv_section(ax_dend,dend_param_names,dend_dist_params,'Dendritic',0.8,xlim)
    ax_soma.set_xlim(xlim)
    ax_dend.set_xlim(xlim)
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")
    #somatic_names = final_best_indv(param_names[somatic_inds])
    #dend_names = final_best_indv(param_names[dend_inds])
    #axon_names = final_best_indv(param_names[axon_inds])
    return [ax_soma,ax_dend,ax_axon]

def final_indv_plot(param_names, final_best_indv, title, file_path_to_save=None, max_xtic=1, vert_size=10):
    plt.figure(figsize=(cm_to_in(8.5), cm_to_in(vert_size)))
    ax = plt.gca()
    y_pos = np.arange(len(param_names))
    ax.barh(y_pos, final_best_indv, height=0.5, linestyle='-', color='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.axvline(x=0, color='black', linewidth=0.4 ,linestyle='--')
    ax.set_xlim(0, max_xtic)
    ax.set_xticks([0, max_xtic])
    ax.set_ylabel('Parameters')
    ax.set_xlabel('Normalized Distance')
    ax.set_title('Deviation From Truth Value ' + title)
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")

def final_indv_plot_vert(param_names, final_best_indv, title, file_path_to_save=None, max_xtic=1, vert_size=10):
    plt.figure(figsize=(cm_to_in(15), cm_to_in(7)))
    ax = plt.gca()
    x_pos = np.arange(len(param_names))
    ax.bar(x_pos,final_best_indv, linestyle='-', color='black',width=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(param_names)
    #ax.invert_yaxis()  # labels read top-to-bottom
    #plt.axvline(x=0, color='black', linewidth=0.4 ,linestyle='--')
    ax.set_ylim(0, max_xtic)
    ax.set_yticks([0, max_xtic])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Normalized Distance')
    ax.set_title('Deviation From Truth Value ' + title)
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")
    
# Code for optimization results analysis
def read_and_normalize_with_neg(opt_result_path, base, lower_bounds, upper_bounds):
    with open(opt_result_path, 'rb') as f:
        best_indvs = pickle.load(f, encoding = "latin1")
    normalized_indvs = []
    for i in range(len(best_indvs)):
        normalized = []
        for j in range(len(best_indvs[i])):
            if (best_indvs[i][j] < base[j]):
                new_value = abs((best_indvs[i][j] - base[j])/(upper_bounds[j] - base[j]))
                normalized.append(new_value)
            else:
                new_value = abs((best_indvs[i][j] - base[j])/(upper_bounds[j] - base[j]))
                normalized.append(new_value)
        normalized_indvs.append(normalized)
    return normalized_indvs, best_indvs


def plot_stim_volts_pair(stim, volts, title_stim, title_volts, file_path_to_save=None,times=def_times):
    fig,axs = plt.subplots(2,figsize=(cm_to_in(8),cm_to_in(7.8)),gridspec_kw={'height_ratios': [1, 8],'wspace': 0.05})
    axs[0].set_title(title_stim)
    axs[0].plot(times,stim, color='black', linewidth=0.25)
    axs[0].locator_params(axis='x', nbins=5)
    axs[0].locator_params(axis='y', nbins=5)
    
    add_scalebar(axs[0])
    #=axs[0].set_title('Voltage Response '+title_volts)
    volts_target = volts[0]
    if len(volts)>1:
        volts_best_response = volts[1]
        axs[1].plot(times,volts_best_response, label='response', color='red',linewidth=1)
    
    
    axs[1].plot(times,volts_target, label='target', color='black',linewidth=1)
    
    axs[1].locator_params(axis='x', nbins=5)
    axs[1].locator_params(axis='y', nbins=8)
    add_scalebar(axs[1])
    
    #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    #plt.tight_layout(pad=1)
    if file_path_to_save:
        plt.savefig(file_path_to_save+'.pdf', format='pdf', dpi=my_dpi, bbox_inches="tight")
    return fig,axs
        
# Running a single volt
def run_single_volts(param_set, stim_data, ntimestep = 10000, dt = 0.02):
    run_file = './neuron_genetic_alg/neuron_files/bbp/run_model_cori.hoc'
    h.load_file(run_file)
    total_params_num = len(param_set)
    timestamps = np.array([dt for i in range(ntimestep)])
    h.curr_stim = h.Vector().from_python(stim_data)
    h.transvec = h.Vector(total_params_num, 1).from_python(param_set)
    h.stimtime = h.Matrix(1, len(timestamps)).from_vector(h.Vector().from_python(timestamps))
    h.ntimestep = ntimestep
    h.runStim()
    out = h.vecOut.to_python()
    return np.array(out),np.cumsum(timestamps)

def plot_comb_scores(opt_path, score_path, title, plot_save_path=None):
    opt_result = h5py.File(opt_path)
    print(opt_result.keys())
    ordered_score_function_list = [e.decode('ascii') for e in opt_result['ordered_score_function_list'][:]]
    optimization_stim_names = [e.decode('ascii') for e in opt_result['opt_stim_name_list'][:]]
    optimization_weightes = opt_result['opt_weight_list'][:]
    best_stims_score_list = []
    for score_name in optimization_stim_names:
        curr_score_data = h5py.File(score_path+score_name+'_scores.hdf5', 'r')
        for sf in ordered_score_function_list:
            curr_stim_sf_pair = curr_score_data['norm_pin_scores_'+sf][:]
            best_stims_score_list.append(curr_stim_sf_pair)
    combined_score = sum([best_stims_score_list[i]*optimization_weightes[i] for i in range(len(optimization_weightes))])
    plt.figure(figsize=(cm_to_in(8.5), cm_to_in(5)))
    plt.title(title)
    plt.xlabel('Parameter Set Rank')
    plt.ylabel('Weighted Score')
    time_step = range(len(combined_score))
    plt.scatter(time_step, combined_score, s=1, color='black')
    if plot_save_path:
        plt.savefig(plot_save_path+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")
def plot_comb_scores_full(ordered_score_function_list, optimization_stim_names, optimization_weightes, score_path, title, plot_save_path=None):
    best_stims_score_list = []
    for score_name in optimization_stim_names:
        curr_score_data = h5py.File(score_path+score_name+'_scores.hdf5', 'r')
        for sf in ordered_score_function_list:
            curr_stim_sf_pair = curr_score_data['norm_pin_scores_'+sf][:]
            best_stims_score_list.append(curr_stim_sf_pair)
    combined_score = sum([best_stims_score_list[i]*optimization_weightes[i] for i in range(len(optimization_weightes))])
    plt.figure(figsize=(cm_to_in(7.5), cm_to_in(4.5)))
    #plt.title(title)
    plt.xlabel('Parameter Set Rank')
    plt.ylabel('Weighted Score')
    #plt.ylim([0,1])
    #plt.xlim([0,10000])
    
    plt.locator_params(axis='x', nbins=2)
    plt.locator_params(axis='y', nbins=3)
    time_step = range(len(combined_score))
    plt.scatter(time_step, combined_score, s=1, color='black')
    if plot_save_path:
        plt.savefig(plot_save_path+'.pdf', format='pdf', dpi=1000, bbox_inches="tight")


pname_replace_map = {
        "gIhbar_Ih_basal": "gIh", 
        "gNaTs2_tbar_NaTs2_t_apical":"gNa_Trns1",
        "gSKv3_1bar_SKv3_1_apical": "gKv_3.1",
        "gIhbar_Ih_apical": "gIh",
        "gImbar_Im_apical": "gKm",
        "gNaTa_tbar_NaTa_t_axonal": "gNa_Trns2",
        "gK_Tstbar_K_Tst_axonal" : "gK_Trns",
        "gamma_CaDynamics_E2_axonal" : "gamma_dynamics",
        "gNap_Et2bar_Nap_Et2_axonal" : "gNa_Prst",
        "gSK_E2bar_SK_E2_axonal" : "gSK",
        "gCa_HVAbar_Ca_HVA_axonal" : "gCa_HVA",
        "gK_Pstbar_K_Pst_axonal" : "gK_Prst",
        "gSKv3_1bar_SKv3_1_axonal" : "gKv_3.1",
        "decay_CaDynamics_E2_axonal" : "decay_cadyns",
        "gCa_LVAstbar_Ca_LVAst_axonal" : "gCa_LVA",
        "gamma_CaDynamics_E2_somatic" : "gamma_dyns",
        "gSKv3_1bar_SKv3_1_somatic" : "gKv3.1",
        "gSK_E2bar_SK_E2_somatic" : "gSK",
        "gCa_HVAbar_Ca_HVA_somatic" : "gCa_HVA",
        "gNaTs2_tbar_NaTs2_t_somatic" : "gNa_Trns2",
        "gIhbar_Ih_somatic" : "gIh",
        "decay_CaDynamics_E2_somatic" : "decay_cadyns",
        "gCa_LVAstbar_Ca_LVAst_somatic" : "gCa_LVA",
        "g_pas" : "gPas"
}

def replace_param_names(old_names):
    new_names = []
    for curr_name in old_names:
        new_name = pname_replace_map[curr_name]
        new_names.append(new_name)
    return new_names

def read_params(params_description_path,params_ind):

    pname_replace_map = {
        "gIhbar_Ih_basal": "gIh", 
        "gNaTs2_tbar_NaTs2_t_apical":"gNa_Trns1",
        "gSKv3_1bar_SKv3_1_apical": "gKv_3.1",
        "gIhbar_Ih_apical": "gIh",
        "gImbar_Im_apical": "gKm",
        "gNaTa_tbar_NaTa_t_axonal": "gNa_Trns2",
        "gK_Tstbar_K_Tst_axonal" : "gK_Trns",
        "gamma_CaDynamics_E2_axonal" : "gamma_dynamics",
        "gNap_Et2bar_Nap_Et2_axonal" : "gNa_Prst",
        "gSK_E2bar_SK_E2_axonal" : "gSK",
        "gCa_HVAbar_Ca_HVA_axonal" : "gCa_HVA",
        "gK_Pstbar_K_Pst_axonal" : "gK_Prst",
        "gSKv3_1bar_SKv3_1_axonal" : "gKv_3.1",
        "decay_CaDynamics_E2_axonal" : "decay_cadyns",
        "gCa_LVAstbar_Ca_LVAst_axonal" : "gCa_LVA",
        "gamma_CaDynamics_E2_somatic" : "gamma_dyns",
        "gSKv3_1bar_SKv3_1_somatic" : "gKv3.1",
        "gSK_E2bar_SK_E2_somatic" : "gSK",
        "gCa_HVAbar_Ca_HVA_somatic" : "gCa_HVA",
        "gNaTs2_tbar_NaTs2_t_somatic" : "gNa_Trns2",
        "gIhbar_Ih_somatic" : "gIh",
        "decay_CaDynamics_E2_somatic" : "decay_cadyns",
        "gCa_LVAstbar_Ca_LVAst_somatic" : "gCa_LVA",
        "g_pas" : "gPas"
}
    if params_description_path.endswith('.csv'):
        df = pd.read_csv(params_description_path, skipinitialspace=True)
        params_names = df['Param name']
        base = df['Base value']
    else:
        print(f'{params_description_path} not ends with csv please figure this out')
        
def compute_parameter_deviations(GA_result_path, params_description_path, params_ind, plot_title, file_path_to_save):
    params_names,base = read_params(params_description_path,params_ind)
    df = pd.read_csv(params_description_path, skipinitialspace=True)
    params_names = df['Param name']
    params_names = replace_param_names(params_names)
    base_full = df['Base value']
    base_ga_result = [base_full[i] for i in params_ind]
    lbs_ga_result = [0.01*p for p in base_ga_result]
    ubs_ga_result = [100*p for p in base_ga_result]
    params_names_ga = [params_names[i] for i in params_ind]
    normalized_indvs, best_indvs = read_and_normalize_with_neg(GA_result_path, base_ga_result, lbs_ga_result, ubs_ga_result)
    final_indv_plot_by_section(params_names_ga, normalized_indvs[-1], plot_title, file_path_to_save, 1, 18)
    return base_full, best_indvs[-1]

def fill_constants(base, ga_result):
    base = list(base)
    for i in range(len(ga_result)):
        base[params_ind_full_bbp[i]] = ga_result[i]
    return base
def run_and_plot_voltage_response_single(base, stim_data, file_save_path, stim_title):
    volts_target,times = run_single_volts(base, stim_data)
    
    plot_stim_volts_pair(stim_data, [volts_target], times, stim_title,'', file_save_path)

def run_and_plot_voltage_response(base, ga_result, stim_data, file_save_path, stim_title, response_title):
    volts_target,times = run_single_volts(base, stim_data)
    volts_best_response,times = run_single_volts(ga_result, stim_data)
    plot_stim_volts_pair(stim_data, [volts_target, volts_best_response], times, stim_title, response_title, file_save_path)



params_ind_full_bbp = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 23] # bbp Full


# For overall params
num_total_params_bbp = 24
normalized_indvs_bbp_overall = [0 for i in range(num_total_params_bbp)]
best_indvs_bbp_overall = [0 for i in range(num_total_params_bbp)]
bbp_params_base = [0.00008,\
                    0.026145,\
                    0.004226,\
                    0.00008,\
                    0.000143,\
                    3.137968,\
                    0.089259,\
                    0.00291,\
                    0.006827,\
                    0.007104,\
                    0.00099,\
                    0.973538,\
                    1.021945,\
                    287.198731,\
                    0.008752,\
                    0.000609,\
                    0.303472,\
                    0.008407,\
                    0.000994,\
                    0.983955,\
                    0.00008,\
                    210.485284,\
                    0.000333,\
                    0.00003]
constant_ind = [0, 7, 13, 15, 21]
params_bbp = ['gIhbar_Ih_basal',
                'gNaTs2_tbar_NaTs2_t_apical',\
                'gSKv3_1bar_SKv3_1_apical',\
                'gIhbar_Ih_apical',\
                'gImbar_Im_apical',\
                'gNaTa_tbar_NaTa_t_axonal',\
                'gK_Tstbar_K_Tst_axonal',\
                'gamma_CaDynamics_E2_axonal',\
                'gNap_Et2bar_Nap_Et2_axonal',\
                'gSK_E2bar_SK_E2_axonal',\
                'gCa_HVAbar_Ca_HVA_axonal',\
                'gK_Pstbar_K_Pst_axonal',\
                'gSKv3_1bar_SKv3_1_axonal',\
                'decay_CaDynamics_E2_axonal',\
                'gCa_LVAstbar_Ca_LVAst_axonal',\
                'gamma_CaDynamics_E2_somatic',\
                'gSKv3_1bar_SKv3_1_somatic',\
                'gSK_E2bar_SK_E2_somatic',\
                'gCa_HVAbar_Ca_HVA_somatic',\
                'gNaTs2_tbar_NaTs2_t_somatic',\
                'gIhbar_Ih_somatic',\
                'decay_CaDynamics_E2_somatic',\
                'gCa_LVAstbar_Ca_LVAst_somatic',\
                'g_pas']

