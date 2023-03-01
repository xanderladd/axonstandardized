import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(
        description='assemble model')
parser.add_argument('--passive', action="store_true")
parser.add_argument('--pdf', action="store_true")
parser.add_argument('--force', action="store_true")
parser.add_argument('--model', type=int, required=True,
                    help='model number')
parser.add_argument('--timesteps', type=int, required=True,
                    help='model number')


## Definitions ##
stims_path = './stims/'
target_path = './target_volts/'
sweep_filter_1 = [str(e) for e in range(79, 100)]
sweep_filter_2 = ['101', '102', '103']
sweep_filter = sweep_filter_1 + sweep_filter_2

# This is the original cell
# stims_to_match = h5py.File('./filtered_dataset/cell_488683425/allen_data_stims_10000.hdf5', 'r')
# sweep_index_to_match = ['4',\
#                         '23',\
#                         '22',\
#                         '48_3',\
#                         '53_3',\
#                         '51_1',\
#                         '58',\
#                         '48_1',\
#                         '66',\
#                         '48_2',\
#                         '44',\
#                         '16',\
#                         '38',\
#                         '28',\
#                         '13',\
#                         '18',\
#                         '15',\
#                         '14']


#### HELPER FUNCTIONALITY ##########

def check_and_remove_fn(fn, force=False):
    if os.path.isfile(fn) and force:
        os.remove(fn)
        return 0
    elif os.path.isfile(fn):
        return 1
    else:
        return 0

def get_target_volts(target_path, model_number):
    target_file_str = target_path+'allen_data_target_volts_{}.hdf5'
    target_volts = h5py.File(target_file_str.format(model_number), 'r')
    return target_volts

def get_stims(stims_path, model_number):
    stims_file_str = stims_path+'allen_data_stims_{}.hdf5'
    stims = h5py.File(stims_file_str.format(model_number), 'r')
    return stims
    
def get_sweep_keys(stims):
    sweep_keys = [e.decode('ascii') for e in stims['sweep_keys']] 
    return sweep_keys
    
def plot_sweep_keys(sweep_keys,stims,sweep_filter):
    for key in sweep_keys:
        if not key in sweep_filter:
            print("Stim sweep number", key)
            print("Max val:", np.max(stims[key]))
            print("Min val:", np.min(stims[key]))
            plt.plot(stims[key])
            plt.show()
            
def match_sweeps(stim,stims_to_match, sweep_keys, sweeep_index_to_match, timesteps, allow_dup = False, verbose=True, show=True):
    """
    Args:
        stim: stim we are going to match
        stims_to_match: reference to match with
        sweep_keys: sweep keys of stim we have
        sweeep_index_to_match: ?
        show: plot to check?
    """
    matched_sweep_numbers = []
    sweep_map_to_original = []
    seen = []
    for stim_num in sweep_index_to_match:
        match_stim = stims_to_match[stim_num]
        if verbose:
            print("Original sweep number:", stim_num)
        if show:
            plt.plot(match_stim, color='red')
            plt.show()
        for key in sweep_keys:
            if key in seen and not allow_dup:
                continue
            if not key in sweep_filter_1+sweep_filter_2:
                curr_stim = stims[key]
                len_to_div = len(curr_stim) / timesteps
                curr_start = np.argmax(np.array(curr_stim) > 0) / len_to_div
                match_start = np.argmax(np.array(match_stim) > 0) 
                same_start = np.abs(curr_start - match_start) <  300
                diff_max = abs(np.max(match_stim)-np.max(curr_stim))
                diff_min = abs(np.min(match_stim)-np.min(curr_stim))
                #if diff_max < 0.0005 and diff_min < 0.0005:
                if diff_max < 0.0005 and diff_min < 0.0005:
                    if verbose:
                        print("Matched sweep number:", key)
                    matched_sweep_numbers.append(key)
                    sweep_map_to_original.append(stim_num)
                    seen.append(key)
                    if show:
                        plt.plot(curr_stim)
                        plt.show()
    return matched_sweep_numbers, sweep_map_to_original

def sample(input_vec, target_len, curr_dt=None):
    vec_len = len(input_vec)
    scale_factor = int(vec_len/target_len)
    dt =  (vec_len/target_len) * curr_dt
    sampled_vec = []
    scale_factor = max(scale_factor,1)
    for i in range(0, vec_len, scale_factor):
        window = input_vec[i:i+scale_factor]
        sampled_vec.append(np.max(window))
    return sampled_vec, dt

def extend(input_vec, target_len, curr_dt=None):
    vec_len = len(input_vec)
    extend_len = target_len - len(input_vec)
    assert extend_len > 0
    extend_vec = np.append(input_vec, np.repeat(input_vec[-1], extend_len))
  
    return extend_vec

def downsample(stims, target_volts, matched_sweep_numbers, resolution=10000, show=True):
    volt_res = {}
    stim_res = {}
    dts = {}
    for sweep_num in matched_sweep_numbers:
        curr_volt = target_volts[sweep_num][:]
        curr_stim = stims[sweep_num][:]
        dts[sweep_num] = resolution/len(curr_volt)
        curr_volt_sampled,_ = sample(curr_volt, resolution, dts[sweep_num])#target_volts['{}_dt'.format(sweep_num)][0])
        curr_stim_sampled,_ = sample(curr_stim, resolution, dts[sweep_num])#target_volts['{}_dt'.format(sweep_num)][0])
        assert False, "fix this function"
        if len(curr_stim_sampled) != 10000:
            # I don't think this is an issue, since nothing important is getting cutoff
            # see comment
            #print("Warning...stim for {} is not len 10k, .... truncating (TODO: Recalc DT)".format(sweep_num))
            #print("truncated by : ", len(curr_stim_sampled) - 10000)
            if len(curr_stim_sampled) - 10000 > 1000 and show:
                plt.figure()
                plt.plot(curr_stim_sampled)
                plt.plot(curr_stim_sampled[:10000])
            curr_stim_sampled = curr_stim_sampled[:10000]
            curr_volt_sampled = curr_volt_sampled[:10000]
            
        volt_res[sweep_num] = curr_volt_sampled
        stim_res[sweep_num] = curr_stim_sampled
        assert len(curr_stim_sampled) == 10000
        if show:
            plt.plot(curr_volt)
            plt.show()
            plt.plot(curr_volt_sampled, color='red')
            #plt.plot(curr_stim_sampled, color='green')

            plt.show()
    return volt_res, stim_res, dts

        
#### MAIN FUNCTIONALITIES ##########

def check_stims(model_number, timesteps):
    f = h5py.File("results/{}/stims_{}.hdf5".format(model_number,model_number), "r")
    all_keys = list(f.keys())
    sweep_keys = [key.decode('ascii') for key in f['sweep_keys'][:]]
    for key in sweep_keys:
        assert "{}_dt".format(key) in all_keys
        #print(int(f["{}_dt".format(key)]))
        assert f["{}_dt".format(key)][0] < 1, "Dt is weird"
        assert len(f[key][:]) == timesteps, "stim is wrong length:  {}".format(len(f[key][:]))
        
def check_volts(model_number, timesteps):
    f = h5py.File("results/{}/target_volts_{}.hdf5".format(model_number,model_number), "r")
    for key in f.keys():
        print(f[key][0], "Start volt")
        assert len(f[key][:]) == timesteps, "Voltage is wrong length: {}".format(len(f[key][:]))

        
def save_results_hdf5(model_number, stims, target_volts, dts, \
                      sweep_index_to_match, matched_sweep_numbers, \
                      sweep_map_to_original, timesteps):
    if not os.path.isdir("results/{}".format(model_number)):
        os.mkdir("results/{}".format(model_number))
    if os.path.isfile("results/{}/stims_{}.hdf5".format(model_number,model_number)):
        os.remove("results/{}/stims_{}.hdf5".format(model_number,model_number))
    if os.path.isfile("results/{}/target_volts_{}.hdf5".format(model_number,model_number)):
        os.remove("results/{}/target_volts_{}.hdf5".format(model_number,model_number))
    stim_file = h5py.File("results/{}/stims_{}.hdf5".format(model_number,model_number), "w")
    target_file = h5py.File("results/{}/target_volts_{}.hdf5".format(model_number,model_number), "w")
    correspondance = []
    seen = []
    skipped =0
    ct = 0
    for sweep_num,mapped_sweep in zip(matched_sweep_numbers, sweep_map_to_original):
        assert len(stims[sweep_num]) == timesteps, "not saving stim, it's wrong len"
        ct += 1
        if sweep_num in seen:
            skipped += 1
            continue
        else:
            seen.append(sweep_num)
        stim_file.create_dataset(sweep_num,data= stims[sweep_num])
        stim_file.create_dataset("{}_dt".format(sweep_num),data=[dts[sweep_num]])
        target_file.create_dataset(sweep_num, data=target_volts[str(sweep_num)])
        correspondance.append(mapped_sweep)
    correspondance = [n.encode("ascii", "ignore") for n in correspondance]
    matched_sweep_numbers = [n.encode("ascii", "ignore") for n in matched_sweep_numbers]
    matched_sweep_numbers_int = [int(num) for num in matched_sweep_numbers]
    stim_file.create_dataset("corresponding_original", data=correspondance)
    stim_file.create_dataset("sweep_keys", data=matched_sweep_numbers)
    stim_file.create_dataset("sweep_nums", data=matched_sweep_numbers_int)

    target_file.close()
    target_file.close()
    print("then SKIPPED", skipped, " and saved: ", ct)
    print("---------- Saved data for {}  ---------".format(model_number))
    print("--------------------------------------")
    
        
def match_sweep_and_save(model_numbers):
    for model_number in model_numbers:
        target_volts = get_target_volts(target_path, model_number )
        stims = get_stims(stims_path,model_number)
        sweep_keys = get_sweep_keys(stims)
        matched_sweep_numbers, sweep_map_to_original = match_sweeps(stims, stims_to_match, sweep_keys, \
                                             sweep_index_to_match, allow_dup=False, verbose=False,show=False)
        print("--------------------------------------")
        filler = sweep_map_to_original[-1]
        sweep_map_to_original.extend([filler,filler,filler,filler,filler])
        sweepNum2volt, sweepNum2Stim, dts = downsample(stims, target_volts, matched_sweep_numbers, show=False)

        save_results_hdf5(model_number, sweepNum2Stim, sweepNum2volt, dts, \
                          sweep_index_to_match, matched_sweep_numbers, sweep_map_to_original)
        
    

def select_stims(all_target_volts, all_stims, num_stims=0, passive=False):
    all_viable = []
    if num_stims == 0:
        num_stims = len(list(all_target_volts.keys()))
    for key in all_target_volts.keys():
        if "stim_types" in key: continue
        curr_stim = all_stims[key]
        curr_targV = all_target_volts[key]
        if "dt" not in key and 'sweep' not in key and '63' not in key:
            try:
                currently_passive = np.max(curr_targV) < 0 
            except:
                import pdb; pdb.set_trace()
            # or (not passive and currently_passive)
            if (passive and not currently_passive)  \
             or np.allclose(curr_stim,0) or key in all_viable:
                continue
            else:
                all_viable.append(key)
    all_viable = np.unique(all_viable)
    choices = np.random.choice(all_viable, min(len(all_viable),num_stims), replace=False)
    choices = np.unique(choices)
    return choices

def save_stims(model_number, passive, timesteps, show=False, pdf=None, force=False):
    all_target_volts = get_target_volts(target_path, model_number )
    all_stims = get_stims(stims_path,model_number)
    # TODO: store these files in var and delete them in case they exist
    if passive:
        stim_path = "results/{}/stims_{}_passive.hdf5".format(model_number,model_number)
        volt_path = "results/{}/target_volts_{}_passive.hdf5".format(model_number,model_number)
        obj_path = "results/{}/allen{}_objectives_passive.hdf5".format(model_number,model_number)
    else:
        stim_path = "results/{}/stims_{}.hdf5".format(model_number,model_number)
        volt_path = "results/{}/target_volts_{}.hdf5".format(model_number,model_number)
        obj_path = "results/{}/allen{}_objectives.hdf5".format(model_number,model_number)
        
    e1, e2, e3 = check_and_remove_fn(stim_path, force), check_and_remove_fn(volt_path, force), check_and_remove_fn(obj_path, force)
    
    
    if not (e1 == 0 and e2 == 0 and e3 == 0):
        print("Attempting to overwrite files without force set to true {}".format(stim_path))
        return

    stim_f = h5py.File(stim_path, "w")
    volt_f = h5py.File(volt_path, "w")
    new_obj_f = h5py.File(obj_path, "w")
    opt_stim_name_list = []
    stim_types = []
    # TODO: move filter to be right before choosing and only here
    # you should filter passive here as well
    stim_names = select_stims(all_target_volts, all_stims, num_stims=0, passive=passive)
    for key in stim_names:
        print("processing :", key)
        curr_stim = all_stims[key]
        stim_idx = np.where(stim_names == key)[0][0]
        stim_type = all_target_volts['stim_types'][stim_idx]
        curr_targV = all_target_volts[key]
        curr_dt = all_stims[key+'_dt'][0]
        prev_len = len(curr_stim)
        if curr_dt > 1:
            print(f'skipped : {key} because dt is {dt}')
            continue
        if len(curr_stim) > timesteps + 500 : # allow for the stim to be a little longer
            curr_stim,dt = sample(curr_stim, timesteps, curr_dt = curr_dt)
            curr_targV,_ = sample(curr_targV, timesteps, curr_dt = curr_dt) 
        elif timesteps > len(curr_stim):
            curr_stim = extend(curr_stim, timesteps, curr_dt = curr_dt)
            curr_targV = extend(curr_targV, timesteps, curr_dt = curr_dt) # add to targv 
            dt = curr_dt # dt stays the same
            
        else:
            dt = all_stims[key+'_dt'][:][0]
            print(f'dt : {dt}') 
        
        # cut off from front!
        cutoff = max(0,len(curr_targV) - timesteps )
        curr_targV = curr_targV[cutoff:]
        curr_stim = curr_stim[cutoff:]
        # dt doesn't give us exact cut off, this is regrettable
        print(f'stim: {key}, start dt: {curr_dt}, end dt: {dt}, prev len: {prev_len}, curr len: {len(curr_stim)}"')
        stim_types.append(stim_type)
        stim_f.create_dataset(key, data=curr_stim)
        stim_f.create_dataset(key+"_dt", data=[dt])
        volt_f.create_dataset(key, data=curr_targV)
        opt_stim_name_list.append(key)

        if show:
            fig = plt.figure()
            plt.title("added targ v " + key + " with DT " + str(dt))
            plt.plot(np.array(all_target_volts[key]))
            
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)
                
            fig = plt.figure()
            plt.plot(curr_targV)
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)
            
            fig = plt.figure()
            plt.plot(curr_stim)
            plt.title(key + " stim")
            if pdf:
                pdf.savefig(fig)
            plt.close(fig)

            
            

    new_weights = np.ones(3000)
    print(opt_stim_name_list, 'opt sim name')
    dt = h5py.special_dtype(vlen=str) 
    if passive:
        opt_stim_name_list = np.array(opt_stim_name_list, dtype=dt) 
        new_obj_f.create_dataset('opt_stim_name_list', data=opt_stim_name_list)
        new_obj_f.create_dataset('opt_weight_list', data=new_weights)
    #     new_obj_f.create_dataset('ordered_score_function_list', data=obj_f['ordered_score_function_list'])
        new_obj_f.create_dataset('ordered_score_function_list', data=[b'chi_square_normal'])

    new_obj_f.close()
    volt_f.close()
    stim_f.create_dataset('stim_types', data=stim_types)
    stim_f.close()
    
    print(len(opt_stim_name_list), "stims")
    
    
if __name__ == "__main__":
    pdf = None
    args = parser.parse_args()
    os.makedirs(os.path.join('results', str(args.model)), exist_ok=True)
    if args.pdf:
        if args.passive:
            pdf = PdfPages(os.path.join('results', str(args.model), 'passive_stims.pdf'))
        else:
            pdf = PdfPages(os.path.join('results', str(args.model), 'full_stims.pdf'))

    save_stims(args.model, passive=args.passive, timesteps=args.timesteps, force=args.force, show=True, pdf=pdf)
    
    if args.pdf:
        pdf.close()