import numpy as np
import h5py
import os
import bluepyopt as bpop
import pandas as pd
from mpi4py import MPI
import hoc_utils
import run_model
import config
os.chdir(config.neuron_path) 
from neuron import h
os.chdir("../../")
import math

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

global GEN
GEN = 0


class hoc_evaluator(bpop.evaluators.Evaluator):
    def __init__(self):
        """Constructor"""
        super(hoc_evaluator, self).__init__()
        # CONFIG: get params_csv from config
        names, self.orig_params, mins, maxs = hoc_utils.get_param_bounds(config.params_csv, config.params_opt_ind)
        
        if config.log_transform_params:
            self.bases, self.orig_params[config.params_opt_ind], mins, maxs = hoc_utils.log_params(maxs, mins, self.orig_params)
        self.params = [ bpop.parameters.Parameter(name, bounds=(minval, maxval)) for name, minval, maxval in zip(names, mins, maxs) ]
        print("Params to optimize:", [(name, minval, maxval) for name, minval, maxval in zip(names, mins, maxs)])
        print("Orig params:", self.orig_params)
        self.objectives = [bpop.objectives.Objective('Weighted score functions')]

        self.target_volts = self.generate_target_volts()
    
    def generate_target_volts(self):
        
        if config.target_volts:
            return config.target_volts
        
        if not os.path.isfile('target_volts.npy'):
            target_volts = run_model.run_model(self.orig_params, config.opt_stim_names)
            np.save('target_volts.npy', target_volts)
        else:
            target_volts = np.load('target_volts.npy')
        return target_volts
        
    def my_evaluate_invalid_fitness(toolbox, population):
        '''Evaluate the individuals with an invalid fitness
        Returns the count of individuals with invalid fitness
        '''
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # invalid_ind = [population[0]] + invalid_ind 
        fitnesses = toolbox.evaluate(invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        return len(invalid_ind)
    

    def assign_params(self, curr_params):
        modified_params = self.orig_params
        for i, curr_opt_ind in enumerate(config.params_opt_ind):
            modified_params[curr_opt_ind] = curr_params[i]
        # undo log x-form
        if config.log_transform_params:
            for i, opt_ind in enumerate(config.params_opt_ind):
                if self.bases[i] > config.base_thresh and self.orig_params[i]:
                    modified_params[opt_ind] = math.pow(self.bases[i], modified_params[opt_ind])
        # turn param neg
        # CONFIG Negative param inds
        if len(config.negative_param_inds) > 0:
            for negative_ind in config.negative_param_inds:
                modified_params[negative_ind] = - np.abs(modified_params[negative_ind])
        return modified_params
    
    def update_score_log(self, score, GEN):
        seed = os.getenv('BLUEPYOPT_SEED')
        argmin = np.argmin(score)
        with open(f'score{seed}.log','a+') as f:
            f.write(str(GEN) + " lowest score: " + str(np.nanmin(score)) + ' \n')
            # f.write(str(GEN) + " : len score " + str(len(score)) + ' \n')
    
    def init_simulator_and_evaluate_with_lists(self, param_values):
        global GEN

        param_values = np.array(param_values)
        nan_inds = np.array(np.where( np.isnan(param_values))[0])
        assert not len(nan_inds), f"{param_values[nan_inds][0]} has nan"
        comm.Barrier() # avoid early bcast
        param_values = comm.bcast(param_values, root=0)
        scores, ranks, curr_rank = [], [], global_rank
        
        while curr_rank < len(param_values):
            start_Vm = self.target_volts[0][0]
            curr_params = self.assign_params(param_values[curr_rank])
            simulated_volts = run_model.run_model(curr_params, config.opt_stim_names, config.dt, start_Vm=start_Vm)
            
            curr_score = hoc_utils.evaluate_score_function(config.opt_stim_names, self.target_volts, simulated_volts, config.weights)
            curr_score = curr_score[0]
            
            scores.append(curr_score)
            ranks.append(curr_rank)
            curr_rank += size 
            
        comm.Barrier() # avoid early GATHER
        scores, ranks = comm.gather(scores, root=0), comm.gather(ranks, root=0)
        scores = hoc_utils.unest_mpi_arr(scores)
        ranks = hoc_utils.unest_mpi_arr(ranks)
        if global_rank == 0: 
            assert all(i < j for i, j in zip(ranks, ranks[1:])), 'ranks are unordered'
            
        scores = comm.bcast(scores, root=0)
        scores = np.array(scores).reshape(-1,1)

        # TODO: should not need these checks ever
        # scores = np.where(~np.isfinite(scores), 10000000, scores)
        # scores = np.clip(scores,-10000, 1200000)
        
        if global_rank  == 0:
            self.update_score_log(scores, GEN)
        
        GEN += 1
        return scores 


import bluepyopt.deapext.algorithms as algo
algo._evaluate_invalid_fitness = hoc_evaluator.my_evaluate_invalid_fitness
