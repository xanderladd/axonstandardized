import bluepyopt as bpop
from bluepyopt.deapext.stoppingCriteria import MaxNGen
import deap.algorithms
import deap.tools
from deap import algorithms, base, creator, tools
import argparse
import pickle
import time
import numpy as np
from datetime import datetime
import os
import sys
import textwrap
import logging
from mpi4py import MPI
import random
import shutil


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='L5PC example',
        epilog=textwrap.dedent('''\
The folling environment variables are considered:
    L5PCBENCHMARK_USEIPYP: if set, will use ipyparallel
    IPYTHON_PROFILE: if set, used as the path to the ipython profile
    BLUEPYOPT_SEED: The seed used for initial randomization
        '''))
    parser.add_argument('--start', action="store_true")
    parser.add_argument('--continu', action="store_true", default=False)
    parser.add_argument('--checkpoint', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--starting_pop', required=False, default=None,
                        help='Checkpoint pickle to avoid recalculation')
    parser.add_argument('--offspring_size', type=int, required=False, default=2,
                        help='number of individuals in offspring')
    parser.add_argument('--max_ngen', type=int, required=False, default=2,
                        help='maximum number of generations')
    parser.add_argument('--responses', required=False, default=None,
                        help='Response pickle file to avoid recalculation')
    parser.add_argument('--analyse', action="store_true")
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--hocanalyse', action="store_true")
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to use for optimization')
    parser.add_argument('--ipyparallel', action="store_true", default=False,
                        help='Use ipyparallel')
    parser.add_argument(
        '--diversity',
        help='plot the diversity of parameters from checkpoint pickle file')
    parser.add_argument('-v', '--verbose', action='count', dest='verbose',
                        default=0, help='-v for INFO, -vv for DEBUG')

    return parser


def save_logs(fn, best_indvs, population):
    output = open("./best_indv_logs/best_indvs_gen_"+str(gen_counter)+fn, 'wb')
    pickle.dump(best_indvs, output)
    output.close()
    
def my_update(halloffame, history, population):
    global gen_counter, cp_freq
    old_update(halloffame, history, population)
    best_indvs.append(halloffame[0])
    gen_counter = gen_counter+1

    if gen_counter%cp_freq == 0 and global_rank == 0:
        fn = '.pkl'
        save_logs(fn, best_indvs, population)


def my_ea(
        population,
        toolbox,
        mu,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        cp_frequency=1,
        cp_filename=None,
        continue_cp=False,
        terminator=None,
        param_names=[]):
    r"""This is the :math:`(~\alpha,\mu~,~\lambda)` evolutionary algorithm
    Args:
        population(list of deap Individuals)
        toolbox(deap Toolbox)
        mu(int): Total parent population size of EA
        cxpb(float): Crossover probability
        mutpb(float): Mutation probability
        ngen(int): Total number of generation to run
        stats(deap.tools.Statistics): generation of statistics
        halloffame(deap.tools.HallOfFame): hall of fame
        cp_frequency(int): generations between checkpoints
        cp_filename(string): path to checkpoint filename
        continue_cp(bool): whether to continue
    """
    global starting_pop_hack

    if cp_filename:
        cp_filename_tmp = cp_filename + '.tmp'
    
    if continue_cp:
        print("continue cp from ", global_rank)
        # A file name has been given, then load the data from the file
        if global_rank == 0:
            cp = pickle.load(open(cp_filename, "rb"))
            population = cp["population"]
            parents = cp["parents"]
            start_gen = cp["generation"]
            halloffame = cp["halloffame"]
            logbook = cp["logbook"]
            history = cp["history"]
            random.setstate(cp["rndstate"])
        else:
            # give normal initialization 
            parents = population[:]
            popualation = None
            history = deap.tools.History()
            logbook = deap.tools.Logbook()
            start_gen = 1
            logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
            # algo._update_history_and_hof(halloffame, history, population)
            # algo._record_stats(stats, logbook, start_gen, population, invalid_count)
            
        # toolbox = comm.bcast(toolbox, root=0)
        parents = comm.bcast(parents, root=0)
        population = comm.bcast(population, root=0)
        # Assert that the fitness of the individuals match the evaluator
        obj_size = len(population[0].fitness.wvalues)
        population = algo._define_fitness(population, obj_size)
        parents = algo._define_fitness(parents, obj_size)
        algo._evaluate_invalid_fitness(toolbox, parents)
        algo._evaluate_invalid_fitness(toolbox, population)
        start_gen = 1

        # cp = comm.bcast(cp, root=0)
        
    else:
        # Start a new evolution
        start_gen = 1
        if starting_pop_hack and global_rank == 0:
            with open(starting_pop_hack,'rb') as f: data = pickle.load(f)
            population = data['population']
        
        population = comm.bcast(population, root=0)
        parents = population[:]
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        history = deap.tools.History()
        invalid_count = algo._evaluate_invalid_fitness(toolbox, population)
        # algo._update_history_and_hof(halloffame, history, population)
        bpop.deapext.utils.update_history_and_hof(halloffame, history, population)
        # algo._record_stats(stats, logbook, start_gen, population, invalid_count)
        bpop.deapext.utils.record_stats(stats, logbook, start_gen, population, invalid_count)

    stopping_criteria = [MaxNGen(ngen)]

    # Begin the generational process
    gen = start_gen + 1
    stopping_params = {"gen": gen}
    
    while not(algo._check_stopping_criteria(stopping_criteria, stopping_params)):
        offspring = algo._get_offspring(parents, toolbox, cxpb, mutpb)
        
        population = parents + offspring

        invalid_count = algo._evaluate_invalid_fitness(toolbox, offspring)
        # algo._update_history_and_hof(halloffame, history, population)
        bpop.deapext.utils.update_history_and_hof(halloffame, history, population)
        # algo._record_stats(stats, logbook, gen, population, invalid_count)
        bpop.deapext.utils.record_stats(stats, logbook, gen, population, invalid_count)

        # Select the next generation parents
        # if global_rank == 0:
        #     import pdb; pdb.set_trace()
        if global_rank == 0:
            parents = toolbox.select(population, mu)
            
            """ more elaborate logging """    
            os.makedirs('extra_logs', exist_ok=True)
            # population is actually parents now
            np.save(f'extra_logs/pop_{gen}.npy', np.array(parents))
            all_scores = [parent.fitness.values[0] for parent in parents]
            np.save(f'extra_logs/{gen}_score.npy', np.array(all_scores))
            np.save(f'extra_logs/hof_{gen}.npy', np.array(halloffame))

            
        parents = comm.bcast(parents, root=0)
                        

        # logger.info(logbook.stream)
        if(cp_filename and cp_frequency and
           gen % cp_frequency == 0) and global_rank == 0:

            with open('score.log','a+') as f:
                f.write(logbook[-1].__repr__())
            
            cp = dict(population=population,
                      generation=gen,
                      parents=parents,
                      halloffame=halloffame,
                      history=history,
                      logbook=logbook,
                      rndstate=random.getstate())
            if global_rank == 0:
                pickle.dump(cp, open(cp_filename_tmp, "wb"))
                if os.path.isfile(cp_filename_tmp):
                    shutil.copy(cp_filename_tmp, cp_filename)
                    logger.debug('Wrote checkpoint to %s', cp_filename)

        gen += 1
        stopping_params["gen"] = gen

    return population, halloffame, logbook, history

