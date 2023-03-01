import bluepyopt as bpop
# when things are setup, make this a conditional import
#import hoc_evaluator_allen as hoc_ev
import config
if model == 'allen':
    import hoc_evaluator_allen as hoc_ev
else:
    import hoc_evaluator as hoc_ev
import bluepyopt.deapext.algorithms as algo
import pickle
import time
import numpy as np
from datetime import datetime
import os
import sys
import logging
from mpi4py import MPI
import random
import shutil
import optimize_helpers as helpers

comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.disabled = True
# logging.getLogger('requests').setLevel(logging.DEBUG)
gen_counter = 0
best_indvs = []
cp_freq = 1
old_update = bpop.deapext.utils.update_history_and_hof 


def create_optimizer(args):
    '''returns configured bluepyopt.optimisations.DEAPOptimisation'''
    if args.ipyparallel:
        from ipyparallel import Client
        rc = Client(profile=os.getenv('IPYTHON_PROFILE'))
        logger.debug('Using ipyparallel with %d engines', len(rc))

        lview = rc.load_balanced_view()
        dview = rc.direct_view()
        def mapper(func, it):
            dview.map(os.chdir, [os.getcwd()]*len(rc.ids))
            start_time = datetime.now()
            ret = dview.map_sync(func, it)
            print('Generation took', datetime.now() - start_time)
            return ret

        map_function = mapper
    else:
        map_function = None

    evaluator = hoc_ev.hoc_evaluator()
    seed = os.getenv('BLUEPYOPT_SEED', args.seed)
    opt = bpop.optimisations.DEAPOptimisation(
        evaluator=evaluator,
        map_function=map_function,
        seed=seed,
        eta=10, # was 20
        mutpb=1,
        cxpb=1)

    return opt



def main():
    global starting_pop_hack
    starting_pop_hack = None
    args = helpers.get_parser().parse_args()
    # algo._update_history_and_hof = my_update
    bpop.deapext.utils.update_history_and_hof = helpers.my_update
    algo.eaAlphaMuPlusLambdaCheckpoint = helpers.my_ea


    opt = create_optimizer(args)
    # if global_rank != 0:
    #     args.checkpoint= None
    #     args.continu = False
    if args.starting_pop:
        starting_pop_hack =  args.starting_pop
        
    pop, hof, log, hst = opt.run(max_ngen=args.max_ngen,
        offspring_size=args.offspring_size,
        continue_cp=bool(args.continu),
        cp_filename=args.checkpoint,
        cp_frequency=5)
    fn = time.strftime("_%d_%b_%Y")
    fn = fn + ".pkl"
    if global_rank == 0:
        output = open("best_indvs_final"+fn, 'wb')
        pickle.dump(best_indvs, output)
        output.close()
        output = open("log"+fn, 'wb')
        pickle.dump(log, output)
        output.close()
        output = open("hst"+fn, 'wb')
        pickle.dump(hst, output)
        output.close()
        output = open("hof"+fn, 'wb')
        pickle.dump(hof, output)
        output.close()

    print ('Hall of fame: ', hof, '\n')
    print ('log: ', log, '\n')
    print ('History: ', hst, '\n')
    print ('Best individuals: ', best_indvs, '\n')
if __name__ == '__main__':
    main()
