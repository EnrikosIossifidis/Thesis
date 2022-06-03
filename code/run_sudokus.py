import argparse
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from distutils.util import strtobool
from helpers.group_helpers import all_sudokus, exhaustive_search, cond_entropies, all_oversized,\
                                group_by_cond, all_lowerorder_sudokus, all_noisies, append_srv

def loadrun(args,filename,f):
    # load selected type of SRVs
    if args.runtype == 'noisy':
        filename = 'final_noisy_states'+str(args.states)
    if args.load:
        if args.runtype =='PSRVs':
            with open(args.folder+filename+'.npy', "rb") as fp:  
                sudokus = np.load(fp,allow_pickle=True)
        else:
            with open(args.folder+filename+'.pkl', "rb") as fp:  
                sudokus = pickle.load(fp)
    # construct selected type of SRVs
    else:
        sudokus = f(args)
        if args.save:
            with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                pickle.dump(sudokus, fp)
    return sudokus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the set of all modulo tables and the subset of unique, independent srvs')
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--lenX', default=2,type=int, help='Number of input variables X')
    parser.add_argument('--dist_type', default='iid', help='Distribution type')
    parser.add_argument('--runtype', default='sudokus', help='what to compute')
    parser.add_argument('--load', default=True,type=lambda x: bool(strtobool(x)), help='Load constructed srvs or not')
    parser.add_argument('--save', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--exhaustive', default=False,type=lambda x: bool(strtobool(x)), help='Load or run exhaustive sudokus')
    parser.add_argument('--sudokus', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute permutation sudokus')
    parser.add_argument('--noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute noisy sudokus')
    parser.add_argument('--exhaustive_noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute exhaustive noisy sudokus')

    parser.add_argument('--condmat', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute conditional entropy matrix of exhaustive PSRVs')
    parser.add_argument('--load_cond', default=False,type=lambda x: bool(strtobool(x)), help='Load cond entropy matrix')
    parser.add_argument('--folder', default='../results/sudokus/', help='Folder in which data is saved')

    args = parser.parse_args()

    # load or construct the selected type of SRVs for a given number of states
    if args.runtype =='exhaustive':
        filename ='exhaustive_sudokus_states'+str(args.states)  
        f = exhaustive_search
    elif args.runtype == 'PSRVs':
        filename = 'permutation_sudokus_states'+str(args.states)
        f = all_sudokus
    elif args.runtype == 'lowerorders':
        filename = 'lowerorder_sudokus_states'+str(args.states)
        f = all_lowerorder_sudokus
        noisy = True
    elif args.runtype == 'noisy':
        filename = 'noisy_states'+str(args.states)
        f = all_noisies
        noisy = True
    elif args.runtype == 'oversized':
        filename = 'oversized_states'+str(args.states)
        f = all_oversized
    
    srvs = loadrun(args,filename,f)

    print("LEN SRV LIST",srvs,len(srvs))

    # compute conditional entropy matrix to cluster exhaustive PSRVs
    if args.runtype == 'exhaustive' and args.condmat==True:
        filenamecond = 'cond_mat_states'+str(args.states)
        if noisy:
            filenamecond = 'noisy_'+filenamecond
        if args.load_cond:
            with open(args.folder+filenamecond+'.pkl', 'rb') as f:
                cond_mat = pickle.load(f)
        else:
            # calculate H(Sj|Si) and H(Si|Sj) for each SRV i
            cond_mat = cond_entropies(srvs, args, cond=True)
            if args.save:
                with open(args.folder+filenamecond+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(cond_mat, fp)
            
        # group two srvs Si,Sj if H(Si|Sj)=H(Sj|Si)=0
        groups = group_by_cond(cond_mat)
        print("Clusters:",groups)