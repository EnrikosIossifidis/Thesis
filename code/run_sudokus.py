import argparse
import cProfile
import pickle
import datetime
import time
import glob
import cProfile, pstats
import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from distutils.util import strtobool
from helpers.group_helpers import all_sudokus, exhaustive_search, cond_entropies, group_by_cond

def sudokus_combs(states,folder):
    allfiles = np.array([name for name in list(glob.iglob(folder+"*.pkl")) if 'permutation' in name])
    allsudokus = {}
    combs = []
    for a in allfiles:
        cur = int(a[-5])
        if cur < states:
            with open(a, 'rb') as f:
                allsudokus[cur] = pickle.load(f)
                combs = combs + list(itertools.combinations(np.arange(states), cur))
    return allsudokus, combs

def all_noisy_sudokus(args):
    sudokus, combs = sudokus_combs(args.states,args.folder)
    noisies = []
    M = args.states
    for i,c in enumerate(combs):
        # print(i)
        curlen = len(c)
        cursudos = sudokus[curlen]
        permutations = list(itertools.permutations(c))
        for c in cursudos:
            for p in permutations:
                unimat = np.full((M,M,M),1/M)
                for i in range(curlen):
                    for j in range(curlen):
                        cur = unimat[p[i]][p[j]]
                        cur[list(p)] = 0
                        cur[p[int(c[i][j])]] = curlen/M
                        unimat[p[i]][p[j]] = cur
                noisies.append(unimat)
    return np.array(noisies)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the set of all modulo tables and the subset of unique, independent srvs')
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--lenX', default=2,type=int, help='Number of input variables X')
    parser.add_argument('--dist_type', default='iid', help='Distribution type')
    parser.add_argument('--p', default=0.0,type=float, help='scale parameter for shape of Pr(S|X)')
    parser.add_argument('--exhaustive', default=False,type=lambda x: bool(strtobool(x)), help='Load or run exhaustive sudokus')
    parser.add_argument('--sudokus', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute permutation sudokus')
    parser.add_argument('--noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute noisy sudokus')
    parser.add_argument('--conds', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute conditional entropy matrix')

    parser.add_argument('--show_mods', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_mods', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_conds', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_sudokus', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load noisy  sudokus or not')
    parser.add_argument('--save', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--save_df', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='../results/sudokus/', help='Folder in which data is saved')

    args = parser.parse_args()
    sudokus = []
    ###########
    # Load or construct Exhausitve,Permutation or Noisy sudokus
    ###########
    if args.exhaustive:
        filename ='exhaustive_sudokus_states'+str(args.states)
        if args.load_mods:
            with open(args.folder+filename+'.pkl', "rb") as fp:   # Unpickling
                mods = pickle.load(fp)
        else:
            mods = exhaustive_search(args)
            if args.save:
                with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(mods, fp)
        sudokus = mods
    elif args.sudokus:
        filename = 'permutation_sudokus_states'+str(args.states)
        if args.load_sudokus:
            with open(args.folder+filename+'.pkl', 'rb') as f:
                sudokus = pickle.load(f)
        else:
            sudokus = np.array(all_sudokus(args.states))
            if args.save:
                with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(sudokus, fp)
    elif args.noisy:
        filename = 'noisy_sudokus_states'+str(args.states)
        if args.load_noisy:
            with open(args.folder+filename+'.pkl', 'rb') as f:
                noisy_sudokus = pickle.load(f)
        else:
            sudokus = all_noisy_sudokus(args)
            if args.save:
                with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(sudokus, fp)
        
    print("LEN SUDOKUS",len(sudokus),sudokus)

    if args.conds:
        # compute conditional entropies for clustering
        filename = 'cond_mat_states'+str(args.states)
        if args.load_conds:
            with open(args.folder+filename+'.pkl', 'rb') as f:
                cond_mat = pickle.load(f)
        else:
            # calculate H(Sj|Si) and H(Si|Sj) for each SRV i
            cond_mat = cond_entropies(sudokus, args)
            if args.save:
                with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(cond_mat, fp)
            
        # create and check if each mod is included once in each group
        groups = group_by_cond(cond_mat)
        print(groups)



# profiler = cProfile.Profile()
# profiler.enable()
# for i in range(1):
#     mods = exhaustive_search(args)
# profiler.disable()
# stats = pstats.Stats(profiler).strip_dirs().sort_stats('tottime').print_stats()
# print(len(mods))