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
from helpers.group_helpers import all_sudokus, exhaustive_search, cond_entropies, \
                                group_by_cond, all_lowerorder_sudokus, exhaustive_noisy, append_srv
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix

def loadrun(args,filename,f):
    if args.load:
        if args.runtype == 'exhaustive_noisy':
            filename = 'final_noisy_states'+str(args.states)
        with open(args.folder+filename+'.pkl', "rb") as fp:   # Unpickling
            sudokus = pickle.load(fp)
    else:
        sudokus = f(args)
        if args.save:
            with open(args.folder+filename+'.pkl', "wb") as fp:   #Pickling
                pickle.dump(sudokus, fp)
    return sudokus

def check_srvs(srvs,groups, tol=0):
    srvgroups = {}
    noisy_srvs = []
    subjects = np.arange(args.lenX)
    pX = JointProbabilityMatrix(args.lenX,args.states,joint_probs=args.dist_type)
    for k,s in groups.items():
        pXSi = append_srv(pX.copy(),srvs[s[0]],args, subjects, noisy)
        mi = pXSi.mutual_information(subjects,[args.lenX])
        indiv_mi = np.array([pXSi.mutual_information([i],[args.lenX]) for i in subjects])
        
        if mi > 0 and indiv_mi[0] <= tol and indiv_mi[1] <= tol:
            srvgroups[k] = s
            noisy_srvs = noisy_srvs + [srvs[ix] for ix in s]
    return srvgroups,np.array(noisy_srvs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the set of all modulo tables and the subset of unique, independent srvs')
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--lenX', default=2,type=int, help='Number of input variables X')
    parser.add_argument('--dist_type', default='iid', help='Distribution type')
    parser.add_argument('--runtype', default='sudokus', help='what to compute')
    parser.add_argument('--load', default=True,type=lambda x: bool(strtobool(x)), help='Load constructed srvs or not')
    parser.add_argument('--exhaustive', default=False,type=lambda x: bool(strtobool(x)), help='Load or run exhaustive sudokus')
    parser.add_argument('--sudokus', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute permutation sudokus')
    parser.add_argument('--noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute noisy sudokus')
    parser.add_argument('--exhaustive_noisy', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute exhaustive noisy sudokus')

    parser.add_argument('--conds', default=False,type=lambda x: bool(strtobool(x)), help='Load or compute conditional entropy matrix')
    parser.add_argument('--load_cond', default=False,type=lambda x: bool(strtobool(x)), help='Load cond entropy matrix')
    parser.add_argument('--save', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='../results/sudokus/', help='Folder in which data is saved')

    args = parser.parse_args()
    noisy = False
    if args.runtype =='exhaustive':
        filename ='exhaustive_sudokus_states'+str(args.states)  
        f = exhaustive_search
    elif args.runtype == 'sudokus':
        filename = 'permutation_sudokus_states'+str(args.states)
        f = all_sudokus
    elif args.runtype == 'noisy':
        filename = 'lowerorder_sudokus_states'+str(args.states)
        f = all_lowerorder_sudokus
        noisy = True
    elif args.runtype == 'exhaustive_noisy':
        filename = 'exhaustive_noisy_states'+str(args.states)
        f = exhaustive_noisy
        noisy = True
    srvs = loadrun(args,filename,f)
    print("LEN SRV LIST",len(srvs),srvs)

    if args.conds:
        filenamecond = 'cond_mat_states'+str(args.states)
        if noisy:
            filenamecond = 'noisy_'+filenamecond
        if args.load_cond:
            with open(args.folder+filenamecond+'.pkl', 'rb') as f:
                cond_mat = pickle.load(f)
        else:
            # calculate H(Sj|Si) and H(Si|Sj) for each SRV i
            cond_mat = cond_entropies(srvs, args, noisy)
            if args.save:
                with open(args.folder+filenamecond+'.pkl', "wb") as fp:   #Pickling
                    pickle.dump(cond_mat, fp)
            
        # group two srvs Si,Sj if H(Si|Sj)=H(Sj|Si)=0
        groups = group_by_cond(cond_mat)

        # check if noisy variables are srvs for iid input 
        if noisy:
            srvgroups, noisy_srvs = check_srvs(srvs, groups)
            print(srvgroups)
            with open(args.folder+'final_'+filename+'.pkl', "wb") as fp:   #Pickling
                pickle.dump(noisy_srvs, fp)
        

# profiler = cProfile.Profile()
# profiler.enable()
# for i in range(1):
#     mods = exhaustive_search(args)
# profiler.disable()
# stats = pstats.Stats(profiler).strip_dirs().sort_stats('tottime').print_stats()
# print(len(mods))