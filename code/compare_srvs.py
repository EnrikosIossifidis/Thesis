from funcy.seqs import first
from JointProbabilityMatrix import JointProbabilityMatrix
import pickle
import argparse
import seaborn as sns

import time
import numpy as np
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dit import Distribution
from distutils.util import strtobool
from plot_helpers import getpXSfound, getcond
from group_helpers import all_srvs, get_all_cycles
from load_helpers import get_data, get_best
from helpers import cond_entropies, group_by_cond

def check_all_len(row,goal):
    for f in row['final_params']:
        if len(f) != goal:
            row['check_len'] = False
            return row
    row['check_len'] = True
    return row

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate the set of all modulo tables and the subset of unique, independent srvs')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='iid', help='Distribution type')
    parser.add_argument('--p', default=0.0,type=float, help='scale parameter for shape of Pr(S|X)')

    parser.add_argument('--load', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--run', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--plot', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--save', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='../experiments/data_acc_speed/srvs_analysis/', help='Folder in which data is saved')

    args = parser.parse_args()

    # load found srvs
    if args.load:
        d = get_data(args)

        # make sure all found SRVs have correct number of parameters
        len_srv_params = (args.states**(args.lenX+1))-(args.states**args.lenX)
        d = d.apply(check_all_len,args=(len_srv_params,),axis=1) 
        d = d.loc[d['check_len']]
        
        d = get_best(d) # best srvs of multiple runs for a joint pdf Pr(X,Y)
        # print(d)

        # load exhaustive searched srvs
        all_files = list(glob.glob(args.folder+"*.pkl"))
        filename = 'srv_per_modulo_group_X2_Y1_states'+str(args.states)
        exhaustive_srvs = [s for s in all_files if filename in s][0]
        with open(exhaustive_srvs, "rb") as fp:   # Unpickling
            exhaustives = pickle.load(fp)
        print("exhaustives", datetime.fromtimestamp(time.time()))

    # get cycle SRVS
    cycle_srvs = all_srvs(get_all_cycles(args.states),args.states)
    print(len(cycle_srvs))

    if args.run:        
        len_parX = (args.states**args.lenX)-1
        pdummy = JointProbabilityMatrix(args.lenX+1,args.states)
        d = d.apply(getpXSfound,args=(len_parX,args,),axis=1)
        d = d.apply(getcond,args=(len_parX,exhaustives['mod_srv'],args,),axis=1)
        pd.to_pickle(d,args.folder+'cond_data_states'+str(args.states)+'.pkl')

    if args.plot:
        def getmincond(row,len_exhaustives):
            mini = min([row['conds'+str(l)] for l in range(len_exhaustives)])
            row['min_conds'] = mini
            return row

        def seperate_mis(row):
            mis = row['indiv_mi']
            for i,mi in enumerate(mis):
                row['mi_'+str(i)] = mi
            return row

        d = pd.read_pickle(args.folder+'cond_data_states'+str(args.states)+'.pkl')
        plot_keys1 = ['conds'+str(i) for i in range(len(exhaustives['mod_srv']))]
        plot_keys2 = ['indiv_mi', 'I(X;Sfound)/H(Sfound)','rel_mis','efficiency','final_params']
        plot_keys = plot_keys1 + plot_keys2
        d_plot = d[plot_keys]
        d_plot = d_plot.apply(pd.Series.explode)

        # # select sudoku i with min_i H(Sfound|HSudoku_i) and min_j I(Xj;Sfound)/H(Xj) for y-axis
        d_plot = d_plot.apply(getmincond, args=(len(exhaustives['mod_srv']),),axis=1)
        d_plot = d_plot.apply(seperate_mis, axis=1)

        # d_plot['test_upper'] = False
        # d_plot.loc[d_plot['efficiency'] <=1, "test_upper"] = True
        ax = sns.scatterplot(data=d_plot, x='min_conds', y='rel_mis', size='I(X;Sfound)/H(Sfound)')
        ax.set(xlabel='min_i H(Sfound/Sgroup_i)', ylabel=r'$\Sigma_i$'+'I(Xi;Sfound)/I(X;Sfound)')
        plt.show()
