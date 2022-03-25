import os
import glob
import pickle
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from datetime import datetime
from distutils.util import strtobool
from helpers.compare_helpers import load_frame,load_sudokus,mincondentropy

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot computed Isyns (for different model parameters) and compare found srvs to sudoku srvs')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='random', help='Distribution type')
    parser.add_argument('--nosyndisc', default=False,type=lambda x: bool(strtobool(x)), help='jointpdf only')
    parser.add_argument('--filefolder', default='rundata', help='From where data is retrieved and where to save fig')
    parser.add_argument('--savefolder', default='comparison', help='From where data is retrieved and where to save fig')
    parser.add_argument('--compare_srvs', default=True,type=lambda x: bool(strtobool(x)), help='Compare found SRVs with sudoku srvs')
    parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Save plot as json or not')
    args = parser.parse_args()

    args.filefolder = '../results/'+args.filefolder+"/"
    args.savefolder = '../results/'+args.savefolder+"/"

    # load found srvs
    d = load_frame(args.states,args.dist_type,args.filefolder)
    sudokus = load_sudokus(d['statesS'].unique())
    print(len(d))
    if args.compare_srvs:
        d = d.apply(lambda row: mincondentropy(row,sudokus[row['statesS']],args.nosyndisc),axis=1) 
        d.to_pickle(args.savefolder+'comparison_'+args.dist_type+'states'+ str(args.states)+'.pkl')
        if args.plot: 
            fig, ax = plt.subplots(figsize=(10,5))        
            xcol = 'WMS(X)/Hmax(X)'
            ycol = 'H(Sfound|min_perm)'
            sns.scatterplot(data=d, x=xcol, y=ycol, 
                            hue='exp_sort',sizes=(10,60),palette='tab10',s=100,ax=ax)

            title = args.dist_type+" input dist w/ "+str(args.states)+'states (datasize='+str(len(d))+')'
            plt.title(title,fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(xcol,fontsize=13)
            plt.ylabel(" H(Sfound|XOR)",fontsize=13)
            plt.show()
            fig.savefig(args.savefolder+'comparison_'+args.dist_type+'states'+ str(args.states))
        #     ax.set(xlabel='min_i H(Sfound/Sgroup_i)', ylabel=r'$\Sigma_i$'+'I(Xi;Sfound)/I(X;Sfound)')
