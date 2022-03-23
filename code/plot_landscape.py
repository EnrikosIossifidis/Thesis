import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import pandas as pd
from distutils.util import strtobool
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import matrix2params_incremental
from jointpdfpython3.toy_functions import append_synergistic_variables,append_variables_with_target_mi
from jointpdfpython3.measures import synergistic_information, synergistic_entropy_upper_bound
from helpers.planes import plot_2D,random_orthogonal_unit_vectors,get_plane_points,get_plane_values,plot_plane

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot computed Isyns of a random/PCA plane of parameters')

    # model parameters
    parser.add_argument('--steps', default=20,type=int, help='Number of points on each axis on 2D PCA grid')
    parser.add_argument('--mag', default=1,type=float, help='Magnitude of random orthogonal vectors')
    parser.add_argument('--plot_func', default='contour', help='Type of plot of 2D plane')
    parser.add_argument('--PCA', default=False,type=lambda x: bool(strtobool(x)), help='If no, then random points')
    parser.add_argument('--sysID', default=0,type=int, help='System ID of minimization path to plot')
    parser.add_argument('--runID', default=0,type=int, help='Run ID(s) of minimization path(s)')
    parser.add_argument('--save', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='landscape', help='Folder in which data is saved')
    parser.add_argument('--pickle_idx',type=int,default=-1,help='Index of pickle in data folder')

    # system parameters
    parser.add_argument('--lenX', type=int, default=2,help='Number of input variables X')
    parser.add_argument('--states', type=int,default=2,help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='dirichlet', help='Distribution type (iid, uniform, dirichlet)')
    parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
    parser.add_argument('--Sprev', default=False,type=lambda x: bool(strtobool(x)), help='Add SRV to X before constructing cost landscape')
    parser.add_argument('--second', default=True,type=lambda x: bool(strtobool(x)), help='Compare 2D with other input distribution')

    args = parser.parse_args()
    args.folder = '../results/preliminaries/'+args.folder+'/'
    
    if args.prev:
        # TO DO: load df as JPB (params2matrix?)
        df = pd.read_pickle(args.prev)
    else:
        # Initialize system 1 
        jpb = JointProbabilityMatrix(args.lenX,args.states,joint_probs=args.dist_type)
        if args.Sprev:
            append_synergistic_variables(jpb,1)
                
        # Initialize system 2 (I(X1;X2)=...)
        if args.second:
            systems = [JointProbabilityMatrix(args.lenX,args.states,joint_probs=args.dist_type)]
            mis = [0.3,0.9]
            for m in mis:
                jpb2 = JointProbabilityMatrix(1,args.states,joint_probs=args.dist_type)
                append_variables_with_target_mi(jpb2,1,m*jpb2.entropy([0]))
                systems.append(jpb2)
            mis = [0] + mis

        # compute each system's properties
        syn_uppers = []
        parXs = []
        lenparXS = ((args.states)**(args.lenX+1))-1
        lenparX = ((args.states)**(args.lenX))-1
        variablesX = np.arange(args.lenX)
        for s in systems:
            syn_uppers.append(synergistic_entropy_upper_bound(s,variables=variablesX))
            parXs.append(matrix2params_incremental(s)[:lenparX])

    if args.PCA:
        if args.prev:
            data = df
        else:
            data = synergistic_information(jpb,variablesX,variablesX)

        # compute (and plot) PCA 2D plane
        plot_2D(data,args.steps)

    else:
        os.chdir("../results/preliminaries/landscape")

        num_plots = 3
        planes = []
        for n in range(num_plots):
            # get number of dimensions and generate two random orthogonal vectors
            v1, v2 = random_orthogonal_unit_vectors(lenparXS-lenparX)    

            # get plane of values 
            params_plane = get_plane_points(v1,v2,steps=args.steps,mag=args.mag)

            # get cost_values in plane for each system and plot them
            for i,s in enumerate(systems):
                print("Hmax(X)",syn_uppers[i])
                print("I(X1;X2)",s.mutual_information([0],[1]))
                cost_plane = get_plane_values(args,params_plane,syn_uppers[i],parXs[i])
                title = 'dist_type_'+args.dist_type+' I(X1;X2)_'+str(mis[i])+' states_'+str(args.states)+'num_'+str(n)
                plot_plane(title,params_plane,cost_plane,plot_func=args.plot_func)
