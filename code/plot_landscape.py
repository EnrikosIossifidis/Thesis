import argparse
import seaborn as sns
import sys
import glob
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from distutils.util import strtobool
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import params2matrix_incremental

from helpers.helpers import load_files
from helpers.landscape_helpers import cost_func_landscape,plot_2D,plot_plane

def swithcols(front,curfile):
    # brings list of 'front' columns to the front of file
    cols = curfile.columns.tolist()
    for f in front:
        cols.pop(cols.index(f))
    cols = front+cols
    return curfile[cols]

# return two orthogonal unit vectors
def random_orthogonal_unit_vectors(dim):
    v1 = np.random.rand(dim)
    v1 = v1/np.linalg.norm(v1)
    v2 = np.random.rand(dim)
    v2 -= v2.dot(v1) * v1
    v2 = v2/np.linalg.norm(v2)
    return v1, v2

def get_plane_points(v1,v2,mag=1,steps=10):
    mag = 1
    dim = len(v1)
    mid = [0.5 for _ in range(dim)]
    v3 = (v1*mag)
    v4 = (v2*mag)

    # get plane given random orthogonal unit vectors
    plane1 = np.linspace(mid - (0.5*v3),mid + (0.5*v3),steps)
    plane2 = np.linspace(mid - (0.5*v4),mid + (0.5*v4),steps)

    plane = []
    for p1 in plane1:
        for p2 in plane2:
            plane.append(p1+(p2-mid))
    return np.array(plane)

def get_plane_values(args,curdf,plane,parX):
    steps=args.steps
    subjects = list(range(args.lenX))
    syn_upper = float(curdf['syn_upper'])
    print(syn_upper)
    Z = []
    for i in range(steps):
        Z.append([])
        for j in range(steps):
            plane_id = (steps*i)+j
            curparams = parX+list(plane[plane_id])

            # get cost of curparams
            Z[-1].append(get_cost(args,parX,list(plane[plane_id]),subjects,syn_upper))
    Z=np.array(Z)
    return Z

def get_cost(args,parX,curparams,subjects,syn_upper):
    jXS = JointProbabilityMatrix(args.lenX+1,args.states)
    return cost_func_landscape(jXS,parX,curparams,subjects,syn_upper)

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot computed Isyns of a random/PCA plane of parameters')

    # model parameters
    parser.add_argument('--steps', default=10,type=int, help='Number of points on each axis on 2D PCA grid')
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
    parser.add_argument('--lenY', type=int,default=1,help='Number of output variables Y') 
    parser.add_argument('--states', type=int,default=2,help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='dirichlet', help='Distribution type')

    args = parser.parse_args()
    args.folder = '../results/preliminaries/'+args.folder+'/'

    # get input parameters 
    cur = glob.glob(args.folder+"*.pkl")
    
    df = pd.read_pickle(cur[args.pickle_idx])
    paramsXY = list(df['parXY']) # one iid, one dirichlet for example
    paramsX = [p[:(args.states**args.lenX)-1] for p in paramsXY]

    # compute orthogonal vectors (with PCA or random)
    if args.PCA:
        print("JA HALLO")
        # args.exp ='exp'+args.exp+'.pkl'
        # # read dataframe of choice
        # curdf = pd.read_pickle(args.folder+args.exp)
        # curdf = swithcols(['systemID','runID','I(Y;S)','all_paths','parXYSold'],curdf)
        # print(curdf)
        # for i in range(len(curdf)):
        #     curfile = curdf.iloc[[i]]
        #     if args.plot:
        #         print(list(curfile['all_paths'])[-1][-1])
        #         print(list(curfile['parXYSold'])[-1][-1])
        #         plot_2D(curfile,args.steps)
    else:
        # get number of dimensions
        lenparXS = ((args.states)**(args.lenX+1))-1
        lenparX = ((args.states)**(args.lenX))-1
        v1, v2 = random_orthogonal_unit_vectors(lenparXS-lenparX)    
        # get plane of values 
        params_plane = get_plane_points(v1,v2,steps=args.steps,mag=args.mag)

    # select input parameters parX of cost landscape
    for i in range(len(df)):
        par_id = i
        args.lenX = list(set(df['lenX']))[0]
        args.states= list(set(df['states']))[0]
        parX = paramsX[par_id]
        # get cost_values 
        cost_plane = get_plane_values(args,df.iloc[[par_id]],params_plane,parX)
        # print(params_plane)
        plot_plane(params_plane,cost_plane,plot_func=args.plot_func)
