import argparse
import numpy as np
import dit
import seaborn as sns
import seaborn as sns; sns.set()
import pandas as pd
import matplotlib.pyplot as plt
from distutils.util import strtobool
from helpers.compare_helpers import load_frame,normcondentropy,bestoffive

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot computed Isyns (for different model parameters) and compare found srvs to sudoku srvs')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--paramstype', default='random', help='Distribution type')
    parser.add_argument('--nosyndisc', default=False,type=lambda x: bool(strtobool(x)), help='jointpdf only')
    parser.add_argument('--filefolder', default='npdirichlet', help='From where data is retrieved and where to save fig')
    parser.add_argument('--savefolder', default='test', help='From where data is retrieved and where to save fig')
    parser.add_argument('--run', default=False,type=lambda x: bool(strtobool(x)), help='Compare found SRVs with sudoku srvs')
    parser.add_argument('--top', default=5,type=int, help='Number of lowest conditional entropies that are taken to compare (max 7)')
    parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Save plot as json or not')
    args = parser.parse_args()

    args.filefolder = '../results/'+args.filefolder+"/"
    args.savefolder = '../results/'+args.savefolder+"/"

    # Load all constructed srvs given number of states
    with open('../results/sudokus/allconstructedSRVstates'+str(args.states)+'.pkl', 'rb') as f:
        symsd = np.load(f,allow_pickle=True)
        syms = []
        for k in symsd.keys():
            syms = syms+list(symsd[k])
        print(syms[0])
        print("TOT NUMBER OF SYMS:",len(syms))

    # load systems and optimized srvs
    with open(args.filefolder+'nprandomdirichlet2345.pkl', 'rb') as f:
        d = pd.read_pickle(f)
        d = d[d['states']==args.states]
        if len(d)>0:
            ds = load_frame(d=d)
        else:
            ds = None
            
    if args.run:
        # compute all H(Sfound|Sym) 
        ds = ds.apply(lambda row: normcondentropy(row,syms),axis=1) 

        # compute conditional entropy of optimized SRV given top 5 most similar constructed SRVs
        ds = ds.apply(lambda row: bestoffive(row,args,syms),axis=1)
        print(ds[['systemID','syn_upper','syn_info','H(Sfound|Smin)','H(Sfound|bestof)']])
        
        ds['normsyn'] = ds['syn_info']/ds['syn_upper']
        ds['normWMS'] = ds['WMS(X;S)']/ds['syn_upper']
        ds.to_pickle(args.savefolder+'comparisonSRVstates'+str(args.states)+'.pkl')