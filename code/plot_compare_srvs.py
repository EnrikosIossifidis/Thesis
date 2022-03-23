import os
import glob
import pickle
import argparse
import dit
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from datetime import datetime
from dit import Distribution
from distutils.util import strtobool

from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix 
from helpers.exhaustivesearch_helpers import append_srv

def reshp(row):
    if row['exp_sort'] == 'syndisc':
        shape = tuple([row['states'] for _ in range(row['lenX'])]+[row['statesS']])
    else:
        shape = tuple([row['states'] for _ in range(row['lenX'])]+[row['states'] for _ in range(row['lenS'])])
    return np.array(row['pXS']).reshape(shape)

def mix1x2(pxs):
    dist = Distribution.from_ndarray(pxs)
    return dit.shannon.mutual_information(dist,rvs_X=[0],rvs_Y=[1])

def getstatesS(expsort,lenpXS,states,lenX):
    if expsort == 'syndisc':
        return int(lenpXS/(states**lenX))
    else:
        return states

# choose dist type and states
def load_frame(states,dist_type,folder):

    # load file(s) as pandas dataframe
    os.chdir(folder)
    allfiles = list(glob.iglob("*.pkl"))
    name = dist_type+'states'+str(states) 
    files_dataframes = [pd.read_pickle(file) for file in allfiles if name in file]
    os.chdir("../../code")

    # prep dataframe for calculations
    d = pd.concat(files_dataframes)
    d = d.replace(np.nan, 0)
    d.loc[d['exp_sort']=='syndisc','lenS'] = 1 
    d = d[d['lenS']>0] # only keep runs where srvs where found, i.e. syn info > 0
    d['lenS'] = d['lenS'].astype(int)
    d['lenX'] = d['lenX'].astype(int)

    d['WMS(X)/Hmax(X)'] = d.apply(lambda row: (row['I(X;S)']-sum(np.array(row['I(Xi;S)'])))/row['syn_upper'],axis=1)
    d['statesS'] = d.apply(lambda row:getstatesS(row['exp_sort'],len(row['pXS']),row['states'],row['lenX']),axis=1) # compute |SRV|
    d['pXS'] = d.apply(lambda row : reshp(row), axis = 1) # reshape pXS to compute entropy measures
    d['I(X1;X2)'] = d.apply(lambda row : mix1x2(np.array(row['pXS'])), axis = 1)
    
    return d

def load_sudokus(states=[]):
    os.chdir("../results/sudokus")
    allfiles = list(glob.iglob("*.pkl"))
    allfiles = [file for file in allfiles if 'permutation' in file or 'noisy' in file]
    d = {}
    
    for state in states:
        sudos = [file for file in allfiles if'states'+str(state) in file]
        d[state] = {'permutation':[],'noisy':[]}
        d[state] = load_sudoku(sudos,d[state])
    
    os.chdir("../../code")
    return d

def load_sudoku(sudos,data):
    for s in sudos:
        if 'noisy' in s:
            with open(s, 'rb') as f:
                data['noisy'] = data['noisy'] + list(pickle.load(f))
        else:
            with open(s, 'rb') as f:
                data['permutation'] = data['permutation'] + list(pickle.load(f))
    return data

# https://stackoverflow.com/a/28345836
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)     

def mincondentropy(row,sudokus):
    args = Namespace(lenX=row['lenX'],lenS=row['lenS'],states=row['states']
                        ,pxs=row['pXS'],statesS=row['statesS'])    

    conds = []
    pXS = JointProbabilityMatrix(args.lenX+args.lenS,args.states)
    pXS.joint_probabilities.joint_probabilities = args.pxs
    for i in range(args.lenS):
        curent = pXS.entropy(variables=[args.lenX+i])
        print("curent",curent)
        cur = []
        for n in sudokus['noisy']:        
            jpb = JointProbabilityMatrix(args.lenX+args.lenS,args.states)
            jpb.joint_probabilities.joint_probabilities = args.pxs
            pxs = append_srv(jpb,n,args,np.arange(args.lenX),noisy=True).joint_probabilities.joint_probabilities
            # cond = jpb.conditional_entropy(variables=[args.lenX+i],given_variables=[args.lenX+args.lenS])
            ditdist = Distribution.from_ndarray(pxs)
            cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX+i], rvs_Y=[args.lenX+args.lenS])
            cur.append(cond/curent)
        for p in sudokus['permutation']:        
            jpb = JointProbabilityMatrix(args.lenX+args.lenS,args.states)
            jpb.joint_probabilities.joint_probabilities = args.pxs
            pxs = append_srv(jpb,p,args,np.arange(args.lenX)).joint_probabilities.joint_probabilities
            # cond = jpb.conditional_entropy(variables=[args.lenX+i],given_variables=[args.lenX+args.lenS])
            ditdist = Distribution.from_ndarray(pxs)
            cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX+i], rvs_Y=[args.lenX+args.lenS])
            cur.append(cond/curent)
        conds.append(min(cur))
    return conds

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot computed Isyns (for different model parameters) and compare found srvs to sudoku srvs')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='uniform', help='Distribution type')

    parser.add_argument('--jpdf_only', default=False,type=lambda x: bool(strtobool(x)), help='jointpdf only')

    parser.add_argument('--filefolder', default='joint_sudokus', help='From where data is retrieved and where to save fig')
    parser.add_argument('--savefolder', default='comparison', help='From where data is retrieved and where to save fig')
    parser.add_argument('--compare_srvs', default=True,type=lambda x: bool(strtobool(x)), help='Compare found SRVs with sudoku srvs')
    parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Save plot as json or not')
    args = parser.parse_args()

    args.filefolder = '../results/'+args.filefolder+"/"
    args.savefolder = '../results/'+args.savefolder+"/"

    # load found srvs
    d = load_frame(args.states,args.dist_type,args.filefolder)
    if args.jpdf_only:
        d = d[d['exp_sort']!='syndisc']

    print(d['statesS'].unique(),len(d))

    # append each soduku srv to each P(X,Sfound) for comparison     
    sudokus = load_sudokus(d['statesS'].unique())
    
    if args.compare_srvs:
        d['Hsfound_sudoku'] = d.apply(lambda row: mincondentropy(row,sudokus[row['statesS']]),axis=1) 
        # print(d)
        d = d.explode('Hsfound_sudoku')
        if args.plot: 
            fig, ax = plt.subplots(figsize=(10,5))        
            xcol = 'WMS(X)/Hmax(X)'
            ycol = 'Hsfound_sudoku'
            sns.scatterplot(data=d, x=xcol, y=ycol, 
                            hue='exp_sort',sizes=(10,60),palette='tab10',s=100,ax=ax)

            title = args.dist_type+" input dist states="+str(args.states)
            plt.title(title,fontsize=14)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(xcol,fontsize=13)
            plt.ylabel("min H(Sfound|sudoku)",fontsize=13)
            plt.show()
            fig.savefig(args.savefolder+'comparison_'+args.dist_type+'states'+ str(args.states))
            d.to_pickle(args.savefolder+'comparison_'+args.dist_type+'states'+ str(args.states)+'.pkl')

        #     ax.set(xlabel='min_i H(Sfound/Sgroup_i)', ylabel=r'$\Sigma_i$'+'I(Xi;Sfound)/I(X;Sfound)')
