import numpy as np
import itertools
import pandas as pd
import time
import os
import pickle
import sys
import glob
import dit
from dit import Distribution
from datetime import datetime
from itertools import cycle, permutations, combinations
from helpers.group_helpers import append_srv
sys.path.append('../')
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix

def reshp(row):
    if row['exp_sort'] == 'syndisc':
        shape = tuple([row['states'] for _ in range(row['lenX'])]+[row['statesS']])
    else:
        shape = tuple([row['states'] for _ in range(row['lenX'])]+[row['states']])
    return np.array(row['pXS']).reshape(shape)

def getstatesS(expsort,lenpXS,states,lenX):
    if expsort == 'syndisc':
        return int(lenpXS/(states**lenX))
    else:
        return states

def load_frame(states,dist_type,folder):
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
    d = d.explode('srv_data')
    d['lenS'] = d['lenS'].astype(int)
    d['lenX'] = d['lenX'].astype(int)

    col_names = []
    df1 = d[['srv_data']]
    for col in list(df1):
        for col_number in range(max(df1[col].apply(len))):
            col_names.append(col + "_" + str(col_number + 1))
    df2 = pd.concat([pd.DataFrame(df1['srv_data'].tolist(), index= df1.index)], axis = 1)
    df2.columns = col_names
    d['H(S)'] = df2['srv_data_1']
    d['I(X;S)'] = df2['srv_data_2']
    d['I(Xi;S)'] = df2['srv_data_3']
    d['pXS'] = df2['srv_data_4']
    d['WMS(X)/Hmax(X)'] = d.apply(lambda row: (row['I(X;S)']-sum(np.array(row['I(Xi;S)'])))/row['syn_upper'],axis=1)
    d['statesS'] = d.apply(lambda row:getstatesS(row['exp_sort'],len(row['pXS']),row['states'],row['lenX']),axis=1) # compute |SRV|
    d['pXS'] = d.apply(lambda row : reshp(row), axis = 1) # reshape pXS to compute entropy measures
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

def mincondentropy(row,sudokus,nosyndisc=False):
    if row['exp_sort']=='syndisc' and nosyndisc == True:
        return None
    else:
        args = Namespace(lenX=row['lenX'],lenS=row['lenS'],states=row['states']
                            ,pxs=row['pXS'],statesS=row['statesS'])    
        variables_X = np.arange(args.lenX)
        cur = []
        pXSs = []
        for n in sudokus['noisy']:        
            jpb = JointProbabilityMatrix(args.lenX+1,args.states)
            jpb.joint_probabilities.joint_probabilities = args.pxs
            pxs = append_srv(jpb,n,args,variables_X,noisy=True).joint_probabilities.joint_probabilities
            ditdist = Distribution.from_ndarray(pxs)
            cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX], rvs_Y=[args.lenX+1])
            cur.append(cond/row['H(S)'])
            pXSs.append(pxs)
        for p in sudokus['permutation']:        
            jpb = JointProbabilityMatrix(args.lenX+1,args.states)
            jpb.joint_probabilities.joint_probabilities = args.pxs
            pxs = append_srv(jpb,p,args,variables_X).joint_probabilities.joint_probabilities
            ditdist = Distribution.from_ndarray(pxs)
            cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX], rvs_Y=[args.lenX+1])
            cur.append(cond/row['H(S)'])
            pXSs.append(pxs)

        mincur = min(cur)
        bestix = cur.index(mincur)
        bestdit = Distribution.from_ndarray(pXSs[bestix])
        besttot = dit.shannon.mutual_information(bestdit,variables_X,[args.lenX+1])
        bestmis = [dit.shannon.mutual_information(bestdit,[i],[args.lenX+1]) for i in variables_X]

        row['H(Sfound|min_perm)'] = mincur
        row['I(X;min_perm)'] = besttot
        row['sum(I(Xi;min_perm))'] = sum(np.array(bestmis))
        return row
