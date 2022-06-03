import numpy as np
import itertools
import pandas as pd
import time
import dit
from dit import Distribution
from datetime import datetime
from helpers.group_helpers import append_srv
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix

def reshp(row):
    if len(row['shapeS'])>row['lenX']:
        return np.array(row['pXS']).reshape(row['shapeS'])
    else:
        return []

def load_frame(states=2,dist_type='uniform',folder='../results/rundata',d=None,explode=True):
    # prep dataframe for calculations
    d['lenX'] = d['lenX'].astype(int)
    d['H(X0)'] = d[['H(Xi)']].apply(lambda row:row['H(Xi)'][0],axis=1)
    d['H(X1)'] = d[['H(Xi)']].apply(lambda row:row['H(Xi)'][1],axis=1)
    col_names = []
    df1 = d[['srv_data']]
    if explode:
        df1 = df1.explode('srv_data')
    for col in list(df1):
        for col_number in range(max(df1[col].apply(len))):
            col_names.append(col + "_" + str(col_number + 1))
    df2 = pd.concat([pd.DataFrame(df1['srv_data'].tolist(), index= df1.index)], axis = 1)
    df2.columns = col_names
    d['H(S)'] = df2['srv_data_1']
    d['I(X;S)'] = df2['srv_data_2']
    d['I(Xi;S)'] = df2['srv_data_3']
    d['pXS'] = df2['srv_data_4']
    d['WMS(X;S)'] = d.apply(lambda row: (row['I(X;S)']-row['I(Xi;S)'])/row['syn_upper'],axis=1)
    d['pXS'] = d.apply(lambda row : np.array(row['pXS']).reshape(row['shapeS']), axis = 1) # reshape pXS to compute entropy measures
    d['pX'] = d[['pX','lenX','states']].apply(lambda row : np.array(row['pX']).reshape(tuple(row['states'] for _ in range(row['lenX']))), axis = 1) # reshape pXS to compute entropy measures
    return d

def normcondentropy(row,syms):
    print(row['systemID'],datetime.fromtimestamp(time.time()))
    lenX = row['lenX']
    pX = row['pX']
    pXS = row['pXS']
    entS = row['H(S)']
    conds = []

    # append each constructed SRV and compute H(Sfound|constructed SRV)
    for s in syms:
        pXSSym = appendtoPXS(lenX,pX,pXS,s)
        d = dit.Distribution.from_ndarray(pXSSym)
        conds.append(dit.shannon.conditional_entropy(d,[lenX],[lenX+1])/entS)
    row['H(Sfound|Sym)'] = conds
    return row

def bestoffive(row,args,syms):
    print(row['systemID'],datetime.fromtimestamp(time.time()))
    conds = row['H(Sfound|Sym)']

    # sort, save and append top 5 constructed SRVs
    cond_top5 = sorted(zip(np.arange(len(conds)),conds), key=lambda t: t[1])[:args.top]
    bestids = np.array(cond_top5,dtype=int)[:,0]
    sym_top5 = syms[bestids]
    jointp = row['pXS'].copy()
    for s in sym_top5:
        jointp = appendtoPXS(args.lenX,row['pX'],jointp,s)
    
    ditd = dit.Distribution.from_ndarray(jointp)
    synvars = list(range(args.lenX+1,len(ditd.rvs)))
    row['H(Sfound|bestof)'] = dit.shannon.conditional_entropy(ditd,[args.lenX],synvars)/row['H(S)']
    row['H(Sfound|Smin)'] = cond_top5[0][1]
    row['WMS(X;Smin)'] = row['WMS(X;sym'+str(bestids[0])+')']
    row['I(Xi;Smin)'] = row['I(Xi;sym'+str(bestids[0])+')']
    return row

def appendtoPXS(lenX,pX,pXS,sym,verbose=False):
    symstates = len(sym[0])
    lenpXS = len(pXS.shape)
    ls = [np.arange(i) for i in pX.shape]
    inpstates=list(itertools.product(*ls))
    inpdict = {inp:i for i,inp in enumerate(inpstates)}
    ls = ls+[np.arange(t) for t in pXS.shape[lenX:]]+[np.arange(symstates)]
    newstates=list(itertools.product(*ls))
    newshape = tuple(list(pXS.shape)+[symstates])
    pXSSym = np.zeros(newshape)
    for i in newstates:
        curinpstate = inpdict[i[:lenX]]
        # value of symSRV given X
        curprob = sym[curinpstate][i[-1]] 
        # p(XSfoundSym) = p(XSfound)*p(Sym|X), since Sym does not depend on Sfound/other Syms
        pXSSym[i] += pXS[i[:lenpXS]]*curprob
    return pXSSym


def addbestsym(lenX,jXS,upper,syms):
    subjects = np.arange(lenX)
    lenjXS = len(jXS)
    pX = jXS[subjects].joint_probabilities.joint_probabilities
    pXS = jXS.joint_probabilities.joint_probabilities
    
    mincost = 1000
    bestpXSSym = []
    bestid = None
    for i,s in enumerate(syms):
        pXSSym = appendtoPXS(lenX,pX,pXS,s)
        d = dit.Distribution.from_ndarray(pXSSym)
        totmi = dit.shannon.mutual_information(d,subjects,[lenjXS])
        indivmi = sum([dit.shannon.mutual_information(d,[j],[lenjXS]) for j in subjects])
        wms = totmi-indivmi
        cost = (upper - wms)/upper
        if totmi!=0:
            cost+=(indivmi/totmi)
        else:
            cost+=indivmi
        if cost < mincost:
            mincost = cost
            bestpXSSym = pXSSym
            bestid = i
    return bestpXSSym, bestid

# def getbestsym(lenX,upper,pX,pXS,syms):
#     # Find the best constructed SRV with the lowest cost given X
#     subjects = np.arange(lenX)
#     lenjXS = len(pXS.shape)
    
#     mincost = 1000
#     bestpXSSym = []
#     bestid = None

#     for i,s in enumerate(syms):
#         pXSSym = appendtoPXS(lenX,pX,pXS,s)
#         d = dit.Distribution.from_ndarray(pXSSym)
#         totmi = dit.shannon.mutual_information(d,subjects,[lenjXS])
#         indivmi = sum([dit.shannon.mutual_information(d,[j],[lenjXS]) for j in subjects])
#         wms = totmi-indivmi
#         cost = (upper - wms)/upper
#         if totmi!=0:
#             cost+=(indivmi/totmi)
#         else:
#             cost+=indivmi
#         if cost < mincost:
#             mincost = cost
#             bestpXSSym = pXSSym
#             bestid = i
#     return bestpXSSym, bestid

# # https://stackoverflow.com/a/28345836
# class Namespace:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)     

# def mincondentropy(row,sudokus,nosyndisc=False):
#     if row['exp_sort']=='syndisc' and nosyndisc == True:
#         return None
#     else:
#         args = Namespace(lenX=row['lenX'],lenS=row['lenS'],states=row['states']
#                             ,pxs=row['pXS'],statesS=row['statesS'])    
#         variables_X = np.arange(args.lenX)
#         cur = []
#         pXSs = []
#         for n in sudokus['noisy']:        
#             jpb = JointProbabilityMatrix(args.lenX+1,args.states)
#             jpb.joint_probabilities.joint_probabilities = args.pxs
#             pxs = append_srv(jpb,n,args,variables_X,noisy=True).joint_probabilities.joint_probabilities
#             ditdist = Distribution.from_ndarray(pxs)
#             cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX], rvs_Y=[args.lenX+1])
#             cur.append(cond/row['H(S)'])
#             pXSs.append(pxs)
#         for p in sudokus['permutation']:        
#             jpb = JointProbabilityMatrix(args.lenX+1,args.states)
#             jpb.joint_probabilities.joint_probabilities = args.pxs
#             pxs = append_srv(jpb,p,args,variables_X).joint_probabilities.joint_probabilities
#             ditdist = Distribution.from_ndarray(pxs)
#             cond = dit.shannon.conditional_entropy(ditdist,rvs_X=[args.lenX], rvs_Y=[args.lenX+1])
#             cur.append(cond/row['H(S)'])
#             pXSs.append(pxs)

#         mincur = min(cur)
#         bestix = cur.index(mincur)
#         bestdit = Distribution.from_ndarray(pXSs[bestix])
#         besttot = dit.shannon.mutual_information(bestdit,variables_X,[args.lenX+1])
#         bestmis = [dit.shannon.mutual_information(bestdit,[i],[args.lenX+1]) for i in variables_X]

#         row['H(Sfound|min_perm)'] = mincur
#         row['I(X;min_perm)'] = besttot
#         row['sum(I(Xi;min_perm))'] = sum(np.array(bestmis))
#         return row