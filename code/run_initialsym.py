from cmath import nan
import time
import argparse
import os
import numpy as np
import seaborn as sns
import pandas as pd
import dit
import matplotlib.pyplot as plt
import dit
from dit import Distribution
from distutils.util import strtobool
from helpers.group_helpers import loadsyms, classifylowerorders, classifyoversized
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import matrix2params_incremental, params2matrix_incremental
from jointpdfpython3.measures import synergistic_entropy_upper_bound,symsyninfo
from helpers.compare_helpers import appendtoPXS

def run_syndisc(args,d,syms):
    """
    Given X, use the constructed SRV (from syms) with the lowest cost as initial guess. Next, optimize the constructed SRV 
    using the cost function from jointpdf and save the results.

    The code can also be used to just generate random samples of X, without computing the syn info.

    Parameters
    ----------
    args: given arguments: lenX, states, etc. 
    d: dictionary with keys to save data
    syms: list with conditional probability distributions of constructed SRVs

    Returns
    -------
    d: dictionary with optimized constructed SRV S, entropy H(S), I(Xi;S), I(X;S), etc.
    """
    pX = JointProbabilityMatrix(args.lenX,args.states)
    variables_X = np.arange(args.lenX)
    systems = [JointProbabilityMatrix(args.lenX,args.states,joint_probs=args.dist_type) for i in range(args.systems)]
    t = 0
    totpars = (args.states**args.lenX)-1
    for i,p in enumerate(systems):   
        if (t % 100)==0:
            print("CUR NUM",i,time.strftime("%H:%M:%S", time.localtime()))
        t += 1
        for k in d.keys():
            d[k].append(nan)

        # load parameters of system i
        if args.npsample:
            p = np.random.random(totpars)
            params2matrix_incremental(pX,p)
        else:
            pX = p.copy()

        pXarray = pX.joint_probabilities.joint_probabilities
        upper = synergistic_entropy_upper_bound(pX)
        subjects = np.arange(args.lenX)          
        d['systemID'][-1]=i
        d['parX'][-1]=matrix2params_incremental(pX)
        d['pX'][-1]=list(pXarray.flatten())
        d['I(X1;X2)'][-1]=pX.mutual_information([0],[1])
        d['H(Xi)'][-1]=[pX.entropy([i]) for i in variables_X]
        d['syn_upper'][-1]=upper

        if args.runtype == 'symstart':
            before = time.time()
            bestsymid,optXS = symsyninfo(args.states,args.lenX,pX,syms)
            d['initialsym_runtime'][-1]=time.time()-before
            lenopt = len(optXS)
            totmi = optXS.mutual_information(subjects,[lenopt-1])
            indivmis = sum([optXS.mutual_information([i],[lenopt-1]) for i in subjects])
            pXS = optXS.joint_probabilities.joint_probabilities
            dXSSym = dit.Distribution.from_ndarray(appendtoPXS(args.lenX,pXarray,pXS,syms[bestsymid]))   
            d['pXoptimizedsym'][-1]=list(pXS.flatten())   
            d['H(optimizedsym)'][-1]=dit.shannon.entropy(dXSSym,[lenopt])
            d['I(X;optimizedsym)'][-1]=totmi
            d['I(Xi;optimizedsym)'][-1]=indivmis
            d['H(optimizedsym|initialsym)'][-1]=dit.shannon.conditional_entropy(dXSSym,[args.lenX],[lenopt])
            d['bestsymid'][-1] = bestsymid
    return d

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='optimize constructed SRVs as initial guess using jointpdf')

    # system parameters
    parser.add_argument('--lenX', type=int, default=2,help='Number of input variables X')
    parser.add_argument('--states', type=int,default=2,help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='random', help='Distribution type (iid, uniform, dirichlet)')
    parser.add_argument('--npsample', default=False,type=lambda x: bool(strtobool(x)), help='Sample systems with numpy.random or dirichlet distribution')
    # model parameters
    parser.add_argument('--run', default=True,type=lambda x: bool(strtobool(x)), help='Run syndisc/append srv for input parameters')
    parser.add_argument('--runtype', default='symstart',help='add symSRV to each input (syndisc, sym, both) and run optimization procedure of jointpdf')
    parser.add_argument('--systems', default=20,type=int, help='Number of random dirichlet samples')
    parser.add_argument('--folder', default='test', help='Folder in which data is saved')
    parser.add_argument('--filename', default='initialsym_', help='Name of file to run')

    args = parser.parse_args()
    args.folder = '../results/'+args.folder+'/'
    filename = args.filename+'states'+str(args.states)
    if args.npsample:
        filename+='_nprandom'
    else:
        filename+='_dirichlet'
    d = {'systemID':[], 'parX':[],'pX':[],'syn_upper':[],'H(Xi)':[],'I(X1;X2)':[],
        'tot_runtime':[],'syn_info':[],'srv_data':[],\
        'pXoptimizedsym':[],'initialsym_runtime':[],'I(X;optimizedsym)':[],\
        'I(Xi;optimizedsym)':[],'H(optimizedsym)':[],'H(optimizedsym|initialsym)':[],'bestsymid':[]}

    # Load all (relevant) constructed srvs given number of states
    concsyms, syms = loadsyms(args.states)
    syms = classifyoversized(syms,args.states)
    if 'lower order' in syms.keys():
        syms = classifylowerorders(args.states,syms)

    # get indexes of all non-oversized SRVs
    listsyms = []
    symids = {}
    previd = 0
    newsyms = {}
    for k in syms.keys():
        if 'oversized' not in k:
            newsyms[k] = syms[k]
            for s in syms[k]:
                listsyms.append(s)
                symids[k] = np.arange(previd,previd+len(syms[k]))
            previd = previd+len(syms[k])
    syms = listsyms
    print(syms,len(syms))
    if args.run:
        d = run_syndisc(args,d,syms)
        df = pd.DataFrame(data=d)
        print(df)
        df['lenX'] = args.lenX
        df['states'] = args.states
        list_of_files = list(filter( lambda x: os.path.isfile(os.path.join(args.folder, x)),
                    os.listdir(args.folder)))
        exp = 1
        while filename+'.pkl' in list_of_files:
            filename += '-'+str(exp)
            exp+=1
        df.to_pickle(args.folder+filename+'.pkl')
        sns.scatterplot(data=df, x='I(Xi;optimizedsym)',y='I(X;optimizedsym)')
        plt.show()