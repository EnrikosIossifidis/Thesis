from cmath import nan
import time
import argparse
import os
import numpy as np
import pandas as pd
import dit
import pickle
from dit import Distribution
from distutils.util import strtobool
from syndisc.syndisc import self_disclosure_channel
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import matrix2params_incremental, params2matrix_incremental
from jointpdfpython3.toy_functions import append_variables_with_target_mi
from jointpdfpython3.measures import synergistic_information,synergistic_entropy_upper_bound,symsyninfo
from helpers.planes import random_orthogonal_unit_vectors,get_plane_points,plane_selected_params
from helpers.group_helpers import load_sudokus,classify_syms
from helpers.compare_helpers import appendtoPXS

def target_mi(args,target):
    jpb2 = JointProbabilityMatrix(1,args.states,joint_probs=args.dist_type)
    append_variables_with_target_mi(jpb2,1,target*jpb2.entropy([0]))
    return jpb2       

def generate_dirichlet_joint_probabilities(numvariables, numvalues):
    # todo: this does not result in random probability densities... Should do recursive
    return np.random.dirichlet([1]*(numvalues**numvariables)).reshape((numvalues,)*numvariables)

def run_syndisc(args,d,pars,syms):
    pX = JointProbabilityMatrix(args.lenX,args.states)
    variables_X = np.arange(args.lenX)
    nopars = False
    if len(pars)==0:
        nopars = True
        pars = [JointProbabilityMatrix(args.lenX,args.states) for i in range(args.samples)]
    t = 0
    for i,p in enumerate(pars):   
        if (t % 100)==0:
            print("CUR NUM",i,time.strftime("%H:%M:%S", time.localtime()))
        t += 1
        print(i,time.strftime("%H:%M:%S", time.localtime()))
        for k in d.keys():
            d[k].append(nan)
        if nopars:
            pX = p.copy()
        else:
            if min(p) < -0.00001 or max(p) > 1.00001:
                print(i,p)
                continue
            params2matrix_incremental(pX,p)
        pXarray = pX.joint_probabilities.joint_probabilities
        upper = synergistic_entropy_upper_bound(pX)
        leninp = 0
        subjects = np.arange(args.lenX)          
        d['systemID'][-1]=i
        d['parX'][-1]=matrix2params_incremental(pX)
        d['pX'][-1]=list(pXarray.flatten())
        d['I(X1;X2)'][-1]=pX.mutual_information([0],[1])
        d['H(Xi)'][-1]=[pX.entropy([i]) for i in variables_X]
        d['syn_upper'][-1]=upper
        if args.runtype == 'syndisc' or args.runtype == 'both':
            # calculation of self-synergy
            dit_syn = Distribution.from_ndarray(pXarray)
            try:
                before = time.time()
                print("START syndisc")
                syn, probs = self_disclosure_channel(dit_syn)
                print("END syndisc")
                d['tot_runtime'][-1]=time.time()-before

                # get joint pXS from syndisc package
                x = pXarray.flatten()
                vgx = probs['pVgX']
                pXS = np.reshape((x*vgx).T, [args.states]*(len(dit_syn.rvs))+[len(x*vgx)])

                dit_XS = Distribution.from_ndarray(pXS)
                synvars = list(range(args.lenX,len(dit_XS.rvs)))
                try:
                    tot_mi = dit.shannon.mutual_information(dit_XS,variables_X,synvars)
                    indiv_mutuals = [dit.shannon.mutual_information(dit_XS,[i],synvars) for i in variables_X]
                    srv_entropy = dit.shannon.entropy(dit_XS,synvars)
                except AssertionError:
                    print('Too large difference prob distributions ')
                    indiv_mutuals = [0 for _ in variables_X]
                    srv_entropy = -1
                    tot_mi = -1
                srv_data = [[srv_entropy,tot_mi,indiv_mutuals,list(pXS.flatten())]]
                d['syn_info'][-1]=syn # if lenY=0 then I(Y;S)=I(X;S)
                d['srv_data'][-1]=srv_data
                d['statesS'][-1]=len(probs['pVgX'])
                d['pXS'][-1]=list(pXS.flatten())
                leninp += len(synvars)
            except RuntimeError as err:
                print("Runtimeerror",p)
            d['lenS'][-1]=len(range(args.lenX,leninp))
        if args.runtype == 'symstart' or args.runtype =='both':
            before = time.time()
            bestsymid,optXS = symsyninfo(args.states,args.lenX,pX,syms)
            d['initialsym_runtime'][-1]=time.time()-before
            lenopt = len(optXS)
            totmi = optXS.mutual_information(subjects,[lenopt-1])
            indivmis = sum([optXS.mutual_information([i],[lenopt-1]) for i in subjects])
            pXS = optXS.joint_probabilities.joint_probabilities
            dXSSym = dit.Distribution.from_ndarray(appendtoPXS(args.lenX,pXarray,pXS,syms[bestsymid]))     
            d['pXinitialsym'][-1]=list(pXS.flatten())   
            d['H(initialsym|bestsym)'][-1]=dit.shannon.conditional_entropy(dXSSym,list(range(args.lenX,lenopt)),[lenopt])
            d['H(initialsym)'][-1]=dit.shannon.entropy(dXSSym,[lenopt])
            d['I(X;initialsym)'][-1]=totmi
            d['I(Xi;initialsym)'][-1]=indivmis
    return d

def allinputs(args,middle,mx=1000000):
    tot = ((args.states)**args.lenX)-1
    start = min(middle)
    if args.states == 2:
        numsteps = 30
        start = 0
        step = (1-start)/numsteps
        totp = [np.arange(start=start,stop=1+(0.1*step),step=step) for _ in range(tot)]
    elif args.states == 3:
        numsteps = 3
        step = (1-start)/numsteps
        totp = [np.arange(start=start,stop=1+(0.1*step),step=step) for _ in range(tot)]
        totp = [[0,0.1,1/3,] for _ in range(tot)]
    else:
        step = 2
        totp = [np.linspace(start=start,stop=1,num=step) for _ in range(tot)]
    print("EX",totp[0])
    if len(totp[0])**tot > mx:
        print("too many datapoints")
    else:
        arr = np.meshgrid(*totp)
        data = np.array(arr).reshape(len(arr), -1).T
    return data

def symparams(args,syms):
    j = JointProbabilityMatrix(args.lenX,args.states)
    mass = 1/(args.states**args.lenX)
    r = list(range(len(syms[0][0])))
    params = []
    for s in syms:
        orders = []
        ts = []
        for i in r:
            orders.append({int(j):(int(j)+i)%args.states for j in r})
            ts.append(tuple(zip(*np.where(s==i))))
        for o in orders:
            r2 = r.copy()
            for fromid in r:
                r2.remove(fromid)
                for toid in r2:
                    uni = np.full((args.states,args.states),mass)
                    fid = o[fromid]
                    toid = o[toid]
                    for t in ts[fid]:
                        uni[t] = uni[t] - mass
                    for t in ts[toid]:
                        uni[t] = uni[t] + mass
                    j.joint_probabilities.joint_probabilities = uni
                    params.append(matrix2params_incremental(j))
                r2 = r.copy()
    return params

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot computed Isyns of a random/PCA plane of parameters')

    # system parameters
    parser.add_argument('--lenX', type=int, default=2,help='Number of input variables X')
    parser.add_argument('--states', type=int,default=2,help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='uniform', help='Distribution type (iid, uniform, dirichlet)')

    # model parameters
    parser.add_argument('--run', default=False,type=lambda x: bool(strtobool(x)), help='Run syndisc/append srv for input parameters')
    parser.add_argument('--paramstype', default='random',help='Make plane by selecting two parameters (select) or make random plane (random)')
    parser.add_argument('--runtype', default='syndisc',help='Run syndisc and/or add symSRV to each input (syndisc, sym, both)')
    parser.add_argument('--steps', default=10,type=int, help='Number of points on each axis on 2D PCA grid')
    parser.add_argument('--mag', default=1,type=float, help='Magnitude of random orthogonal vectors')
    parser.add_argument('--idx', default=1,type=int, help='x axis param (must be <= (states**lenX)-1)')
    parser.add_argument('--idy', default=2,type=int, help='y axis param (must be <= (states**lenX)-1)')
    parser.add_argument('--samples', default=20,type=int, help='Number of random dirichlet samples')
    parser.add_argument('--plot_func', default='contour', help='Type of plot of 2D plane')
    parser.add_argument('--folder', default='test', help='Folder in which data is saved')
    parser.add_argument('--filename', default=None, help='Name of file to run')

    args = parser.parse_args()
    args.folder = '../results/'+args.folder+'/'
    if not args.filename:
        filename = args.paramstype+"states"+str(args.states)
    else:
        filename = args.filename+'states'+str(args.states)
    filename += args.runtype
    d = {'systemID':[], 'parX':[],'pX':[],'syn_upper':[],'H(Xi)':[],'I(X1;X2)':[],'statesS':[],
        'lenS':[],'tot_runtime':[],'syn_info':[],'srv_data':[],'pXS':[],\
            'pXinitialsym':[],'initialsym_runtime':[],\
                'H(initialsym|bestsym)':[],'I(X;initialsym)':[],'I(Xi;initialsym)':[],'H(initialsym)':[]}

    # Load all (relevant) constructed srvs given number of states
    with open('../results/sudokus/permutation_sudokus_states'+str(args.states)+'.npy', 'rb') as f:
        syms = np.load(f,allow_pickle=True)

    jpb = JointProbabilityMatrix(args.lenX,args.states,joint_probs='iid')
    middle = np.array(matrix2params_incremental(jpb))
    totpars = len(middle)
    params = []
    print("MIDDLE",middle)

    if args.paramstype == 'all':
        params = allinputs(args,middle)
    elif args.paramstype == 'symparams':
        params = symparams(args,syms)
    elif args.paramstype == 'select':
        params = plane_selected_params(args.idx,args.idy,mid=middle,steps=args.steps)
        # for i in range(totpars-1):
        #     for j in range(2,args.steps+2):
        #         # make plane by sliding two input parameters from uniform input
        #         params = params + plane_selected_params(i,i+1,mid=np.zeros(len(middle))+(j/(2*args.steps)),steps=args.steps)
    elif args.paramstype == 'randomplane':
        # make plane with two random vectors
        v1, v2 = random_orthogonal_unit_vectors(totpars)    
        params = get_plane_points(v1,v2,steps=args.steps,mag=args.mag,mid=middle)
        plane1 = np.linspace(0,args.steps,args.steps,endpoint=True)
        plane2 = np.linspace(0,args.steps,args.steps,endpoint=True)
        arr = np.meshgrid(*[plane1,plane2])
        data = np.array(arr).reshape(len(arr), -1).T
    elif args.paramstype == 'iid':
        params = [middle]
    # params = params[:100]
    print("NUM INPUT DISTRIBUTIONS",len(params))
    print(params)
    if args.run:
        d = run_syndisc(args,d,params,syms)
        df = pd.DataFrame(data=d)
        df = df.dropna(axis=1, how='all')
        if args.paramstype == 'randomplane':
            df['X'] = data[:,0]
            df['Y'] = data[:,1]
        df = df.dropna()
        print(df)
        df['steps'] = args.steps
        df['lenX'] = args.lenX
        df['states'] = args.states
        list_of_files = list(filter( lambda x: os.path.isfile(os.path.join(args.folder, x)),
                    os.listdir(args.folder)))
        exp = 1
        while filename+'.pkl' in list_of_files:
            filename += '-'+str(exp)
            exp+=1
        df.to_pickle(args.folder+filename+'.pkl')