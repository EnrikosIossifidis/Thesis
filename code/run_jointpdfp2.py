import time
import json
import glob
import argparse
import numpy as np
from distutils.util import strtobool
from jointpdf_original.jointpdf import JointProbabilityMatrix

def run_jpdf(args,d):
    """
    Compute for the given args the synergistic information (syn info) using jointpdf. 
    The data of all random systems is saved as a JSON file.

    Parameters
    ----------
    d: dictionary with model parameters to run (number of systems, states, etc.)

    Returns
    -------
    d: dictionary with computed synergistic information, the found SRV S,
    entropy H(S), I(Xi;S), I(X;S), etc.
    """
    # load previous systems if available
    prev = None
    if args.prev:
        with open(args.folder+args.prev) as file:
            prev = json.load(file)
            prev_pars = prev['data']['parXY']

    tot_vars = args.lenX + args.lenY
    p_XY = JointProbabilityMatrix(tot_vars,args.states)
    variables_X = np.arange(args.lenX)
    variables_Y = np.arange(args.lenX,tot_vars)
    if args.lenY == 0:
        variables_Y = variables_X

    # compute syn info and srv data for random systems
    for i in range(args.systems):
        print("CUR NUM",i,time.strftime("%H:%M:%S", time.localtime()))
        # generate or load Pr(XY)c
        if prev:
            p_XY.params2matrix_incremental(prev_pars[i])
            d['data']['syn_upper'].append(prev['data']['syn_upper'][i])
        else:
            if args.dist_type == 'dirichlet' or args.dist_type == 'random':
                p_XY.generate_dirichlet_joint_probabilities()
            else:
                p_XY.generate_uniform_joint_probabilities(tot_vars,args.states)
            d['data']['syn_upper'].append(p_XY[variables_X].synergistic_entropy_upper_bound())

        d['data']['systemID'].append(i)
        d['data']['pX'].append(list(p_XY[variables_X].joint_probabilities.joint_probabilities.flatten()))
        d['data']['parXY'].append(list(p_XY.matrix2params_incremental()))
        d['data']['I(X1;X2)'].append(p_XY.mutual_information([0],[1]))
        d['data']['H(Xi)'].append([p_XY.entropy([i]) for i in variables_X])

        # compute syn info
        before = time.time()
        syn,p_XYS = p_XY.synergistic_information(variables_Y,variables_X,tol_nonsyn_mi_frac=args.tol
                                ,minimize_method=args.mm,num_repeats_per_srv_append=args.n_repeats,
                                initial_guess_summed_modulo=args.summed_modulo,verbose=False)
        d['data']['tot_runtime'].append(time.time()-before)
        d['data']['syn_info'].append(syn)

        # save data of each srv for system i
        entS = 0
        totmi = 0
        indivmi = 0
        pfinal = []
        synvars = list(np.arange(tot_vars,len(p_XYS)))
        if len(synvars)>0:
            for svar in range(len(synvars)):
                entS += p_XYS.entropy(variables=[tot_vars+svar])
                totmi += p_XYS.mutual_information(variables_X,[tot_vars+svar])
                indivmi += sum([p_XYS.mutual_information([i],[tot_vars+svar]) for i in variables_X])
            pfinal = list(p_XYS.joint_probabilities.joint_probabilities.flatten())
        d['data']['srv_data'].append([entS,totmi,indivmi,pfinal])
        d['data']['shapeS'].append(p_XYS.joint_probabilities.joint_probabilities.shape)
    d['args'] = vars(args)
    return d

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='generate new distributions and calculate synergy')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='dirichlet', help='Distribution type')
    parser.add_argument('--n_repeats', default=3,type=int, help='Number of tries to find SRV')
    parser.add_argument('--num_srvs', default=1,type=int, help='Number of SRVs to search for during one search')
    parser.add_argument('--tol', default=0.05,type=float, help='Fraction of tolerated individual mutual information')
    parser.add_argument('--mm', default=None, type=lambda x: None if x == 'None' else x,help='Scipy optimize minimize minimization method')
    parser.add_argument('--summed_modulo', default=False,type=lambda x: bool(strtobool(x)), help='Start with parity SRV or not')
    # run parameters
    parser.add_argument('--all_initials', default=False,type=lambda x: bool(strtobool(x)), help='Start with all previous initial guess')
    parser.add_argument('--systems', default=3,type=int, help='Number of different system distribution pXY')
    parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
    parser.add_argument('--exp', default=0,type=int, help='Experiment ID')
    parser.add_argument('--save', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default="../results/test/", help='Folder in which data is saved')
    parser.add_argument('--plot', default=False)

    args = parser.parse_args()
    
    d = {'data':{'systemID':[],'pX':[],'parXY':[],'syn_upper':[],'H(Xi)':[],'I(X1;X2)':[],'runID':[],
            'shapeS':[],'tot_runtime':[],'syn_info':[],'srv_data':[]}}

    d = run_jpdf(args,d)

    if args.save:
        filename_beg = 'python2_exp'
        filename_end = args.dist_type+'_States'+str(args.states)+'.json'
        files = glob.glob(args.folder+"*.json")
        while any(filename_beg+str(args.exp)+filename_end in x for x in files):
            args.exp += 1
        filename = filename_beg+str(args.exp)+filename_end
        with open(args.folder+filename,'w') as fp:
            json.dump(d,fp)
    