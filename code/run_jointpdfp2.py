import time
import json
import glob
import argparse
import numpy as np
from distutils.util import strtobool
from jointpdf_original.jointpdf import JointProbabilityMatrix

def run_jpdf(args,d):
    # args = vars(d['args'])
    prev = None
    if args.prev:
        with open(args.folder+args.prev) as file:
            prev = json.load(file)
            prev_pars = prev['data']['parXY']

    tot_vars = args.lenX + args.lenY
    variables_X = np.arange(args.lenX)
    variables_Y = np.arange(args.lenX,tot_vars)
    if args.lenY == 0:
        variables_Y = variables_X

    p_XY = JointProbabilityMatrix(tot_vars,args.states)
    for i in range(args.systems):
        print("CUR NUM",i)
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
        d['data']['parXY'].append(list(p_XY.matrix2params_incremental()))
        d['data']['I(X1;X2)'].append(p_XY.mutual_information([0],[1]))
            
        before = time.time()
        syn,p_XYS = p_XY.synergistic_information(variables_Y,variables_X,tol_nonsyn_mi_frac=args.tol
                                ,minimize_method=args.mm,num_repeats_per_srv_append=args.n_repeats,
                                initial_guess_summed_modulo=args.summed_modulo,verbose=False)
        d['data']['tot_runtime'].append(time.time()-before)
        d['data']['syn_info'].append(syn)

        # # calculate I(Xi;SRV) for all Xi
        synvars = list(np.arange(tot_vars,len(p_XYS)))
        d['data']['lenS'].append(len(synvars))

        srv_data = []  
        if len(synvars)>0:
            for svar in range(len(synvars)):
                srv = []            
                srv.append(p_XYS.entropy(variables=[tot_vars+svar]))
                srv.append(p_XYS.mutual_information(variables_X,[tot_vars+svar]))
                srv.append([p_XYS.mutual_information([i],[tot_vars+svar]) for i in variables_X])
                srv.append(list(p_XYS[list(variables_X)+[tot_vars+svar]].joint_probabilities.joint_probabilities.flatten()))
                srv_data.append(srv)
        d['data']['srv_data'].append(srv_data)

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
    parser.add_argument('--folder', default="../results/", help='Folder in which data is saved')
    parser.add_argument('--rowfolder', default="/row_data/", help='Folder in which each single run is saved')

    args = parser.parse_args()
    
    # print("DEBUG",__debug__)
    d = {'data':{'systemID':[], 'parXY':[],'syn_upper':[],'I(X1;X2)':[],'runID':[],
            'lenS':[],'tot_runtime':[],'syn_info':[],'srv_data':[]}}

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