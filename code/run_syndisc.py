import time
import json
import glob
import dit
from dit import Distribution
import numpy as np
import argparse
from distutils.util import strtobool

from syndisc.syndisc import disclosure_channel,self_disclosure_channel
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import params2matrix_incremental, matrix2params_incremental
from jointpdfpython3.measures import synergistic_entropy_upper_bound

def run_syndisc(d):
    args = vars(d['args'])
    prev = None
    if d['args'].prev:
        with open(d['args'].folder+d['args'].prev) as file:
            prev = json.load(file)
            prev_pars = prev['data']['parXY']
            args = prev['args']

    tot_vars = args['lenX'] + args['lenY']
    variables_X = np.arange(args['lenX'])
    variables_Y = np.arange(args['lenX'],tot_vars)
    p_XY = JointProbabilityMatrix(tot_vars,args['states'])

    for i in range(args['systems']):
        print("CUR NUM",i)
        d['data']['systemID'].append(i)
        # generate or load Pr(XY)
        if prev:
            params2matrix_incremental(p_XY,prev_pars[i])
            d['data']['syn_upper'].append(prev['data']['syn_upper'][i])

            # check if same dist by checking same upperbound
            cur = synergistic_entropy_upper_bound(p_XY[variables_X])
            if not np.allclose(cur,d['data']['syn_upper'][-1]):
                print("ALERT syn_upper not the same (i.e. pX not the same)")
        else:
            if args['dist_type'] == 'dirichlet' or args['dist_type'] == 'random':
                p_XY.generate_dirichlet_joint_probabilities()
            else:
                p_XY.generate_uniform_joint_probabilities(tot_vars,args['states'])
            d['data']['syn_upper'].append(synergistic_entropy_upper_bound(p_XY[variables_X]))

        d['data']['parXY'].append(list(matrix2params_incremental(p_XY)))

        # (time) calculation of self-synergy
        dit_syn = Distribution.from_ndarray(p_XY.joint_probabilities.joint_probabilities)
        if args['lenY'] == 0:
            before = time.time()
            syn, probs = self_disclosure_channel(dit_syn)
            d['data']['tot_runtime'].append(time.time()-before)
        else:
            # (time) calculation of synergy
            before = time.time()
            syn, probs = disclosure_channel(dit_syn)
            d['data']['tot_runtime'].append(time.time()-before)

        d['data']['I(Y;S)'].append([syn]) # if lenY=0 then I(Y;S)=I(X;S)

        # calculate I(Xi;SRV) for all Xi
        x = p_XY[variables_X].joint_probabilities.joint_probabilities.flatten()
        vgx = probs['pVgX']
        pXS = np.reshape((x*vgx).T, [args['states']]*args['lenX']+[len(x*vgx)])

        dit_XS = Distribution.from_ndarray(pXS)
        synvars = list(range(len(variables_X),len(dit_XS.rvs)))
        try:
            tot_mi = dit.shannon.mutual_information(dit_XS,variables_X,synvars)
            indiv_mutuals = [dit.shannon.mutual_information(dit_XS,[i],synvars) for i in variables_X]
            srv_entropy = dit.shannon.entropy(dit_XS,synvars)
        except AssertionError:
            print('Too large difference prob distributions ')
            indiv_mutuals = [-1 for _ in variables_X]
            srv_entropy = -1
            tot_mi = -1

        d['data']['I(X;S)'].append(tot_mi) # should be equal to I(X;S) if lenY = 0
        d['data']['I(Xi;S)'].append(indiv_mutuals)
        d['data']['H(S)'].append(srv_entropy)
        d['data']['pXS'].append(list(pXS.flatten()))
        
    d['args'] = args
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
    parser.add_argument('--num_srvs', default=1,type=int, help='Number of SRVs to search for during one search')

    # run parameters
    parser.add_argument('--systems', default=1,type=int, help='Number of different system distribution pXY')
    parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
    parser.add_argument('--exp', default=0,type=int, help='Experiment ID')
    parser.add_argument('--save', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default="../results/", help='Folder in which data is saved')
    parser.add_argument('--rowfolder', default="/row_data/", help='Folder in which each single run is saved')

    args = parser.parse_args()
    
    d = {'args':[],'data':{'systemID':[], 'parXY':[],'syn_upper':[],'runID':[],
            'pXS':[],'H(S)':[],'I(X;S)':[],'I(Y;S)':[],'I(Xi;S)':[],'tot_runtime':[]}}
    d['args'] = args
    d = run_syndisc(d)

    if args.save:
        filename_beg = 'syndisc_exp'
        filename_end = '_X'+str(d['args']['lenX'])+'_Y'+str(d['args']['lenY'])+'_States'+\
                        str(d['args']['states'])+'.json'
        files = glob.glob(d['args']['folder']+"*.json")
        while any(filename_beg+str(args.exp)+filename_end in x for x in files):
            args.exp += 1
        filename = filename_beg+str(args.exp)+filename_end
        print(d['args']['folder']+filename)
        with open(d['args']['folder']+filename,'w') as fp:
            json.dump(d,fp)