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
            p_XY.generate_dirichlet_joint_probabilities()
            # p_XY.generate_uniform_joint_probabilities(tot_vars,args['states'])
            d['data']['syn_upper'].append(synergistic_entropy_upper_bound(p_XY[variables_X]))

        d['data']['parXY'].append(list(matrix2params_incremental(p_XY)))

        # (time) calculation of self-synergy
        dit_selfsyn = Distribution.from_ndarray(p_XY[variables_X].joint_probabilities.joint_probabilities)
        before = time.time()
        syn, probs = self_disclosure_channel(dit_selfsyn)
        d['data']['tot_runtime'].append(time.time()-before)

        # # (time) calculation of synergy
        # dit_syn = Distribution.from_ndarray(p_XY.joint_probabilities.joint_probabilities)
        # before = time.time()
        # syn, probs = disclosure_channel(dit_syn)
        # d['data']['tot_runtime'].append(time.time()-before)

        d['data']['I(Y;S)'].append([syn])

        # calculate I(Xi;SRV) for all Xi
        x = p_XY[variables_X].joint_probabilities.joint_probabilities.flatten()
        vgx = probs['pVgX']
        pXS = np.reshape((x*vgx).T, [args['states']]*args['lenX']+[len(x*vgx)])

        dit_XS = Distribution.from_ndarray(pXS)
        # print(dit_XS)
        synvars = list(range(len(variables_X),len(dit_XS.rvs)))
        try:
            indiv_mutuals = [dit.shannon.mutual_information(dit_XS,[i],synvars) for i in variables_X]
            srv_entropy = dit.shannon.entropy(dit_XS,synvars)
        except AssertionError:
            print('Too large difference prob distributions ')
            indiv_mutuals = [-1 for _ in variables_X]
            srv_entropy = 0

        d['data']['H(S)'].append(srv_entropy)
        d['data']['I(Xi;S)'].append(indiv_mutuals)
        d['data']['pXS'].append(list(pXS.flatten()))

        ## Save each calculated synergy in seperate file
        # row_name = "/"
        # for nk in name_keys:
        #     row_name += nk + str(args[nk]) + "_"
        # row_name += time.strftime("%Y%m%d-%H%M%S")+".json" # add timestamp

        # with open("./experiments/"+args['folder']+args['rowfolder']+row_name,'w') as fp:
        #     json.dump(row_data,fp)
        
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
            'pXS':[],'H(S)':[],'I(Y;S)':[],'I(Xi;S)':[],'tot_runtime':[]}}
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
        with open(d['args']['folder']+filename,'w') as fp:
            json.dump(d,fp)

    #  if args.plot_df:
    #       from helpers.load_helpers import get_best, get_data, swithcols
    #       c1 = int(args.c1recalcs)
    #       c2 = int(args.c2recalcs)
    #       if args.code1 != "''" and args.code2 != "''":
    #            last = len(system_strings)*(((1+c1)*len(strings1))\
    #                                          +((1+c2)*len(strings2)))
    #       elif args.code1!="''":
    #            last = len(system_strings)*((1+c1)*len(strings1))
    #       elif args.code2!="''":
    #            last = len(system_strings)*((1+c2)*len(strings1))
    #       else:
    #            last = 0

    #       d = get_data(args,last)
          
    #       # get path lengths of SRV's optimization path
    #       d = swithcols(['systemID','runID','syn_info','tot_runtime','exp_sort','multi','no_test','tot_repeats'],d)
    #       # d = get_best(d) # best srvs of multiple runs for a joint pdf Pr(X,Y)
    #       curtime = time.strftime("%Y%m%d-%H%M%S")
    #       filename = args.folder+'exp'+str(args.exp)+'states'+str(args.states)+curtime+'.pkl'
    #       d.to_pickle(filename)
    #       print(d.explode('all_initials'))

    #  if args.plot:
    #       d = pd.read_pickle(filename)     
    #       fig, ax = plt.subplots(figsize=(14,8))        
    #       sns.scatterplot(data=d, x='tot_runtime', y='syn_info', hue='systemID',style='multi',palette='tab10',s=100,ax=ax)
    #       plt.show()