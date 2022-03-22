import time
import json
import glob
import argparse
import numpy as np
from distutils.util import strtobool
from jointpdf_original.jointpdf import JointProbabilityMatrix

def run_jpdf(d,all_initials=False):
    args = vars(d['args'])
    prev = None
    if d['args'].prev:
        with open(d['args'].folder+d['args'].prev) as file:
            prev = json.load(file)
            prev_pars = prev['data']['parXY']
            args = prev['args']
            if all_initials:
                args['all_initials'] = all_initials
                prev_initials = prev['data']['all_initials']

    tot_vars = args['lenX'] + args['lenY']
    variables_X = np.arange(args['lenX'])
    variables_Y = np.arange(args['lenX'],tot_vars)

    p_XY = JointProbabilityMatrix(tot_vars,args['states'])
    name_keys = ['lenX','lenY','states']
    initial_guess = []
    for i in range(args['systems']):
        print("CUR NUM",i)
        # generate or load Pr(XY)c
        if prev:
            p_XY.params2matrix_incremental(prev_pars[i])
            d['data']['syn_upper'].append(prev['data']['syn_upper'][i])
        else:
            print(d['args'].dist_type)
            if d['args'].dist_type == 'dirichlet':
                p_XY.generate_dirichlet_joint_probabilities()
            else:
                p_XY.generate_uniform_joint_probabilities(tot_vars,args['states'])
            d['data']['syn_upper'].append(p_XY[variables_X].synergistic_entropy_upper_bound())

        curparams = list(p_XY.matrix2params_incremental())

        d['data']['systemID'].append(i)
        d['data']['parXY'].append(curparams)

        # calculate synergy multiple times for given pXY and initial guess
        if prev != None and all_initials == True:
            initial_guess = prev_initials[i]
            
        before = time.time()
        syn,p_XYS = p_XY.synergistic_information(variables_Y,variables_X,tol_nonsyn_mi_frac=args['tol']
                                ,minimize_method=args['mm'],num_repeats_per_srv_append=args['n_repeats'],
                                initial_guess_summed_modulo=args['summed_modulo'],verbose=False)
        d['data']['tot_runtime'].append(time.time()-before)
        d['data']['I(Y;S)'].append([syn])

        # # calculate I(Xi;SRV) for all Xi
        synvars = list(np.arange(tot_vars,len(p_XYS)))

        d['data']['lenS'].append(len(synvars))
        srv_entropy = 0
        indiv_mutuals = []

        if len(synvars)>0:
            srv_entropy = p_XYS.entropy(variables=synvars)   
            indiv_mutuals = [p_XYS.mutual_information([i],synvars) for i in variables_X]
            tot_mutual = p_XYS.mutual_information(variables_X,synvars)
            d['data']['H(S)'].append(srv_entropy)
            d['data']['I(Xi;S)'].append(indiv_mutuals)
            d['data']['I(Y;S)'][-1][0] = tot_mutual
            pXS = p_XYS[list(variables_X)+synvars].joint_probabilities.joint_probabilities.flatten()
            d['data']['pXS'].append(list(pXS))

        # row_data = {}
        # for k,v in d_syn.items():
        #     if k not in d['data'].keys():
        #         d['data'][k] = []
        #     d['data'][k].append(v)
        #     row_data[k] = [v]

        ## Save each calculated synergy in seperate file
        # row_name = "/"
        # for nk in name_keys:
        #     row_name += nk + str(args[nk]) + "_"
        # row_name += time.strftime("%Y% m%d-%H%M%S")+".json" # add timestamp

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
    d = {'args':[],'data':{'systemID':[], 'parXY':[],'syn_upper':[],'runID':[],
            'pXS':[],'lenS':[],'H(S)':[],'I(Y;S)':[],'I(X;S)':[],'I(Xi;S)':[],'tot_runtime':[]}}

    d['args'] = args
    d = run_jpdf(d,all_initials=args.all_initials)

    if args.save:
        filename_beg = 'python2_exp'
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