import argparse
import pickle

from numpy.lib.function_base import append
from numpy.linalg.linalg import cond
from funcy.seqs import first
import numpy as np
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from distutils.util import strtobool
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
# from helpers.group_helpers import group_by_cond,plot_data, modulo_entropies, append_srv, find_sudokus, cond_entropies, operations, all_modulo_srvs
# from measures import append_with_mpi
from helpers.group_helpers import all_sudokus

@np.vectorize
def lookup(a, b):
    return cond_mat[a][b]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate the set of all modulo tables and the subset of unique, independent srvs')

    # model parameters
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='iid', help='Distribution type')
    parser.add_argument('--p', default=0.0,type=float, help='scale parameter for shape of Pr(S|X)')

    parser.add_argument('--show_mods', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_mods', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--load_conds', default=True,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--save', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--save_df', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='../experiments/data_acc_speed/srvs_analysis/', help='Folder in which data is saved')

    args = parser.parse_args()

    print("ALL MODS",all_sudokus(args.states))

#     # loads mods and or conditional entropy matrix between found mods
#     filename ='all_sudokus_states'+str(args.states)+'.txt'
#     if args.load_mods:
#         with open(args.folder+filename+'.txt', "rb") as fp:   # Unpickling
#             mods = pickle.load(fp)
#     else:
#         # get all sudoku srvs (exhaustively)
#         mods = all_modulo_srvs(args)
#         if args.save:
#             with open(args.folder+filename+'.txt', "wb") as fp:   #Pickling
#                 pickle.dump(mods, fp)
    
#     print(mods[:1350])
    
#     if args.load_conds:
#         filename = 'cond_mat'+str(args.lenX)+'_Y'+str(args.lenY)+'_states'+str(args.states)
#         with open(args.folder+filename+'.txt', 'rb') as f:
#             cond_mat = pickle.load(f)
#     else:
#         # calculate H(Sj|Si) and H(Si|Sj) for each SRV i
#         cond_mat = cond_entropies(mods, args)
#         filename = 'cond_mat'+str(args.lenX)+'_Y'+str(args.lenY)+'_states'+str(args.states)
#         if args.save:
#             with open(args.folder+filename+'.txt', "wb") as fp:   #Pickling
#                 pickle.dump(cond_mat, fp)

    
#     # create and check if each mod is included once in each group
#     groups = group_by_cond(cond_mat)
#     print("ALL GROUPs", groups, len(groups))
#     all = []
#     for g in groups.values():
#         all = all + g
#     all = sorted(all)

#     # included once check
#     if not list(set(all)) == all:
#         print("Some mod srvs are included twice")

#     if not len(mods) == len(all):
#         print("Not all mods included in groups")

#     # check if each element in each group is redundant given another element in that group
#     srvs = []  
#     srvs_for_comparison = {'mod_pdf':[], 'mod_srv':[]} 
#     for i, group in enumerate(groups.values()):
#         # check for equality in each group
#         X, Y = np.meshgrid(group, group)
#         if not np.array_equal(lookup(X,Y),np.zeros((len(group),len(group))), equal_nan=True):
#             print("Not all mods in group "+str(i)+" are the same")
#         else:
#             # check for independency from input variables
#             cur = mods[group[0]]
#             pXSi, mutuals = append_srv(JointProbabilityMatrix(args.lenX,args.states,joint_probs='iid')\
#                             ,cur,args)
#             if np.all(np.array(mutuals[0])==0):
#                 srvs_for_comparison['mod_srv'].append(cur)
#                 srvs_for_comparison['mod_pdf'].append(pXSi.joint_probabilities.joint_probabilities)
#     print("LEN Exhaustive Srvs",len(srvs_for_comparison['mod_srv']))

#     # generate SRVs with cycles
#     with open(args.folder+'cycles_tm_5states.pkl', "rb") as fp:   # Unpickling
#         cycles = pickle.load(fp)
    
#     print(cycles[args.states])

#     if args.show_mods:
#         mis, conds = modulo_entropies(srvs_for_comparison['mod_srv'],args)
#         print("I(Si;Sj)")
#         print(mis)
#         print("H(Si|Sj)") 
#         print(conds)

#         if len(srvs_for_comparison['mod_srv'])<9 or args.states==5:
            
#             pX = JointProbabilityMatrix(args.lenX, args.states, joint_probs='iid')
#             srvs = []
#             if args.states==5:
#                 for m5 in [0,4,5]:
#                     srvs.append(srvs_for_comparison['mod_srv'][m5])
#             else:
#                 srvs = srvs_for_comparison['mod_srv']

#             for s in srvs:
#                 pXSi, mutuals = append_srv(pX, s, args)
#             pdf = pXSi.joint_probabilities.joint_probabilities
#             # df_mods = pd.DataFrame({'states':args.states, 'mod_srvs_pdf':pdf})
#             print(Distribution.from_ndarray(pdf))

#     if args.save_df:
#         df = pd.DataFrame(data=srvs_for_comparison)    
#         filename = 'srv_per_modulo_group_'+'X'+str(args.lenX)+'_Y'+str(args.lenY)+'_states'+str(args.states)+'.pkl'
#         df.to_pickle(args.folder+filename)