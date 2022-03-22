import re
import os
import glob
import os.path
import argparse
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from operator import mod
from distutils.util import strtobool

def argstring(row,keys):
     curstring = ''
     for i in range(len(keys)):
          curstring += ' --'+keys[i]+'='+row[i]
     return curstring

def getstrings(args,keys):
     d = vars(args)    
     split_args = [d[k].split(',') for k in keys]
     rows = list(itertools.product(*split_args))
     strings = [argstring(r,keys) for r in rows]
     return strings

def lastN_rows(args,N):
     # Get list of all files only in the given directory
     list_of_files = filter( lambda x: os.path.isfile(os.path.join(args.folder, x)),
                    os.listdir(args.folder))
     # Sort list of files based on last modification time in ascending order
     list_of_files = sorted( list_of_files,
                    key = lambda x: os.path.getmtime(os.path.join(args.folder, x)))[-N:]
     return list_of_files

def get_notest(string):
     splitted_str = re.split('=|--| ',string)
     no_test = ''
     if splitted_str[splitted_str.index('no_test')+1] == 'True':
          no_test = '-O '
     return no_test

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
     parser = argparse.ArgumentParser(description='generate new distributions and calculate synergy')

     # system parameters
     parser.add_argument('--lenX', default='2',help='Number of input variables X')
     parser.add_argument('--lenY', default='1',help='Number of output variables Y') 
     parser.add_argument('--states', default='2',help='Number of states for each random variable')
     parser.add_argument('--dist_type', default='dirichlet', help='Distribution type')
     # run parameters
     parser.add_argument('--systems', default='1',help='Number of different probability distributions of XY')
     parser.add_argument('--c1recalcs', default='0',help='Number of recalcs for particular settings of model for python2 code')
     parser.add_argument('--c2recalcs', default='0',help='Number of recalcs for particular settings of model p3 code')
     # model parameters
     parser.add_argument('--n_repeats', default='1', help='Number of tries to find SRV')
     parser.add_argument('--num_srvs', default='1', help='Number of SRVs to search for during one search')
     parser.add_argument('--tol', default='0.05', help='Fraction of tolerated individual mutual information')
     parser.add_argument('--mm', default='None',help='Scipy optimize minimize minimization method')
     parser.add_argument('--summed_modulo', default='False', help='Start with parity SRV or not')
     parser.add_argument('--multi', default='False', help='Start with parity SRV or not')
     parser.add_argument('--no_test', default='False', help='Start with parity SRV or not')
     # exp parameters
     parser.add_argument('--exp', default=0,help='Experiment ID')
     parser.add_argument('--folder', default='preliminaries', help='Folder starting from results repository')
     parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
     parser.add_argument('--all_initials',default='False', help='Load initial guesses from previous run')
     parser.add_argument('--code1', default='run_jointpdfp2.py', help='Original (python2) way of calculating synergy')
     parser.add_argument('--code2', default='run_syndisc.py', help='New (python3) way of calculating synergy')
     parser.add_argument('--save', default='True', help='Save JSONs')
     parser.add_argument('--save_df', default=True,type=lambda x: bool(strtobool(x)), help='Save all experiments in one DataFrame')
     parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Scatterplot of results of experiments')

     file_type = '/*.json'
     systemargs = ['systems','lenX','lenY','states','dist_type','folder','save']
     modelargs = ['n_repeats','num_srvs','tol','mm','summed_modulo','all_initials']
     p3modelargs = ['multi','no_test']
     args = parser.parse_args()
     args.folder = '../results/'+args.folder+'/'

     system_strings = getstrings(args,systemargs)
     model_strings = getstrings(args,modelargs)
     p3model_strings = getstrings(args,modelargs+p3modelargs)
     onlyp3model_strings = getstrings(args,p3modelargs)

     def run_model_parameters(code,strings,prevs=[],recalcs=0):
          if prevs:
               for p in prevs:
                    cur_string = code+' --prev='+p+' --folder='+args.folder
                    if 'jointpdf' in code:
                         cur_string += ' --all_initials='+args.all_initials
                         if 'p2' in code:
                              subprocess.run('conda activate python2 && python '+cur_string+' && conda deactivate', shell=True)
                              for _ in range(recalcs):
                                   subprocess.run('conda activate python2 && python '+cur_string+' && conda deactivate', shell=True)
                         else:
                              for o in onlyp3model_strings:
                                   no_test = get_notest(o)
                                   subprocess.run('python '+no_test+cur_string+o)
                                   for _ in range(recalcs):
                                        subprocess.run('python '+no_test+cur_string+o)
                    else:
                         # run_syndisc
                         subprocess.run('python '+cur_string)
                         for _ in range(recalcs):
                              subprocess.run('python '+cur_string)
          else:
               for s in system_strings:
                    # first run to generate pXY (and initial guesses) for other experiments
                    if strings[0] != 0:
                         if 'p2' in code:
                              subprocess.run('conda activate python2 && python '+code+s+strings[0]+' && conda deactivate', shell=True)
                         else:
                              no_test = get_notest(strings[0])
                              subprocess.run('python '+no_test+code+s+strings[0])

                         # use previously generated pXY for rest of experiments 
                         args.prev = max(glob.iglob(args.folder+file_type), key=os.path.getmtime)[len(args.folder):]
                         s += ' --prev='+args.prev
                         for m in strings[1:]:
                              print("CUR STRING",m)
                              if 'p2' in code:
                                   subprocess.run('conda activate python2 && python '+code+s+m+' && conda deactivate', shell=True)
                              else:
                                   no_test = get_notest(m)
                                   subprocess.run('python '+no_test+code+s+m)

                         for _ in range(recalcs):
                              for m in strings:
                                   print("RECALC STRING",m)
                                   if 'p2' in code:
                                        subprocess.run('conda activate python2 && python '+code+s+m+' && conda deactivate', shell=True)
                                   else:
                                        no_test = get_notest(m)
                                        subprocess.run('python '+no_test+code+s+m)
                    # run syndisc (no jointpdf model params needed)
                    else:
                         print('python '+code+s)
                         subprocess.run('python '+code+s)
                         for _ in range(recalcs):
                              subprocess.run('python '+code+s)

     if args.code1 != "''" and args.code2 != "''":
          if 'p2' in args.code1:
               strings1 = model_strings
          elif 'syndisc' in args.code1:
               strings1 = [0]
          else:
               strings1 = p3model_strings
          print(strings1)
          
          if not args.prev:
               args.prev = []
          else:
               args.prev = [args.prev]
          run_model_parameters(args.code1,strings1,prevs=args.prev,recalcs=int(args.c1recalcs))

          # Get list of all files only in the given directory
          list_of_files = lastN_rows(args, len(system_strings)*len(strings1))
          if 'p2' in args.code2:
               strings2 = model_strings
          elif 'syndisc' in args.code2:
               strings2 = [0]
          else:
               strings2 = p3model_strings
          print(strings2,list_of_files)
          run_model_parameters(args.code2,strings2,list_of_files,int(args.c2recalcs))

     elif args.code1 != "''":
          if 'p2' in args.code1:
               strings1 = model_strings
          elif 'syndisc' in args.code1:
               strings1 = [0]
          else:
               strings1 = p3model_strings
          
          if not args.prev:
               args.prev = []
          else:
               args.prev = [args.prev]
          print(strings1)
          run_model_parameters(args.code1,strings1,prevs=args.prev,recalcs=int(args.c1recalcs))

     elif args.code2 != "''":
          if 'p2' in args.code2:
               strings1 = model_strings
          elif 'syndisc' in args.code2:
               strings1 = [0]
          else:
               strings1 = p3model_strings

          if not args.prev:
               args.prev = []
          else:
               args.prev = [args.prev]
          print(strings1)
          run_model_parameters(args.code2,strings1,prevs=args.prev,recalcs=int(args.c2recalcs))     

     if args.save_df:
          from helpers.helpers import get_best, get_data, swithcols
          c1 = int(args.c1recalcs)
          c2 = int(args.c2recalcs)
          
          if args.code1 != "''" and args.code2 != "''":
               last = len(system_strings)*(((1+c1)*len(strings1))\
                                             +((1+c2)*len(strings2)))
          elif args.code1!="''":
               last = len(system_strings)*((1+c1)*len(strings1))
          elif args.code2!="''":
               last = len(system_strings)*((1+c2)*len(strings1))
          else:
               last = 0

          d = get_data(args,last)
          
          # get path lengths of SRV's optimization path
          d = swithcols(['exp_sort','states','systemID','H(S)','syn_upper','syn_info','I(Xi;S)'],d)
          print(d)
          # d = get_best(d) # best srvs of multiple runs for a joint pdf Pr(X,Y)
          curtime = time.strftime("%Y%m%d-%H%M%S")
          args.exp = 'states'+str(args.states)+'time'+curtime+'.pkl'
          d.to_pickle(args.folder+args.exp)

     if args.plot:
          print("EXP",args.folder+args.exp)
          d = pd.read_pickle(args.folder+args.exp)     
          fig, ax = plt.subplots(figsize=(14,8))        
          sns.scatterplot(data=d, x='tot_runtime', y='syn_info', hue='systemID',style='exp_sort',palette='tab10',s=100,ax=ax)
          plt.show()

          