"""
Comparing the S^a(X->Y) (Rosas, syndisc) and Isyn(X->Y) (Quax, syndisc) code implementation. 
Since the jointpdf is in python2, the functions involving the computation of Isyn(X->Y) 
are also converted to python3. Besides comparing the packages for synergy,
experiments can be done for the jointpdf package by comparing values of some of the code's parameters. 

References:
    R. Quax, O. Har-Shemesh and P.M.A. Sloot (2017). Quantifying Synergistic Information 
    Using Intermediate Stochastic Variables

    F. Rosas*, P. Mediano*, B. Rassouli and A. Barrett (2019). An operational
    information decomposition via synergistic disclosure.

    B. Rassouli, Borzoo, F. Rosas, and D. Gündüz (2018). Latent Feature
    Disclosure under Perfect Sample Privacy. In 2018 IEEE WIFS, pp. 1-7.

Distributed under the modified BSD licence. See LICENCE for details.

Enrikos Iossifidis, 2022
"""
import re
import os
import os.path
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from distutils.util import strtobool

def getcodes(code1,code2):
     # Compute from the given strings of code which python scripts need to be run.
     p2 = None
     p3 = None
     syndisc = None
     if code1 != "''":
          if 'p2' in code1:   
               p2 = code1
          elif 'p3' in code1:
               p3 = code1
          else:
               syndisc = code1
     if code2 != "''":
          if 'p2' in code2:   
               p2 = code2
          elif 'p3' in code2:
               p3 = code2
          else:
               syndisc = code2
     return p2,p3,syndisc

def argstring(row,keys):
     # convert args into one string for specific experiment
     curstring = ''
     for i in range(len(keys)):
          curstring += ' --'+keys[i]+'='+row[i]
     return curstring

def getstrings(args,keys):
     # convert args into different experiments with model parameters
     d = vars(args)    
     split_args = [d[k].split(',') for k in keys]
     rows = list(itertools.product(*split_args))
     strings = [argstring(r,keys) for r in rows]
     return strings

def lastN_rows(folder,N):
     # Get list of all files only in the given directory
     list_of_files = filter( lambda x: os.path.isfile(os.path.join(folder, x)),
                    os.listdir(folder))
     # Sort list of files based on last modification time in ascending order
     list_of_files = sorted( list_of_files,
                    key = lambda x: os.path.getmtime(os.path.join(folder, x)))[-N:]
     return list_of_files

def get_notest(string):
     # compute whether turning assertions off is given as argument
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
     parser.add_argument('--lenY', default='0',help='Number of output variables Y') 
     parser.add_argument('--states', default='2',help='Number of states for each random variable')
     parser.add_argument('--dist_type', default='random', help='Distribution type')
     # run parameters
     parser.add_argument('--systems', default='1',help='Number of different probability distributions of XY')
     parser.add_argument('--c1recalcs', default='0',help='Number of recalcs for particular settings of model for python2 code')
     parser.add_argument('--c2recalcs', default='0',help='Number of recalcs for particular settings of model p3 code')
     # model parameters
     parser.add_argument('--n_repeats', default='1', help='Number of tries to find SRV')
     parser.add_argument('--tol', default='0.05', help='Fraction of tolerated individual mutual information')
     parser.add_argument('--summed_modulo', default='False', help='Start with parity SRV or not')
     parser.add_argument('--no_test', default='False', help='Start with parity SRV or not')
     # exp parameters
     parser.add_argument('--exp', default=0,help='Experiment ID')
     parser.add_argument('--folder', default='test', help='Folder starting from results repository')
     parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
     parser.add_argument('--code1', default='run_jointpdfp2.py', help='Original (python2) way of calculating synergy')
     parser.add_argument('--code2', default='run_syndisc.py', help='New (python3) way of calculating synergy')
     parser.add_argument('--save', default='True', help='Save JSONs')
     parser.add_argument('--sort', default=True,type=lambda x: bool(strtobool(x)), help='Sort all experiments by system and run (not necessary for 1 run)')
     parser.add_argument('--save_df', default=True,type=lambda x: bool(strtobool(x)), help='Save all experiments in one DataFrame')
     parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Scatterplot of results of experiments')

     file_type = '/*.json'
     systemargs = ['systems','lenX','lenY','states','dist_type','folder','save']
     modelargs = ['n_repeats','tol','summed_modulo']
     p3modelargs = ['no_test']
     args = parser.parse_args()
     args.folder = '../results/'+args.folder+'/'

     p2,p3,syndisc = getcodes(args.code1,args.code2)
     system_strings = getstrings(args,systemargs)
     model_strings = getstrings(args,modelargs)
     p3model_strings = getstrings(args,modelargs+p3modelargs)
     onlyp3model_strings = getstrings(args,p3modelargs)

     # run each experiment for syndisc and jointpdf. Note: for jointpdf multiple model parameters
     # and therefore experiments can be run for the same systems
     for s in system_strings:
          if syndisc:
               subprocess.run('python '+syndisc+s)
               prev = lastN_rows(args.folder,1)        
          else:
               prev=['None']  
          for m in model_strings:
               if p2:
                    if syndisc:
                         subprocess.run('conda activate python2 && python '+p2+s+m+' --prev='+prev[0]+' && conda deactivate', shell=True)
                    else:
                         subprocess.run('conda activate python2 && python '+p2+s+m+' && conda deactivate', shell=True)
                         prev = lastN_rows(args.folder,1)
               for p in p3model_strings:
                    if p3:
                         no_test = get_notest(p)
                         print(no_test)
                         subprocess.run('python '+p3+no_test+s+m+p+' --prev='+prev[0])

     if args.save_df:
          from helpers.helpers import get_data, swithcols
          # get the last saved experiments and put them into one DataFrame
          last = 1
          if syndisc:
               last *= len(system_strings)
          if p2:
               last = last + (last*len(model_strings))
          if p3:
               last = last + (last*len(model_strings)*len(p3model_strings))
          print('tot last files',last)
          d = get_data(args,last=last,sort=args.sort)
          d = swithcols(['exp_sort','systemID','syn_upper','shapeS','srv_data'],d)
          args.exp = args.dist_type+'states'+str(args.states)+'.pkl'
          d.to_pickle(args.folder+args.exp)
          d = pd.read_pickle(args.folder+args.exp)     
          print(d)

          if args.plot:
               fig, ax = plt.subplots(figsize=(14,8))        
               sns.scatterplot(data=d, x='tot_runtime', y='syn_info', hue='systemID',style='exp_sort',palette='tab10',s=100,ax=ax)
               plt.show()

          