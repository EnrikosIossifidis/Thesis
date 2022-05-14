import argparse
from cgitb import small
from ntpath import join
import numpy as np
import dit
import glob
import seaborn as sns
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from distutils.util import strtobool
from helpers.compare_helpers import load_frame,normcondentropy,bestoffive
from helpers.group_helpers import oversizedsyms, load_sudokus,classify_syms,srv_to_mat

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot computed Isyns (for different model parameters) and compare found srvs to sudoku srvs')

    # model parameters
    parser.add_argument('--lenX', default=2, type=int,help='Number of input variables X')
    parser.add_argument('--lenY', default=1,type=int, help='Number of input variables Y') 
    parser.add_argument('--states', default=2,type=int, help='Number of states for each random variable')
    parser.add_argument('--paramstype', default='random', help='Distribution type')
    parser.add_argument('--nosyndisc', default=False,type=lambda x: bool(strtobool(x)), help='jointpdf only')
    parser.add_argument('--filefolder', default='test', help='From where data is retrieved and where to save fig')
    parser.add_argument('--savefolder', default='syndisc_syms', help='From where data is retrieved and where to save fig')
    parser.add_argument('--run', default=False,type=lambda x: bool(strtobool(x)), help='Compare found SRVs with sudoku srvs')
    parser.add_argument('--top', default=5,type=int, help='Number of lowest conditional entropies that are taken to compare (max 7)')
    parser.add_argument('--plot', default=True,type=lambda x: bool(strtobool(x)), help='Save plot as json or not')
    args = parser.parse_args()

    args.filefolder = '../results/'+args.filefolder+"/"
    args.savefolder = '../results/'+args.savefolder+"/"

    # Load all (relevant) constructed srvs given number of states
    allfiles = glob.glob('../results/sudokus/'+"*.npy")
    filename = None
    for file in allfiles:
        if 'constructedSRVstates'+str(args.states)+'.npy' in file:
            filename = file
            # load symSRVs for number of states
            with open(file, 'rb') as f:
                syms = np.load(f,allow_pickle=True)
    if filename is None:
        allsyms = load_sudokus([args.states,args.states+1,args.states+2])
        syms,lcursyms = load_sym(args.states)
        biggernoisies = {n:allsyms[n]['noisy'] for n in allsyms.keys() if n>args.states}
        oss = oversizedsyms(args.states,biggernoisies)
        syms['noisy'] = syms['noisy'] + oss
        syms = np.array(syms['permutation']+syms['noisy'])
        lsyms, classes = classify_syms(syms,args.states)
        for i in classes['full sym']:
            syms[i] = srv_to_mat(syms[i],args.states,args.lenX)
        selectionids = []
        for k in classes.keys():
            print(k, len(classes[k]))
            if len(classes[k])<100:
                selectionids = selectionids+classes[k]
            else:
                selectionids = selectionids+classes[k][:int(args.top*10)]
        syms = syms[selectionids]
        with open('../results/sudokus/constructedSRVstates'+str(args.states)+'.npy', 'wb') as f:
            np.save(f, np.array(syms),allow_pickle=True)

    d = load_frame(args.states,args.paramstype,args.filefolder)
    ds = d
    print(len(ds))
    if args.run:
        # compute all H(Sfound|Sym) 
        ds = ds.apply(lambda row: normcondentropy(row,args,syms),axis=1) 
        print("COMPUTE BEST OF",ds)
        ds = ds.apply(lambda row: bestoffive(row,args,syms),axis=1)
        # print(ds[['systemID','syn_upper','syn_info','H(Sfound|Smin)','H(Sfound|bestof)']])
        
        ds['normsyn'] = ds['syn_info']/ds['syn_upper']
        ds['normWMS'] = ds['WMS(X;S)']/ds['syn_upper']
        ds.to_pickle(args.filefolder+'comparison'+args.paramstype+'states'+ str(args.states)+'.pkl')

        # if args.plot: 
        #     fig, ax = plt.subplots(figsize=(10,5))        
        #     xcol = 'WMS(X)/Hmax(X)'
        #     ycol = 'H(Sfound|min_perm)'
        #     sns.scatterplot(data=d, x=xcol, y=ycol, 
        #                     hue='exp_sort',sizes=(10,60),palette='tab10',s=100,ax=ax)

        #     title = args.paramstype+" input dist w/ "+str(args.states)+'states (datasize='+str(len(d))+')'
        #     plt.title(title,fontsize=14)
        #     plt.xticks(fontsize=14)
        #     plt.yticks(fontsize=14)
        #     plt.xlabel(xcol,fontsize=13)
        #     plt.ylabel(" H(Sfound|XOR)",fontsize=13)
        #     plt.show()
        #     fig.savefig(args.savefolder+'comparison_'+args.paramstype+'states'+ str(args.states))
        #     ax.set(xlabel='min_i H(Sfound/Sgroup_i)', ylabel=r'$\Sigma_i$'+'I(Xi;Sfound)/I(X;Sfound)')
