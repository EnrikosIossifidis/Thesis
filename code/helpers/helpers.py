import time
import json
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.cm as cmx

def swithcols(front,curfile):
    # brings list of 'front' columns to the front of curfile dataframe
    cols = curfile.columns.tolist()
    for f in front:
        cols.pop(cols.index(f))
    cols = front+cols
    return curfile[cols]

# https://stackoverflow.com/a/27179208
def scatter3d(d,x,y,z, cskey, colorsMap='jet',hue='syn_info',angle=145,states=[2,3]):
    cs = d[cskey]
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig, axes = plt.subplots(1,len(states),figsize=(18,18),subplot_kw=dict(projection='3d'))
    for i,state in enumerate(states):
        cur = d[d['states']==state]
        plotx = cur[x]
        ploty = cur[y]
        plotz = cur[z]
        print(i,state)
        axes[i].view_init(azim=angle)
        axes[i].scatter(plotx, ploty, plotz, c=scalarMap.to_rgba(cur[cskey]))
        axes[i].set_xlabel(plotx.name, fontsize=18,labelpad=15)
        axes[i].set_ylabel(ploty.name, fontsize=18,labelpad=13)
        axes[i].set_zlabel(plotz.name, fontsize=18,labelpad=11)
        axes[i].tick_params(labelsize=15)
    scalarMap.set_array(cs)
    cax = fig.add_axes([axes[-1].get_position().x1+0.05,axes[-1].get_position().y0,0.02,axes[-1].get_position().height])
    cbar = fig.colorbar(scalarMap,cax=cax)
    cbar.set_label(hue,fontsize=18)
    cbar.set_ticks([min(cs),max(cs)])
    axes[1].set_title('states = '+str(states),fontsize=20)
    return fig,cbar

###############################################################
# Load, sort and merge results of systems of different packages
###############################################################
def get_data(args,last=0,sort=False):
    files = sorted(glob.iglob(args.folder+'/*.json'), key=os.path.getmtime)[-last:]
    d = load_files(files,sort=sort)
    return d

def load_files(filenames,sort=True):
    dfs = []
    prevs = {}
    prev_systems = []
    for filename in filenames:
        d = {}
        with open(filename) as file:
            d = json.load(file)
            d['data']['runID'] = list(np.zeros(len(d['data']['systemID'])))
        
        # correct systemID and runID to concat all data
        if sort:
            for i, sys in enumerate(d['data']['parXY']):
                if len(prev_systems)>0:
                    prev_id = get_prev_id(sys,prev_systems)
                    
                    if len(prev_id)==0:
                        prev_systems.append(sys)
                        cur = len(prev_systems)-1
                        prevs[cur] = [0]
                        d['data']['systemID'][i] = cur
                        d['data']['runID'][i] = 0
                    else:
                        assert len(prev_id) == 1
                        cur = prevs[prev_id[0]]
                        cur.append(cur[-1]+1)
                        d['data']['systemID'][i] = prev_id[0]
                        d['data']['runID'][i] = cur[-1]
                else:
                    prev_systems.append(sys)
                    cur = 0
                    prevs[cur] = [0]
                    d['data']['systemID'][i] = 0
                    d['data']['runID'][i] = 0
            
        df = dict_to_pd(d, filename)    
        dfs.append(df)
    return pd.concat(dfs)

def get_prev_id(sys,prev_systems):
    l_sys = len(sys)
    for i,p in enumerate(prev_systems):
        if l_sys == len(p):
            if np.allclose(np.array(p),np.array(sys)):
                return [i]
    return []

def get_psort(filename):
    if 'python2' in filename:
        return 'python2'
    elif 'syndisc' in filename:
        return 'syndisc'
    else:
        return 'python3'

def dict_to_pd(d, filename):
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d['data'].items()]))

    # add other run args of data to compare model settings
    df['exp_sort'] = get_psort(filename)
    for m in d['args'].keys():
        if m == 'n_repeats':
            df['tot_repeats'] = d['args'][m]
        elif m != 'all_initials':
            df[m] = d['args'][m]

    return df

# combine system and runIDs of multiple pickles dfs into 1 sorted df
def sort_systems_runs(unsorted_df):
    dfs = []
    temp = unsorted_df.copy()
    for i in temp['states'].unique():
        d = temp[temp['states']==i]
        prevs = {}
        prev_systems = []
        for j, sys in enumerate(d['parXY']):
            if len(prev_systems)>0:
                prev_id = get_prev_id(sys,prev_systems)
                if len(prev_id)==0:
                    prev_systems.append(sys)
                    cur = len(prev_systems)-1
                    prevs[cur] = [0]
                    d['systemID'].iloc[j] = cur
                    d['runID'].iloc[j] = 0
                else:
                    assert len(prev_id) == 1
                    cur = prevs[prev_id[0]]
                    cur.append(cur[-1]+1)
                    d['systemID'].iloc[j] = prev_id[0]
                    d['runID'].iloc[j] = cur[-1]
            else:
                prev_systems.append(sys)
                cur = 0
                prevs[cur] = [0]
                d['systemID'].iloc[j] = 0
                d['runID'].iloc[j] = 0
        dfs.append(d)
    return pd.concat(dfs)
    
# def load_frame_sym(states=2,d=None):        
#     # prep dataframe for calculations
#     d = d.replace(np.nan, 0)
#     d['lenS'] = d['lenS'].astype(int)

#     col_names = []
#     if 'I(X;sym)' in d.columns:
#         df1 = d[['I(X;sym)','I(Xi;sym)']]
#         for col in list(df1):
#             for col_number in range(max(df1[col].apply(len))):
#                 col_names.append(col + "_" + str(col_number + 1))
#         df2 = pd.concat([pd.DataFrame(df1['I(X;sym)'].tolist(), index= df1.index),
#                         pd.DataFrame(df1['I(Xi;sym)'].tolist(), index= df1.index)], axis = 1)
#         df2.columns = col_names
#         for col in [col for col in col_names if 'Xi' in col]:
#             df2[col] = [sum(a) for a in df2[col].tolist()]
#         for c in col_names:
#             d[c] = df2[c]
#     return d, col_names