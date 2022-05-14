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

def get_prev_id(sys,prev_systems):
    l_sys = len(sys)
    for i,p in enumerate(prev_systems):
        if l_sys == len(p):
            if np.allclose(np.array(p),np.array(sys)):
                return [i]
    return []

def get_data(args,last=0,sort=False):
    files = sorted(glob.iglob(args.folder+'/*.json'), key=os.path.getmtime)[-last:]
    d = load_files(files,sort=sort)
    return d

def get_best(d, run=True):
    if run:
        sysids = list(set(d['systemID']))
        bests = []
        for sys in sysids:
            df_temp = d[d['systemID']==sys]
            argmax = df_temp['syn_info'].argmax()
            bests.append(df_temp.iloc[[argmax]])
        return pd.concat(bests)
    
    print("TEST")

def get_psort(filename):
    if 'python2' in filename:
        return 'python2'
    elif 'syndisc' in filename:
        return 'syndisc'
    else:
        return 'python3'

def swithcols(front,curfile):
    # brings list of 'front' columns to the front of file
    cols = curfile.columns.tolist()
    for f in front:
        cols.pop(cols.index(f))
    cols = front+cols
    return curfile[cols]

############################################
# LOADING, SORTING, CONVERTING HELPERS
############################################

def dict_to_pd(d, filename):
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d['data'].items()]))

    # if df['tot_runtime'].dtype == np.float64:
    #     df['tot_runtime'] = df['tot_runtime'].apply(lambda x:(time.gmtime(x).tm_hour,time.gmtime(x).tm_min,
    #                                         time.gmtime(x).tm_sec))
    
    # add other run args of data to compare model settings
    df['exp_sort'] = get_psort(filename)
    for m in d['args'].keys():
        if m == 'n_repeats':
            df['tot_repeats'] = d['args'][m]
        elif m != 'all_initials':
            df[m] = d['args'][m]

    return df

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

####################
# PLOT HELPERS
####################
def load_frame_sym(states=2,d=None):        
    # prep dataframe for calculations
    d = d.replace(np.nan, 0)
    d['lenS'] = d['lenS'].astype(int)

    col_names = []
    if 'I(X;sym)' in d.columns:
        df1 = d[['I(X;sym)','I(Xi;sym)']]
        for col in list(df1):
            for col_number in range(max(df1[col].apply(len))):
                col_names.append(col + "_" + str(col_number + 1))
        df2 = pd.concat([pd.DataFrame(df1['I(X;sym)'].tolist(), index= df1.index),
                        pd.DataFrame(df1['I(Xi;sym)'].tolist(), index= df1.index)], axis = 1)
        df2.columns = col_names
        for col in [col for col in col_names if 'Xi' in col]:
            df2[col] = [sum(a) for a in df2[col].tolist()]
        for c in col_names:
            d[c] = df2[c]
    return d, col_names

# https://stackoverflow.com/a/27179208
def scatter3d(d,x,y,z, cskey, colorsMap='jet',hue='syn_info',angle=145,states=[2,3]):
    cs = d[cskey]
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig, axes = plt.subplots(1,len(states),figsize=(20,20),subplot_kw=dict(projection='3d'))
    for i,state in enumerate(states):
        cur = d[d['states']==state]
        plotx = cur[x]
        ploty = cur[y]
        plotz = cur[z]
        print(i,state)
        axes[i].view_init(azim=angle)
        axes[i].scatter(plotx, ploty, plotz, c=scalarMap.to_rgba(cur[cskey]))
        axes[i].set_xlabel(plotx.name, fontsize=14)
        axes[i].set_ylabel(ploty.name, fontsize=14)
        axes[i].set_zlabel(plotz.name, fontsize=14)
    scalarMap.set_array(cs)
    cax = fig.add_axes([axes[-1].get_position().x1+0.033,axes[-1].get_position().y0,0.02,axes[-1].get_position().height])
    cbar = fig.colorbar(scalarMap,cax=cax)
    cbar.set_label(hue)
    cbar.set_ticks([min(cs),max(cs)])
    axes[1].set_title('states = '+str(states),fontsize=15)
    # plt.title('states='+str(states))
    plt.savefig('../../results/test/3dwmsstates'+str(states)+'.png', bbox_inches='tight')    
    plt.show()

# https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
import matplotlib.gridspec as gridspec

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec, title=None):
        self.fig = fig
        self.title = title
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1],self.title)
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs,title=None):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)
        if title:
            ax.set_title(title)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())