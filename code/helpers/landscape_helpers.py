from matplotlib import colors
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import os
import glob
import json
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import pandas as pd
import sys
sys.path.append('../')
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import params2matrix_incremental

############
# RECENT
############
def plot_plane(Z_params,Z,plot_func='contour'):
    ax_x = np.arange(Z.shape[0])
    ax_y = np.arange(Z.shape[1])
    X,Y = np.meshgrid(ax_x,ax_y)
    fig = plt.figure()
    if plot_func == 'contour':
        ax = plt.contour(X,Y,Z, cmap='Spectral')
        try:
            fig.colorbar(ax)
        except IndexError:
            print("Too small cost range for colorbar")
    else:
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='winter', edgecolor='none')

    # plot params as axis labels
    assert Z.shape[0] == Z.shape[1]
    idx = [0,Z.shape[0]-1,Z.shape[0]*(Z.shape[0]-1)]
    xvals = [0,ax_x[-1]]
    xtick = [Z_params[idx[0]],Z_params[idx[1]]]
    yval = [ax_y[-1]]
    ytick = [Z_params[idx[-1]]]
    plt.xticks(xvals,xtick,fontsize=10)
    plt.yticks(yval,ytick,fontsize=10)
    plt.yticks(rotation=70)
    plt.show()

def cost_func_landscape(jXS,parX,free_params,subjects,syn_upper,extra_cost_rel_error=True,agnostic_about=None):
    if min(free_params) < -0.00001 or max(free_params) > 1.00001:
        # high cost for invalid parameter values
        # note: maximum cost normally from this function is about 2.0
        return -0.1
        # return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
        #                             + [np.abs(p) for p in free_params if p < 0.0])

    free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors
    params2matrix_incremental(jXS,list(parX) + list(free_params))

    len_subs = len(subjects)
    len_pXSnew = len(jXS)

    tot_mi = jXS.mutual_information(range(len_subs, len_pXSnew),
                    range(len_subs)) # I(X;S))

    indiv_mis = [jXS.mutual_information([var],range(len_subs,len_pXSnew))
                    for var in range(len_subs)] # I(Xi;S)

    syninfo_naive = tot_mi - sum(indiv_mis)

    # # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
    # # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
    if not subjects is None:
        # this can be considered to be in range [0,1] although particularly bad solutions can go >1
        if not extra_cost_rel_error:
            cost = (syn_upper - jXS.synergistic_information_naive(
                variables_SRV=range(len_subs, len_pXSnew),
                variables_X=subjects)) / syn_upper
        else:
            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            cost = (syn_upper - syninfo_naive)/syn_upper
            # add an extra cost term for the fraction of 'individual' information versus the total information
            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            if tot_mi != 0:
                cost += sum(indiv_mis) / tot_mi
            else:
                cost += sum(indiv_mis)

    # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
    if not agnostic_about is None:
        assert not subjects is None, 'how can all variables be subject_variables and you still want' \
                                            ' to be agnostic about certain (other) variables? (if you did' \
                                            ' not specify subject_variables, do so.)'

        if np.ndim(agnostic_about) == 1:
            # note: cost term for agnostic is in [0,1]
            
            agn_mis = jXS.mutual_information([-1], agnostic_about)
            upper_bound_agnostic_information = jXS.entropy(agnostic_about)
            cost += agn_mis / syn_upper

    assert np.isscalar(cost)
    assert np.isfinite(cost)
    return float(cost)

##########
# OLD
##########
def get_args(curfile):
    args = {}
    args['given_params'] = list(curfile['parX'])[0]
    args['parXYSold'] = list(curfile['parXYSold'])[0]
    number = len(args['parXYSold'])+1
    base = int(curfile['states'])
    exponent = int(math.log(number, base))  # = 3

    args['pdf_subjects_snew'] = JointProbabilityMatrix(int(curfile['lenX'])+1,int(curfile['states']))
    args['pdf_XYSold'] = JointProbabilityMatrix(exponent,int(curfile['states']))
    params2matrix_incremental(args['pdf_XYSold'],args['parXYSold'])

    if len(np.arange(int(curfile['lenX']+curfile['lenY']),exponent))>0:
        args['agnostic_about'] = np.arange(int(curfile['lenX']+curfile['lenY']),exponent)
    else:
        args['agnostic_about'] = None
    args['agn_upperbound'] = args['pdf_XYSold'].entropy(args['agnostic_about'])
    args['subjects'] = np.arange(int(curfile['lenX']))
    args['syn_upperbound'] = args['pdf_XYSold'][args['subjects']].entropy()-max([args['pdf_XYSold'][[s]].entropy() for s in args['subjects']])
    args['num_search_srvs'] = int(curfile['num_srvs'])
    return args

def get_ranges(trans_X,steps,eps=0.05):
    xmin = min(trans_X[:,0])-eps
    xmax = max(trans_X[:,0])+eps
    xrange = np.linspace(xmin,xmax,steps)
    ymin = min(trans_X[:,1])-eps
    ymax = max(trans_X[:,1])+eps
    yrange = np.linspace(ymin,ymax,steps)
    xrange = np.vstack((xrange,np.zeros(steps)))
    yrange = np.vstack((np.zeros(steps),yrange))
    return [xrange.T,yrange.T]
    
# get 2D pca representation
def pca_data(data,steps,d=2):
    pca = PCA(n_components=d)
    pca.fit(data)
    trans_X = pca.transform(data)
    inv_trans_X = pca.inverse_transform(trans_X)
    ax_ranges = get_ranges(trans_X,steps)
    
    data_2D = []
    for i in ax_ranges[0]:
        data_2D.append([])
        for r in ax_ranges[1]:
            curv = i+r
            data_2D[-1].append(curv)
    data_2D = np.array(data_2D)
    return inv_trans_X,trans_X,pca.inverse_transform(data_2D),data_2D,ax_ranges

def plot_2D(curfile,grid_steps=10):
    """Plots 2D optimization path of each try when calling synergistic_information. 
    For each try there are n_repeats paths. The path with the best final params is plotted."""

    curpathspars = curfile[['all_paths','parXYSold']]
    curpathspars = curpathspars.apply(pd.Series.explode)
    for i in range(len(curpathspars)):
        print("TRY",i)
        cur = curfile.copy()
        curdf = curpathspars.iloc[[i]].copy()

        # TO DO: Now first min path is taken, but needs to be the path 
        # with final params that have minimum cost 
        cur['all_paths'] = curdf['all_paths']
        cur['parXYSold'] = curdf['parXYSold']
        args = get_args(cur)
        curpath = list(curdf['all_paths'])[0]
        inv_trans_X,trans_X,grid_nD,grid_2D,axes_2D = pca_data(curpath,grid_steps)
        plot_landscape_path(args,trans_X,grid_nD,grid_2D,axes_2D,grid_steps)

def plot_landscape_path(args,trans_X,grid_nD,grid_2D,axes_2D,range_steps):
    # idx = [0,range_steps-1,range_steps*(range_steps-1)]
    # ax_x = axes_2D[0][:,0]
    # ax_y = axes_2D[1][:,1]
    # X, Y = np.meshgrid(ax_x,ax_y)
    # Z = []
    # for i in range(len(ax_x)):
    #     Z.append([])
    #     for j in range(len(ax_y)):
    #         # print(grid_nD[i][j],cost_func_subjects_only(args,grid_nD[i][j]))
    #         Z[-1].append(cost_func_subjects_only(args,grid_nD[i][j]))

    # Z = np.array(Z)

    plt.plot(trans_X[:,0],trans_X[:,1])
    plt.scatter(trans_X[0][0],trans_X[0][1],color='red') # initial params
    plt.scatter(trans_X[-1][0],trans_X[-1][1],color='green') # final params
    plt.contour(X, Y, Z, cmap='Spectral')
    try:
        plt.colorbar()
    except IndexError:
        print("Too small cost range for colorbar")

    # plot params as axis labels
    xvals = [min(trans_X[:,0]),max(trans_X[:,0])]
    xtick = [grid_nD[0][0],grid_nD[-1][0]]
    yval = [max(trans_X[:,1])]
    ytick = [grid_nD[0][-1]]
    plt.xticks(xvals,xtick,fontsize=10)
    plt.yticks(yval,ytick,fontsize=10)
    plt.yticks(rotation=90)
    # plt.scatter(grid_2D[i][j][0],grid_2D[i][j][1],color='orange')
    
    plt.show()


def cost_func_subjects_only(args,free_params,extra_cost_rel_error=True):
    if min(free_params) < -0.00001 or max(free_params) > 1.00001:
        # warnings.warn('scipy\'s minimize() is violating the parameter bounds 0...1 I give it: '
        #             + str(args['free_params']))

        # high cost for invalid parameter values
        # note: maximum cost normally from this function is about 2.0
        # return -1
        return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
                                    + [np.abs(p) for p in free_params if p < 0.0])

    # assert max(free_params) <= 1.00001, \
    #     'scipy\'s minimize() is violating the parameter bounds 0...1 I give it: ' + str(free_params)

    free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors
    pdf_subjects_snew = args['pdf_subjects_snew'].copy()
    params2matrix_incremental(pdf_subjects_snew,list(args['given_params']) + list(free_params))

    # make a conditional distribution of the synergistic variables conditioned on the subject variables
    # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
    # agnostic constraint
    len_subs = len(args['subjects'])
    len_pXSnew = len(pdf_subjects_snew)
    cond_pdf_syns_on_subjects = pdf_subjects_snew.conditional_probability_distributions(
        range(len_subs)) # pSnew|subjects

    # assert type(cond_pdf_syns_on_subjects) == dict \
    #     or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)
    pdf_XYSold = args['pdf_XYSold'].copy()
    pdf_XYSold.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                args['subjects']) #pXYSoldSnew

    tot_mi = pdf_XYSold.mutual_information(range(len_subs, len_pXSnew),
                    range(len_subs)) # I(X;S))

    indiv_mis = [pdf_XYSold.mutual_information([var],range(len_subs,len_pXSnew))
                    for var in range(len_subs)] # I(Xi;S)

    syninfo_naive = tot_mi - sum(indiv_mis)

    # # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
    # # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
    if not args['subjects'] is None:
        assert pdf_subjects_snew.numvariables == len_subs + args['num_search_srvs']

        # this can be considered to be in range [0,1] although particularly bad solutions can go >1
        if not extra_cost_rel_error:
            cost = (args['syn_upperbound'] - pdf_subjects_snew.synergistic_information_naive(
                variables_SRV=range(len_subs, len_pXSnew),
                variables_X=args['subjects'])) / args['syn_upperbound']
        else:
            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            cost = (args['syn_upperbound'] - syninfo_naive)/args['syn_upperbound']
            # add an extra cost term for the fraction of 'individual' information versus the total information
            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            if tot_mi != 0:
                cost += sum(indiv_mis) / tot_mi
            else:
                cost += sum(indiv_mis)

    # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
    if not args['agnostic_about'] is None:
        assert not args['subjects'] is None, 'how can all variables be subject_variables and you still want' \
                                            ' to be agnostic about certain (other) variables? (if you did' \
                                            ' not specify subject_variables, do so.)'

        if np.ndim(args['agnostic_about']) == 1:
            # note: cost term for agnostic is in [0,1]
            
            agn_mis = pdf_XYSold.mutual_information([-1], args['agnostic_about'])
            upper_bound_agnostic_information = pdf_XYSold.entropy(args['agnostic_about'])
            cost += agn_mis / args['agn_upperbound']

    assert np.isscalar(cost)
    assert np.isfinite(cost)
    return float(cost)

