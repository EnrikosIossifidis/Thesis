
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../')
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import params2matrix_incremental

def swithcols(front,curfile):
    # brings list of 'front' columns to the front of file
    cols = curfile.columns.tolist()
    for f in front:
        cols.pop(cols.index(f))
    cols = front+cols
    return curfile[cols]

# return two orthogonal unit vectors
def random_orthogonal_unit_vectors(dim):
    v1 = np.random.rand(dim)
    v1 = v1/np.linalg.norm(v1)
    v2 = np.random.rand(dim)
    v2 -= v2.dot(v1) * v1
    v2 = v2/np.linalg.norm(v2)
    return v1, v2

def get_plane_points(v1,v2,mag=1,steps=10,mid=[]):
    mag = 1
    dim = len(v1)
    if len(mid) == 0:
        mid = [0.5 for _ in range(dim)]
    v3 = (v1*mag)
    v4 = (v2*mag)

    # get plane given random orthogonal unit vectors
    plane1 = np.linspace(mid - (0.5*v3),mid + (0.5*v3),steps)
    plane2 = np.linspace(mid - (0.5*v4),mid + (0.5*v4),steps)
    plane = []
    for p1 in plane1:
        for p2 in plane2:
            plane.append(p1+(p2-mid))
    return np.array(plane)

def plane_selected_params(ix,iy,mid,steps=10):
    bef = list(range(min(ix,iy)))
    m = list(range(min(ix,iy)+1,max(ix,iy)))
    aft = list(range(max(ix,iy)+1,len(mid)))

    # get plane given random orthogonal unit vectors
    plane1 = np.linspace(0,1,steps,endpoint=True)
    plane2 = np.linspace(0,1,steps,endpoint=True)
    # plane1 = list(np.arange(0,1,mid[ix]/(steps/2)))+[1]
    # plane2 = list(np.arange(0,1,mid[iy]/(steps/2)))+[1]
    arr = np.meshgrid(*[plane1,plane2])
    data = np.array(arr).reshape(len(arr), -1).T
    conc = []
    for d in data:
        conc.append(np.concatenate([mid[bef],[d[0]],mid[m],[d[1]],mid[aft]]))
    return conc

def get_plane_values(args,plane,syn_upper,parX):
    steps=args.steps
    subjects = list(range(args.lenX))
    Z = []
    for i in range(steps):
        Z.append([])
        for j in range(steps):
            plane_id = (steps*i)+j

            # get cost of curparams
            Z[-1].append(get_cost(args,parX,list(plane[plane_id]),subjects,syn_upper))
    Z=np.array(Z)
    return Z

def get_cost(args,parX,curparams,subjects,syn_upper):
    jXS = JointProbabilityMatrix(args.lenX+1,args.states)
    return cost_func_landscape(jXS,parX,curparams,subjects,syn_upper)

def plot_plane(title,Z,plot_func='contour',save_fig=False):
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
    elif plot_func=='3D':
        ax = Axes3D(fig)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.25, aspect=5)
        fig.savefig("../results/landscapes/"+title + '_3dsurface-2.png', dpi=300,
                    bbox_inches='tight', format='png')
        
        plt.xlabel('X')
        plt.ylabel('Y')
    # plot params as axis labels
    assert Z.shape[0] == Z.shape[1]
    plt.title(title)
    plt.show()

def cost_func_landscape(jXS,parX,free_params,subjects,syn_upper,extra_cost_rel_error=True,agnostic_about=None):
    if min(free_params) < -0.00001 or max(free_params) > 1.00001:
        # high or no cost for invalid parameter values
        return None
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

    assert np.isscalar(cost)
    assert np.isfinite(cost)
    return float(cost)
