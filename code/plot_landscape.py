import argparse
import numpy as np
from distutils.util import strtobool
from jointpdfpython3.JointProbabilityMatrix import JointProbabilityMatrix
from jointpdfpython3.params_matrix import matrix2params_incremental, params2matrix_incremental
from jointpdfpython3.measures import synergistic_entropy_upper_bound
from helpers.planes import random_orthogonal_unit_vectors,get_plane_points,get_plane_values,plot_plane

###############################################################
#                          MAIN
###############################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot computed Isyns of a random/PCA plane of parameters')

    # model parameters
    parser.add_argument('--steps', default=20,type=int, help='Number of points on each axis on 2D PCA grid')
    parser.add_argument('--mag', default=1,type=float, help='Magnitude of random orthogonal vectors')
    parser.add_argument('--plot_func', default='contour', help='Type of plot of 2D plane')
    parser.add_argument('--sysID', default=0,type=int, help='System ID of minimization path to plot')
    parser.add_argument('--runID', default=0,type=int, help='Run ID(s) of minimization path(s)')
    parser.add_argument('--save', default=False,type=lambda x: bool(strtobool(x)), help='Save run as json or not')
    parser.add_argument('--folder', default='landscapes', help='Folder in which data is saved')

    # system parameters
    parser.add_argument('--lenX', type=int, default=2,help='Number of input variables X')
    parser.add_argument('--states', type=int,default=2,help='Number of states for each random variable')
    parser.add_argument('--dist_type', default='dirichlet', help='Distribution type (iid, uniform, dirichlet)')
    parser.add_argument('--prev', default=None,type=lambda x: None if x == 'None' else x, help='Previous data to load in')
    parser.add_argument('--Sprev', default=False,type=lambda x: bool(strtobool(x)), help='Add SRV to X before constructing cost landscape')
    parser.add_argument('--second', default=True,type=lambda x: bool(strtobool(x)), help='Compare 2D with other input distribution')

    args = parser.parse_args()
    syn_uppers = []
    parXs = []
    lenparXS = ((args.states)**(args.lenX+1))-1
    lenparX = ((args.states)**(args.lenX))-1
    variablesX = np.arange(args.lenX)
    num_plots = 3
    
    # Initialize systems and middle point
    npsampling=True
    if npsampling:
        pars = [np.random.random(lenparX) for _ in range(num_plots)]
        pX = JointProbabilityMatrix(args.lenX,args.states)
        systems = []
        for p in pars:
            pnew = pX.copy()
            params2matrix_incremental(pnew,p)
            systems.append(pnew)
    else:
        systems = [JointProbabilityMatrix(args.lenX,args.states,joint_probs=args.dist_type) for _ in range(num_plots)]

    for s in systems:
        syn_uppers.append(synergistic_entropy_upper_bound(s,variables=variablesX))
        parXs.append(matrix2params_incremental(s)[:lenparX])

    mid = matrix2params_incremental(JointProbabilityMatrix(args.lenX+1,args.states,joint_probs='iid'))[:lenparXS-lenparX]
    print(len(mid))

    # plot random planes
    planes = []
    for n in range(num_plots):
        # get number of dimensions and generate two random orthogonal vectors
        v1, v2 = random_orthogonal_unit_vectors(lenparXS-lenparX)    

        # get plane of values 
        params_plane = get_plane_points(v1,v2,steps=args.steps,mag=args.mag,mid=mid)

        # get cost_values in plane for each system and plot them
        print("Hmax(X)",syn_uppers[n])
        cost_plane = get_plane_values(args,params_plane,syn_uppers[n],parXs[n])
        title = 'dist_type_'+args.dist_type+' states_'+str(args.states)+'num_'+str(n)
        plot_plane(title,cost_plane,plot_func=args.plot_func)

