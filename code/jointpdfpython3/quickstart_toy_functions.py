from operator import sub
from numba.cuda import test
import numpy as np
import multiprocessing as mp

from scipy.optimize import minimize
from scipy.sparse import data
from JointProbabilityMatrix import JointProbabilityMatrix, ConditionalProbabilities
from params_matrix import matrix2params_incremental,params2matrix_incremental
from measures import append_variables_using_state_transitions_table,synergistic_entropy_upper_bound,synergistic_information_naive

import toy_functions

lenX=2
states=2

pXS = JointProbabilityMatrix(lenX,states)
try:
    toy_functions.append_synergistic_variables(pXS,1,subject_variables=[0,1])
except UserWarning as e:
    assert 'minimize() failed'

# print("WHAA")

# a = JointProbabilityMatrix(1,2)
# print(len(a))
# b = JointProbabilityMatrix(1,2)
# toy_functions.append_independent_variables(a,b)
# print(len(a))

# a = JointProbabilityMatrix(1,2)
# print(len(a))
# b = JointProbabilityMatrix(1,2)
# print(b.entropy())
# toy_functions.append_variables_with_target_mi(a,1,b.entropy()*0.5)
# print(len(a))
# print(a.mutual_information([0],[1]))