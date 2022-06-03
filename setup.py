import re
import os
import sys
import os.path
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from distutils.util import strtobool
from cmath import nan
from operator import sub
from numba.cuda import test
import numpy as np
import warnings
import time
import itertools
import multiprocessing as mp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import os
import glob
import dit
from datetime import datetime

# jointpdf imports
import csv
import copy
from scipy.optimize import minimize
from compiler.ast import flatten  # note: deprecated in Python 3, in which case: find another flatten
from collections import Sequence
from abc import abstractmethod, ABCMeta  # for requiring certain methods to be overridden by subclasses
from numbers import Integral, Number
import pathos.multiprocessing as mp
from astroML.plotting import hist  # for Bayesian blocks: automatic determining of variable-size binning of data

from scipy.interpolate import InterpolatedUnivariateSpline
import numbers
from scipy.optimize import brentq

# syndisc imports
from numpy.linalg import svd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist 
from cvxopt import matrix, solvers
from scipy.stats import entropy
import pypoman as pm

from dit.pid.pid import BasePID
from dit.multivariate import coinformation
from dit.utils import flatten
from dit.utils import powerset
from itertools import combinations, islice, permutations
import networkx as nx