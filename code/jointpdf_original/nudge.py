__author__ = 'rquax'

import seaborn as sns
import pandas as pd
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import networkx as nx
from itertools import combinations, chain
from multiprocessing import Pool
import scipy.stats as st
import entropy_estimators as ee
import pathos.multiprocessing as mp
import jointpdf
import warnings
import scipy.spatial as ss
import itertools
import csv
import cPickle as pickle
import datetime
import statsmodels.nonparametric.kernel_density as snkde
from sklearn.neighbors import KernelDensity

sns.set_style('white')


### HELPER FUNCTIONS


def standardize_columns(rows):  # helper: used further down below
    return (np.array(rows) - np.mean(rows, axis=0)) / np.std(rows, axis=0)


def strnow():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def skew_dist(a, f=1.0, ntrials=1000):
    assert np.ndim(a) == 1
    a = np.array(a)
    skews = [st.skew(a[np.random.randint(len(a), size=int(len(a)*f))]) for _ in range(ntrials)]
    return skews


def skew_diff(a1, a2, f=1.0, alpha=0.05, ntrials=2000, relative=False):
    """
    If ci_high_diff < 0 then skew(a1) < skew(a2) 'with confidence'; if ci_low_diff > 0 then skew(a1) > skew(a2)
    'with confidence'; otherwise we cannot tell either way for sure.
    :param a1:
    :param a2:
    :param f: Fraction of samples to subsample. 1.0 = standard bootstrap.
    :param alpha:
    :param ntrials:
    :param relative:
    :return: (median_diff, ci_low_diff, ci_high_diff)
    """
    assert np.ndim(a1) == 1
    assert np.ndim(a2) == 1
    a1 = np.array(a1)
    a2 = np.array(a2)
    skews1 = skew_dist(a1, f=f, ntrials=ntrials)
    skews2 = skew_dist(a2, f=f, ntrials=ntrials)

    skew_diffs = np.subtract(skews1, skews2)

    if not relative:
        ci_low_diff = np.percentile(skew_diffs, alpha / 2.0 * 100.)
        ci_high_diff = np.percentile(skew_diffs, 100.0 - alpha / 2.0 * 100.)
        median_diff = np.median(skew_diffs)
    else:
        skew1 = st.skew(a1)  # just to make sure it is compatible with the skewness found by the caller
        # skew2 = st.skew(a2)

        ci_low_diff = np.percentile(skew_diffs, alpha / 2.0 * 100.) / np.abs(skew1)
        ci_high_diff = np.percentile(skew_diffs, 100.0 - alpha / 2.0 * 100.) / np.abs(skew1)
        median_diff = np.median(skew_diffs) / np.abs(skew1)

    return (median_diff, ci_low_diff, ci_high_diff)


### ACTUAL FUNCTIONS


def generate_bivariate_pdf(frac_mi, k, verbose=True, pY='unbiased'):
    pdf = jointpdf.JointProbabilityMatrix(1, k, pY)
    pdf.append_variables_with_target_mi(1, frac_mi * pdf.entropy())
    i = 0
    while abs(pdf.mutual_information([0], [1]) / pdf.entropy([0]) - frac_mi) / frac_mi > 0.1:
        if verbose:
            print 'note: MI/H=%s, will retry...' % ((pdf.mutual_information([0], [1]) / pdf.entropy([0]) - frac_mi) / frac_mi)
        pdf = jointpdf.JointProbabilityMatrix(1, k, 'unbiased')
        pdf.append_variables_with_target_mi(1, frac_mi * pdf.entropy())
        i += 1
        if i >= 10:
            raise UserWarning('I give up, no way to get b!')
    pdf.reorder_variables([1,0])
    if verbose:
        print pdf.entropy([0]), pdf.mutual_information([0], [1]), pdf.entropy([1])
    return pdf


def find_k_d_nearest_neighbors_up_to_m(z, k, m, kdtree, dmax=None):
    """
    This function will use the supplied KDTree to find (max) k**d unique nearest neighbors to datapoint `z`,
    namely by finding the `k` nearest neighbors, then the `k` nearest neighbors of those neighbors, etc.,
    up to distance `d`. It will stop at distance `d` where the number of unique neighbors `m` has been reached,
    but takes all nodes at this distance into account, unlike find_at_least_m_unique_nn().

    Note: it turns out from simple tests that find_at_least_m_unique_nn() seems to work better and makes a tighter
    cloud around a point compared to this function. Also it is faster. If as the last step the neighbor ids are not
    permuted randomly but chosen based on distance then it could potentially be better, but

    Note: due to overlap of neighborhoods the actual number of neighbors returned may be significantly less
    than k**d.
    """
    if dmax is None:
        dmax = len(kdtree.data)  # just some insane high number

    assert dmax > 0, 'dmax should be positive'

    dist_to_nbr = dict()

    # get indices of k nearest neighbors
    if np.isscalar(z):
        nbrs = kdtree.query(kdtree.data[z], k)[1]
    else:
        nbrs = kdtree.query(z, k)[1]

    dist_to_nbr[1] = nbrs  # store them at distance 1

    for d in xrange(2, dmax):
        if np.random.randint(10) == 0:
            assert len(np.unique(np.concatenate(dist_to_nbr.values()))) == len(np.concatenate(dist_to_nbr.values())), \
                'dict should already be unique by construction'

        if sum(map(len, dist_to_nbr.values())) >= m:
            break
        else:
            dist_to_nbr[d] = np.array([], dtype=np.int)
            for n in dist_to_nbr[d-1]:
                # note: this does not mean that in total these neighbors are unique, they could have still occurred
                # already at a lower distance, which I only account for at the end
                dist_to_nbr[d] = np.concatenate([dist_to_nbr[d], kdtree.query(kdtree.data[n], k)[1]])
            dist_to_nbr[d] = np.unique(dist_to_nbr[d])
            # make sure now that these neighbors do not already occur in lower distances, so that entire dict is unique
            dist_to_nbr[d] = np.setdiff1d(dist_to_nbr[d], np.concatenate([dist_to_nbr[d2] for d2 in xrange(1, d)]))

    dmax = max(dist_to_nbr.keys())

    nbrs = np.concatenate([np.random.permutation(dist_to_nbr[d2]) for d2 in dist_to_nbr.iterkeys()])
    if len(nbrs) >= m:
        nbrs = nbrs[:m]
    else:
        warnings.warn('find_k_d_nearest_neighbors_up_to_m: could not find `m` neighbors, only found (%s), '
                      'so perhaps the data manifold forms a sphere or something?' % m)

    return nbrs


def find_k_d_nearest_neighbors(z, k, d, kdtree):
    """
    This function will use the supplied KDTree to find (max) k**d unique nearest neighbors to datapoint `z`,
    namely by finding the `k` nearest neighbors, then the `k` nearest neighbors of those neighbors, etc.,
    up to distance `d`.

    Note: due to overlap of neighborhoods the actual number of neighbors returned may be significantly less
    than k**d.

    Note: probably you want find_at_least_m_unique_nn() if you intend to infer causality.
    """
    if d > 0:
        if np.isscalar(z):
            nbrs = kdtree.query(kdtree.data[z], k)[1]
        else:
            nbrs = kdtree.query(z, k)[1]

        if d > 1:
            more_nbrs = [find_k_d_nearest_neighbors(nbr, k, d - 1, kdtree) for nbr in nbrs]

            nbrs = list(nbrs) + list(chain.from_iterable(more_nbrs))

        return list(set(nbrs))
    else:
        return []


def find_at_least_m_unique_nn(z, k, m, kdtree, verbose=False):
    """
    Same as previous function but now surely return at least `m` neighbors
    (unless the neighborhoods form a loop/sphere).
    """
    if np.isscalar(z):
        nbrs = zip(*reversed(kdtree.query(kdtree.data[z], k)))
        zdata = kdtree.data[z]
    else:
        nbrs = zip(*reversed(kdtree.query(z, k)))
        zdata = z

    assert len(nbrs) == k, 'len(nbrs)=%s' % len(nbrs)

    uniquify_nbrs = True

    # note: `nbrs` is now a list of (zix, distance) tuples

    # note: the values are distances but they are kind of meaningless, as they are
    # distances not necessarily to `z` but potentially to any d-distance neighbor of `z`.
    # The reason for a dictionary is just to get unique values, at least for now.
    nbrs_dict = dict(nbrs)

    while len(nbrs_dict) < m:
        if len(nbrs) == 0:
            if verbose:
                warnings.warn('neighborhood forms a clique (isolated island), cannot get more '
                              'than %s neighbors, you may want to increase k' % len(nbrs_dict))
            break

        new_nbrs = np.array(np.transpose(nbrs)[0], dtype=np.int)  # unique new neighbor indices
        # new_nbrs = np.random.permutation(new_nbrs)  # kill any potential structure in the selection process
        nbrs = []  # will be filled with new neighbor indices

        for nbrix in new_nbrs:
            # get nn's for each newly added neighbor
            # note: `nbrs` is now a list of (zix, distance) tuples
            nbrs_i = kdtree.query(kdtree.data[nbrix], k)[1]
            if uniquify_nbrs:  # hopefully faster this way
                nbrs_i = [(ix, np.linalg.norm(kdtree.data[ix] - zdata)) for ix in nbrs_i if not nbrs_dict.has_key(ix)]
            nbrs = nbrs + nbrs_i

            nbrs_dict.update(nbrs_i)

            # instead of once adding all in `nbrs` in one go, and then checking for size,
            # doing it this way gives a closer count to `m` since the total number of
            # neighbors at distance d will grow exponentially, so the overshoot may be
            # large if I first collect all neighbors at distance d and then check for size.
            if len(nbrs_dict) >= m:
                break

        nbrs = sorted(nbrs, key=lambda x: x[1])  # sort on ascending distance, so that closest nbrs come first, next

    # return nbrs_dict.keys()

    # make sure only (max) `m` NEAREST neighbors are returned
    nbrs = np.array(np.transpose(sorted(nbrs_dict.items(), key=lambda x: x[1]))[0], dtype=np.int)
    nbrs = nbrs[:min(len(nbrs), m)]

    return nbrs


def find_m_unique_random_nn(z, m, kdtree, verbose=False):
    """
    Same as find_at_least_m_unique_nn but instead of neighboring z's this function finds simply random values,
    for the purpose of generating a null-hypothesis to compare the localized version against.
    """
    # if np.isscalar(z):
    #     nbrs = zip(*reversed(kdtree.query(kdtree.data[z], k)))
    # else:
    #     nbrs = zip(*reversed(kdtree.query(z, k)))

    # note: distances don't matter anyway, so leave them out
    # nbrs = zip(*[np.random.choice(len(kdtree.data), size=m, replace=False), np.random.random(size=m)])
    nbrs = np.random.choice(len(kdtree.data), size=m, replace=False)

    return nbrs

    # assert len(nbrs) == k, 'len(nbrs)=%s' % len(nbrs)
    #
    # uniquify_nbrs = True
    #
    # # note: `nbrs` is now a list of (zix, distance) tuples
    #
    # # note: the values are distances but they are kind of meaningless, as they are
    # # distances not necessarily to `z` but potentially to any d-distance neighbor of `z`.
    # # The reason for a dictionary is just to get unique values, at least for now.
    # nbrs_dict = dict(nbrs)
    #
    # while len(nbrs_dict) < m:
    #     if len(nbrs) == 0:
    #         if verbose:
    #             warnings.warn('neighborhood forms a clique (isolated island), cannot get more '
    #                           'than %s neighbors, you may want to increase k' % len(nbrs_dict))
    #         break
    #
    #     new_nbrs = np.array(np.transpose(nbrs)[0], dtype=np.int)  # unique new neighbor indices
    #     nbrs = []  # will be filled with new neighbor indices
    #
    #     for nbrix in new_nbrs:
    #         # get nn's for each newly added neighbor
    #         # note: `nbrs` is now a list of (zix, distance) tuples
    #         nbrs_i = zip(*reversed(kdtree.query(kdtree.data[nbrix], k)))
    #         if uniquify_nbrs:  # hopefully faster this way
    #             nbrs_i = [(ix, distance) for (ix, distance) in nbrs_i if not nbrs_dict.has_key(ix)]
    #         nbrs = nbrs + nbrs_i
    #
    #         nbrs_dict.update(nbrs_i)
    #
    #         # instead of once adding all in `nbrs` in one go, and then checking for size,
    #         # doing it this way gives a closer count to `m` since the total number of
    #         # neighbors at distance d will grow exponentially, so the overshoot may be
    #         # large if I first collect all neighbors at distance d and then check for size.
    #         if len(nbrs_dict) >= m:
    #             break
    #
    # return nbrs_dict.keys()


# not needed for anything at the moment:
def num_unique_nbrs_within_dist(k, d, kdtree, ntrials=100):
    numnbrs = [len(find_k_d_nearest_neighbors(np.random.randint(len(kdtree.data)), k, d, kdtree))
               for _ in xrange(ntrials)]
    return np.mean(numnbrs), np.std(numnbrs)


def determine_min_numpts_knn_mi(dataset_gauss, xix, yix, ntrials=500, verbose=True):
    """
    Find the minimal number of points to use to estimate MI using k-nn. This determines `mz` in below functions.

    Use either the returned `mz` directly or eyeball the returned results yourself.

    Todo: change the way the optimal point is chosen, I think can be lower because we do not care about the actual
    true value but about a value which does not vary so much and is not skewed (for the purpose of detecting whether
    I(X:Y|Z) is skewed).
    :param dataset_gauss02_XY_only:
    :param ntrials:
    :param verbose:
    :return: mz, numpts_list, means, sems, skewzs, skewps
    """
    numpts_list = map(int, np.linspace(50, 1000, 21))

    dataset_gauss02_XY_only = np.array(dataset_gauss)[:, [xix, yix]]
    # make format easier to compute MI
    dataset_gauss02_XY_only = np.transpose(dataset_gauss02_XY_only)
    dataset_gauss02_XY_only = np.reshape(dataset_gauss02_XY_only, np.shape(dataset_gauss02_XY_only) + (1,))

    time_before = time.time()
    mis_per_nps = [[ee.mi(dataset_gauss02_XY_only[0][list(set(map(int, subset)))],
                          dataset_gauss02_XY_only[1][list(set(map(int, subset)))], k=3)
                    for subset in np.random.randint(len(dataset_gauss02_XY_only[0]), size=(ntrials, nps))]
                   for nps in numpts_list]

    if verbose:
        print 'note: took %s seconds' % (time.time() - time_before)

    yerr = [np.std(mis) / np.sqrt(len(mis)) for mis in mis_per_nps]
    sems = yerr
    means = [np.mean(mis) for mis in mis_per_nps]

    skewzs = [st.skewtest(mis)[0] for mis in mis_per_nps]
    skewps = [st.skewtest(mis)[1] for mis in mis_per_nps]

    true_mi = np.mean([np.mean(mis) for mis in mis_per_nps[-3:]])
    upperbounds_true_mi = np.greater_equal(np.add(means, np.multiply(2., yerr)), true_mi)
    lowerbounds_true_mi = np.less_equal(np.subtract(means, np.multiply(2., yerr)), true_mi)
    contains_true_mi = np.logical_and(upperbounds_true_mi, lowerbounds_true_mi)
    # print contains_true_mi
    index_last_false = (len(contains_true_mi) - 1) - list(reversed(contains_true_mi)).index(False)

    #     mz = numpts_list[min(len(numpts_list) - 1, index_last_false+1)]
    # note: technically we should pick the index for which 5% is actually False in contains_true_mi,
    # since under the null hypothesis that is expected. As hack I just take
    mz = numpts_list[min(len(numpts_list) - 1, index_last_false)]

    print 'note: automatically selected mz=%s' % mz

    if mz < 100 or mz > 1000:
        warnings.warn('Please check if mz makes sense! Seems that it may actually NOT make sense.')

    return mz, numpts_list, means, sems, skewzs, skewps


def conditional_mi_knn(z, k, m, kdtree_Z_only, dataset_XY_only, kxy=3):
    # note: this shape for XY makes it faster to compute MI for a subset
    # note: the order of the XY points should match that of the Z points used to build `kdtree_Z_only`
    assert np.shape(dataset_XY_only) == (2, len(kdtree_Z_only.data), 1), 'assumed shape for XY unmatched'
    assert m >= 10, 'm=%s should preferably be even >=100' % m

    nbrs = find_at_least_m_unique_nn(z, k, m, kdtree=kdtree_Z_only)

    return ee.mi(dataset_XY_only[0][nbrs], dataset_XY_only[1][nbrs], k=kxy)


def conditional_mi_knn_null(z, k, m, kdtree_Z_only, dataset_XY_only, kxy=3):
    """
    Same as conditional_mi_knn() but now the neighbors of `z` are actually chosen simply at random, not based on `z`.
    :param z:
    :param k:
    :param m:
    :param kdtree_Z_only:
    :param dataset_XY_only:
    :param kxy:
    :return:
    """
    # note: this shape for XY makes it faster to compute MI for a subset
    # note: the order of the XY points should match that of the Z points used to build `kdtree_Z_only`
    assert np.shape(dataset_XY_only) == (2, len(kdtree_Z_only.data), 1), 'assumed shape for XY unmatched'
    assert m >= 10, 'm=%s should preferably be even >=100' % m

    nbrs = find_m_unique_random_nn(z, m, kdtree=kdtree_Z_only)

    return ee.mi(dataset_XY_only[0][nbrs], dataset_XY_only[1][nbrs], k=kxy)


# TODO: instead of using once a knn MI with `n` to see if MI diff is > 0,
# maybe do a smaller `n` but do three times and then estimate
# whether 0 is inside or below the 95% CI?

def potential_Zs(dataset_gauss, setZ=None, xix=None, yix=None, n=30000, method='backward', tol=0.2, tol_abs=0.0,
                 min_num_Zs=1, max_num_Zs=5, max_num_setsZ=100, min_num_indivZs=3, max_num_indivZs=15, nprocs=1):
    """
    This will first transform `dataset_gauss` into a more amenable format and then use that to call
    potential_Zs_direct(), so a wrapper. If you need to call this function many times then maybe you should do
    this transformation yourself once and then not use this wrapper.

    :return: best_Zs
    :rtype: list of list
    """

    if setZ is None and xix is None and yix is None:
        setZ = range(np.shape(dataset_gauss)[1] - 2)
        numZ = len(setZ)
        xix = numZ
        yix = numZ + 1
    elif xix is None and yix is None:
        # there should be two left preferably, otherwise I just take the lowest numbered two
        leftover = np.setdiff1d(range(np.shape(dataset_gauss)[1]), setZ)
        xix, yix = leftover[:2]
        if len(leftover) > 2:
            warnings.warn('the complement of the given setZ in the columns is %s which leaves me freedom in '
                          'picking X and Y. I simply pick the lowest two possible ones.' % str(leftover))
        numZ = len(setZ)
    elif setZ is None:
        assert not xix is None and not yix is None, 'I don\'t understand what you want'
        setZ = np.setdiff1d(range(np.shape(dataset_gauss)[1]), [xix, yix])

    dataset_gauss_XY_only = np.array(dataset_gauss)[:, [xix, yix]]
    # make format easier to compute MI
    dataset_gauss_XY_only = np.transpose(dataset_gauss_XY_only)
    dataset_gauss_XY_only = np.reshape(dataset_gauss_XY_only, np.shape(dataset_gauss_XY_only) + (1,))

    dataset_gauss_Z_only = np.array(dataset_gauss)[:, setZ]

    bestzs = potential_Zs_direct(dataset_gauss_Z_only, dataset_gauss_XY_only, n, method, tol, tol_abs,
                                 min_num_Zs=min_num_Zs, max_num_Zs=max_num_Zs, max_num_setsZ=max_num_setsZ,
                                 min_num_indivZs=min_num_indivZs, max_num_indivZs=max_num_indivZs, nprocs=nprocs)

    # note: the integers in `bestzs` are 0,numZ-1 and need to be converted to numbers in `setZ`
    bestzs = [[setZ[int(zix)] for zix in zs] for zs in bestzs]
    return bestzs


def potential_Zs_direct(dataset_gauss_Z_only, dataset_gauss_XY_only, n=30000, method='backward', tol=0.2, tol_abs=0.0,
                        min_num_Zs=1, max_num_Zs=5, max_num_setsZ=100, min_num_indivZs=3, max_num_indivZs=15, nprocs=1,
                        uniform_num_per_len=5):
    """
    Find a sequence of subsets of Z indices (in range(len(dataset_gauss_Z_only))) to be tried
    for the skewness calculations of the cond. MI distribution, in order.

    :param max_num_indivZs: maximum number of individual Z_i's to be selected for creating sets of Zs.
    Based on: sum([scipy.special.binom(15, k) for k in range(3, 7)]) ~= 1e04 which seems already a lot
    to do computations for.
    """
    assert len(dataset_gauss_Z_only) == len(dataset_gauss_XY_only[0])
    assert len(dataset_gauss_Z_only) == len(dataset_gauss_XY_only[1])

    if n is None:
        dataset_gauss01_Z_only = np.array(dataset_gauss_Z_only)
        dataset_gauss01_XY_only = np.array(dataset_gauss_XY_only)
    elif n < len(dataset_gauss_Z_only):
        indices = np.random.choice(range(len(dataset_gauss_Z_only)), n, replace=False)
        dataset_gauss01_Z_only = np.array(dataset_gauss_Z_only)[indices]
        dataset_gauss01_XY_only = np.array([np.array(dataset_gauss_XY_only[0])[indices],
                                            np.array(dataset_gauss_XY_only[1])[indices]])
    else:
        dataset_gauss01_Z_only = np.array(dataset_gauss_Z_only)
        dataset_gauss01_XY_only = np.array(dataset_gauss_XY_only)

    mi_xy = ee.mi(dataset_gauss01_XY_only[0], dataset_gauss01_XY_only[1])
    # print('note: mi_xy = %s' % mi_xy)

    dataset_gauss01_Z_only_transpose = np.transpose(dataset_gauss01_Z_only)
    # how ee.mi wants it:
    dataset_gauss01_Z_only_transpose = np.reshape(dataset_gauss01_Z_only_transpose,
                                                  np.shape(dataset_gauss01_Z_only_transpose) + (1,))

    if nprocs == 1:
        mis_zx = np.array([ee.mi(dataset_gauss01_XY_only[0], dataset_gauss01_Z_only_transpose[zix])
                           for zix in range(len(dataset_gauss01_Z_only_transpose))], dtype=np.float64)
        mis_zy = np.array([ee.mi(dataset_gauss01_XY_only[1], dataset_gauss01_Z_only_transpose[zix])
                           for zix in range(len(dataset_gauss01_Z_only_transpose))], dtype=np.float64)
    else:
        def worker_mis_zx(zix):
            try:
                return ee.mi(dataset_gauss01_XY_only[0], dataset_gauss01_Z_only_transpose[zix])
            except KeyboardInterrupt as e:
                # solution to interrupt problem as per bottom of
                # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                pass
        def worker_mis_zy(zix):
            try:
                return ee.mi(dataset_gauss01_XY_only[1], dataset_gauss01_Z_only_transpose[zix])
            except KeyboardInterrupt as e:
                # solution to interrupt problem as per bottom of
                # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                pass

        pool = mp.Pool(nprocs)
        try:
            mis_zx = pool.map(worker_mis_zx, range(len(dataset_gauss01_Z_only_transpose)))
            mis_zy = pool.map(worker_mis_zy, range(len(dataset_gauss01_Z_only_transpose)))
        except KeyboardInterrupt as e:
            pool.close()
            pool.terminate()

            return None
        else:
            pool.close()
            pool.terminate()

        mis_zx = np.array(mis_zx, dtype=np.float64)
        mis_zy = np.array(mis_zy, dtype=np.float64)

    # print('debug: mis_zx = %s' % mis_zx)
    # print('debug: mis_zy = %s' % mis_zy)
    # print('debug: mis_zx - mis_zy = %s' % (mis_zx - mis_zy))

    mi_diffs = (mis_zx - mis_zy)
    mi_diff_tuples = sorted(zip(range(len(mi_diffs)), mi_diffs), key=lambda x: x[-1], reverse=True)

    # select only Z's which have I(Z:X) > I(Z:Y) as a heuristic
    try:
        max_pos_diff = np.max(mis_zx - mis_zy)
        num_x_heavy_zs = list(np.greater(np.array(mi_diff_tuples)[:, 1], 0.0 - max(tol * max_pos_diff, tol_abs))).index(False)
    except ValueError as e:
        num_x_heavy_zs = len(mi_diff_tuples)
    mi_diff_tuples = mi_diff_tuples[:max(min(num_x_heavy_zs, max_num_indivZs), min_num_indivZs)]

    # print 'debug: mi_diff_tuples = %s' % mi_diff_tuples

    if len(mi_diff_tuples) == 0:
        warnings.warn('There were no Z\'s which correlate stronger with X than with Y, so I return empty Z.')

        return []

    if method in ('linear', 'forward'):
        mi_ixs = np.array(mi_diff_tuples)[:, 0]
        #         mi_diffs = np.array(mi_diff_tuples)[:,1]

        return [mi_ixs[:l] for l in range(1, len(mi_ixs) + 1) if min_num_Zs <= len(mi_ixs[:l]) <= max_num_Zs]
    elif method in ('backward',):
        mi_ixs = np.array(mi_diff_tuples)[:, 0]
        #         mi_diffs = np.array(mi_diff_tuples)[:,1]

        return [mi_ixs[:l] for l in reversed(range(1, len(mi_ixs) + 1)) if min_num_Zs <= len(mi_ixs[:l]) <= max_num_Zs]
    elif method in ('bruteforce',):
        mi_ixs = np.array(mi_diff_tuples)[:, 0]
        #         mi_diffs = np.array(mi_diff_tuples)[:,1]

        # NOTE: at the moment the algorithm below leads trivially to a sorted from largest to smallest sets,
        # mostly. Should somehow penalize very large sets?

        all_sets = []
        for nz in range(min_num_Zs, max_num_Zs + 1):
            all_sets.extend(itertools.combinations(mi_ixs, nz))
        all_sets_mis = [sum([mi_diffs[int(zi)] for zi in zsi]) for zsi in all_sets]
        all_sets_mi_tuples = zip(all_sets, all_sets_mis)
        all_sets_mi_tuples = sorted(all_sets_mi_tuples, key=lambda x: x[-1], reverse=True)
        all_sets_mi_tuples = all_sets_mi_tuples[:int(max_num_setsZ)]
        all_sets = map(lambda x: map(int, x[0]), all_sets_mi_tuples)  # get only the sets of Zs
        return all_sets
    elif method in ('uniform',):
        mi_ixs = np.array(mi_diff_tuples)[:, 0]
        #         mi_diffs = np.array(mi_diff_tuples)[:,1]

        # NOTE: at the moment the algorithm below leads trivially to a sorted from largest to smallest sets,
        # mostly. Should somehow penalize very large sets?

        all_sets = []
        lens = range(min_num_Zs, max_num_Zs + 1)
        for nz in lens:
            all_sets.append(itertools.combinations(mi_ixs, nz))
            all_sets_mis = [sum([mi_diffs[int(zi)] for zi in zsi]) for zsi in all_sets[-1]]
            all_sets_mi_tuples = zip(all_sets[-1], all_sets_mis)
            all_sets_mi_tuples = sorted(all_sets_mi_tuples, key=lambda x: x[-1], reverse=True)
            all_sets_mi_tuples = all_sets_mi_tuples[:int(max_num_setsZ)]
            # get only the sets of Zs:
            all_sets[-1] = map(lambda x: map(int, x[0]),
                               all_sets_mi_tuples)[:min(len(all_sets_mi_tuples), uniform_num_per_len)]
        all_sets_flat = []
        for alls in all_sets:
            all_sets_flat.extend(alls)
        return all_sets_flat
    else:
        raise NotImplementedError('unknown method=%s' % method)


def calculate_condmi_knn(dataset, sigma=0, xix=None, yix=None, setZ=None, kz=10, mz=400, kxy=3,
                         numsamples=None, nprocs=1, numsamples_z=4000, sample_Z_uniformly=False, bw_kde_Z=None,
                         verbose=False):
    """
    This function does all of the above in sequence, eventually spitting out the list of
    conditional MIs estimated.

    :param dataset: A continuous dataset of shape (#samples, #feats), or a discrete dataset
    from a JointProbabilityMatrix object and sigma > 0 to make it a continuous dataset.
    :param numsamples_z: number of cond. MI values will be estimated around a ramdomly chosen
    Z=z datapoint.
    :param mz: each cond. MI is estimated using `mz` samples around this point
    """

    if not numsamples is None:
        if sigma > 0:
            pdf_temp = jointpdf.JointProbabilityMatrix(1, 3)  # dummy
            # todo: there is probably a more efficient way of selecting `numsamples` random rows...
            dataset_gauss = pdf_temp.generate_samples_mixed_gaussian(np.random.permutation(dataset)[:numsamples], sigma)
        else:
            dataset_gauss = np.random.permutation(dataset)[:numsamples]
    elif sigma > 0:
        pdf_temp = jointpdf.JointProbabilityMatrix(1, 3)  # dummy
        dataset_gauss = np.array(pdf_temp.generate_samples_mixed_gaussian(dataset, sigma))
    else:
        dataset_gauss = np.array(dataset)  # dataset is already continuous or assumed so

    # make the `dataset_gauss` in column order Z1,...,1Zn,X,Y
    if not (setZ is None and xix is None and yix is None):
        assert not np.isscalar(setZ), 'at least for for now'
        assert np.ndim(setZ) == 1, 'at least for now'

        setZ = map(int, setZ)

        numZ = len(setZ)
        setNonZ = np.setdiff1d(range(np.shape(dataset_gauss)[1]), setZ)
        assert len(setNonZ) + len(setZ) == np.shape(dataset_gauss)[1], 'you seem to have supplied non-existing Z\'s'

        if xix is None:
            xix = int(setNonZ[0])
        if yix is None:
            yix = int(setNonZ[1])

        # dataset_gauss = dataset_gauss[:, np.array(list(setZ) + [xix, yix])]

        numZ = len(setZ)

        if not setZ is None:
            if xix in setZ:
                warnings.warn('X should not be in Z (or are you testing something?)')
            if yix in setZ:
                warnings.warn('Y should not be in Z (or are you testing something?)')
    else:
        setZ = range(np.shape(dataset_gauss)[1] - 2)
        xix = np.shape(dataset_gauss)[1] - 2
        yix = np.shape(dataset_gauss)[1] - 1
        numZ = len(setZ)

        if not setZ is None:
            if xix in setZ:
                warnings.warn('X should not be in Z (or are you testing something?)')
            if yix in setZ:
                warnings.warn('Y should not be in Z (or are you testing something?)')

        pass  # assuming that the dataset was already supplied in column order Z1,...,1Zn,X,Y

    if mz is None:  # automatically select a `mz`
        mz = determine_min_numpts_knn_mi(dataset_gauss, xix, yix, verbose=True)

    if mz > 0.2 * len(dataset):
        warnings.warn('mz=%s is more than 20percent of the number of data samples, that is quite large!' % mz)

    # take the subset of columns from the data which pertain to Z (currently: not X or Y)
    dataset_gauss_Z_only = np.array(dataset_gauss)[:, setZ]

    # KDTree makes it fast to find neighbors in the Z space
    kdtree_Z_only = ss.cKDTree(dataset_gauss_Z_only)

    # now take out the X,Y dataset
    dataset_gauss_XY_only = np.array(dataset_gauss)[:, [xix, yix]]
    # make format (shape) easier to compute MI using the k-nn algorithm
    dataset_gauss_XY_only = np.transpose(dataset_gauss_XY_only)
    dataset_gauss_XY_only = np.reshape(dataset_gauss_XY_only,
                                       np.shape(dataset_gauss_XY_only) + (1,))  # this is how ee.mi() wants it...

    if not sample_Z_uniformly:
        def worker_condmi(i):
            try:
                # select random Z'=z and estimate I(X':Y'|z)
                return conditional_mi_knn(np.random.randint(len(dataset_gauss_XY_only[0])),
                                          kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
            except KeyboardInterrupt as e:
                pass

        if nprocs == 1:
            condmi_knn = [conditional_mi_knn(np.random.randint(len(dataset_gauss_XY_only[0])),
                                             kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
                          for _ in xrange(numsamples_z)]
        else:
            # use multiple processors, most likely a very good idea!
            pool = mp.Pool(nprocs)
            try:
                condmi_knn = pool.map(worker_condmi, xrange(numsamples_z))
            except KeyboardInterrupt as e:
                pool.close()
                pool.terminate()

                return None
            else:
                pool.close()
                pool.terminate()
    else:
        if verbose:
            time_before = time.time()
            print 'debug: will start doing KDE for p(Z)...'

        if bw_kde_Z is None:
            kde_Z = snkde.KDEMultivariate(data=dataset_gauss_Z_only, var_type='c' * np.shape(dataset_gauss_Z_only)[1])
            bw_kde_Z = np.mean(kde_Z.bw)
            # del kde_Z  # do not actually use it because retrieval is slow

        kde_Z = KernelDensity(bandwidth=bw_kde_Z, rtol=1e-04)
        kde_Z.fit(dataset_gauss_Z_only)

        if nprocs == 1:
            # pZs = np.array([kde_Z.pdf(z) for z in dataset_gauss_Z_only])  # note: these do not necessarily sum up to one
            pZs = np.exp(kde_Z.score_samples(dataset_gauss_Z_only))  # note: these do not necessarily sum up to one
        else:
            def worker_pz(zix):
                try:
                    return np.exp(kde_Z.score_samples([dataset_gauss_Z_only[zix]])[0])
                except KeyboardInterrupt as e:
                    pass

            pool = mp.Pool(nprocs)
            try:
                pZs = pool.map(worker_pz, range(len(dataset_gauss_Z_only)))
            except:
                pool.close()
                pool.terminate()

                return None
            else:
                pool.close()
                pool.terminate()

            pZs = np.array(pZs)

        if np.any(pZs == 0.0):
            warnings.warn('some pZs are zero, not good! I\'ll fix it crudely')
            mean_pZs = np.mean(pZs)
            if mean_pZs > 0.0:
                pZs += mean_pZs * 0.01
            else:
                pZs = np.ones_like(pZs)
        lZs = 1.0 / pZs  # likelihood for sampling in order to become uniform in Z space: 2x more samples, 2x unlikely
        lZs /= np.sum(lZs)  # normalize for np.random.choice()

        if verbose:
            print 'debug: fitting KDE for p(Z) and calculating all probs took %s seconds' % (time.time() - time_before)

        z_choices = np.random.choice(len(dataset_gauss_Z_only), p=lZs, replace=True, size=numsamples_z)

        def worker_condmi(zc):
            try:
                return conditional_mi_knn(zc, kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
            except KeyboardInterrupt as e:
                pass

        if nprocs == 1:
            condmi_knn = [worker_condmi(zc) for zc in z_choices]
        else:
            # use multiple processors, most likely a very good idea!
            pool = mp.Pool(nprocs)
            try:
                condmi_knn = pool.map(worker_condmi, z_choices)
            except KeyboardInterrupt as e:
                pool.close()
                pool.terminate()

                return None
            else:
                pool.close()
                pool.terminate()

        # del kde_Z  # don't know if this is memory intensive but should not be used anymore accidentally anyway
        del pZs, lZs, z_choices

    return condmi_knn


def calculate_condmi_knn_null(dataset, sigma=0, xix=None, yix=None, setZ=None, kz=10, mz=400, kxy=3,
                              numsamples=None, nprocs=1, numsamples_z=4000, sample_Z_uniformly=False, bw_kde_Z=None,
                              verbose=False):
    """
    Calculates the null-hypothesis distribution of MI, where localizing on values for variables `setZ` does
    not actually change the distribution, especially regarding the skewness. In other words, this function simply
    takes random values for `setZ` instead of neighboring values around one chosen value each time.

    :param dataset: A continuous dataset of shape (#samples, #feats), or a discrete dataset
    from a JointProbabilityMatrix object and sigma > 0 to make it a continuous dataset.
    """

    if not numsamples is None:
        if sigma > 0:
            pdf_temp = jointpdf.JointProbabilityMatrix(1, 3)  # dummy
            # todo: there is probably a more efficient way of selecting `numsamples` random rows...
            dataset_gauss = pdf_temp.generate_samples_mixed_gaussian(np.random.permutation(dataset)[:numsamples], sigma)
        else:
            dataset_gauss = np.random.permutation(dataset)[:numsamples]
    elif sigma > 0:
        pdf_temp = jointpdf.JointProbabilityMatrix(1, 3)  # dummy
        dataset_gauss = np.array(pdf_temp.generate_samples_mixed_gaussian(dataset, sigma))
    else:
        dataset_gauss = np.array(dataset)  # dataset is already continuous or assumed so

    # make the `dataset_gauss` in column order Z1,...,1Zn,X,Y
    if not (setZ is None and xix is None and yix is None):
        assert not np.isscalar(setZ), 'at least for for now'
        assert np.ndim(setZ) == 1, 'at least for now'

        setZ = map(int, setZ)

        numZ = len(setZ)
        setNonZ = np.setdiff1d(range(np.shape(dataset_gauss)[1]), setZ)
        assert len(setNonZ) + len(setZ) == np.shape(dataset_gauss)[1], 'you seem to have supplied non-existing Z\'s'

        if xix is None:
            xix = int(setNonZ[0])
        if yix is None:
            yix = int(setNonZ[1])

        # dataset_gauss = dataset_gauss[:, np.array(list(setZ) + [xix, yix])]

        numZ = len(setZ)

        if not setZ is None:
            if xix in setZ:
                warnings.warn('X should not be in Z (or are you testing something?)')
            if yix in setZ:
                warnings.warn('Y should not be in Z (or are you testing something?)')
    else:
        setZ = range(np.shape(dataset_gauss)[1] - 2)
        xix = np.shape(dataset_gauss)[1] - 2
        yix = np.shape(dataset_gauss)[1] - 1
        numZ = len(setZ)

        if not setZ is None:
            if xix in setZ:
                warnings.warn('X should not be in Z (or are you testing something?)')
            if yix in setZ:
                warnings.warn('Y should not be in Z (or are you testing something?)')

        pass  # assuming that the dataset was already supplied in column order Z1,...,1Zn,X,Y

    if mz is None:  # automatically select a `mz`
        mz = determine_min_numpts_knn_mi(dataset_gauss, xix, yix, verbose=True)

    if mz > 0.1 * len(dataset):
        warnings.warn('mz=%s is more than 10 percent of the number of data samples, that is quite large!' % mz)

    # take the subset of columns from the data which pertain to Z (currently: not X or Y)
    dataset_gauss_Z_only = np.array(dataset_gauss)[:, setZ]

    # KDTree makes it fast to find neighbors in the Z space
    kdtree_Z_only = ss.cKDTree(dataset_gauss_Z_only)

    # now take out the X,Y dataset
    dataset_gauss_XY_only = np.array(dataset_gauss)[:, [xix, yix]]
    # make format (shape) easier to compute MI using the k-nn algorithm
    dataset_gauss_XY_only = np.transpose(dataset_gauss_XY_only)
    dataset_gauss_XY_only = np.reshape(dataset_gauss_XY_only,
                                       np.shape(dataset_gauss_XY_only) + (1,))  # this is how ee.mi() wants it...

    if not sample_Z_uniformly:
        if nprocs == 1:
            condmi_knn = [conditional_mi_knn_null(np.random.randint(len(dataset_gauss_XY_only[0])),
                                                  kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
                          for _ in xrange(numsamples_z)]
        else:
            def worker_condmi(i):
                try:
                    # select random Z'=z and estimate I(X':Y'|z)
                    return conditional_mi_knn_null(np.random.randint(len(dataset_gauss_XY_only[0])),
                                                   kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
                except KeyboardInterrupt as e:
                    # solution to interrupt problem as per bottom of
                    # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                    pass

            # use multiple processors, most likely a very good idea!
            pool = mp.Pool(nprocs)
            try:
                condmi_knn = pool.map(worker_condmi, xrange(numsamples_z))
            except KeyboardInterrupt as e:
                condmi_knn = None  # interrupted
            pool.close()
            pool.terminate()
    else:
        # kde_Z = snkde.KDEMultivariate(data=dataset_gauss_Z_only, var_type='c' * np.shape(dataset_gauss_Z_only)[1])
        #
        # pZs = np.array([kde_Z.pdf(z) for z in dataset_gauss_Z_only])  # note: these do not necessarily sum up to one

        if verbose:
            time_before = time.time()
            print 'debug: will start doing KDE for p(Z) (null)...'

        if bw_kde_Z is None:
            kde_Z = snkde.KDEMultivariate(data=dataset_gauss_Z_only, var_type='c' * np.shape(dataset_gauss_Z_only)[1])
            bw_kde_Z = np.mean(kde_Z.bw)
            # del kde_Z  # do not actually use it because retrieval is slower so free memory and prevent use

        kde_Z = KernelDensity(bandwidth=bw_kde_Z, rtol=1e-04)
        kde_Z.fit(dataset_gauss_Z_only)

        if nprocs == 1:
            # pZs = np.array([kde_Z.pdf(z) for z in dataset_gauss_Z_only])  # note: these do not necessarily sum up to one
            pZs = np.exp(kde_Z.score_samples(dataset_gauss_Z_only))  # note: these do not necessarily sum up to one
        else:
            def worker_pz(zix):
                try:
                    return np.exp(kde_Z.score_samples([dataset_gauss_Z_only[zix]])[0])
                except KeyboardInterrupt as e:
                    # solution to interrupt problem as per bottom of
                    # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                    pass

            pool = mp.Pool(nprocs)
            try:
                pZs = pool.map(worker_pz, range(len(dataset_gauss_Z_only)))
                pZs = np.array(pZs)
            except KeyboardInterrupt as e:
                # solution to interrupt problem as per bottom of
                # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                pZs = None  # interrupted
            pool.close()
            pool.terminate()

        if verbose:
            print 'debug: fitting KDE for p(Z) and calculating all probs took %s seconds (null)' % (time.time() - time_before)

        if np.any(pZs == 0.0):
            warnings.warn('some pZs are zero, not good! I\'ll fix it crudely')
            mean_pZs = np.mean(pZs)
            if mean_pZs > 0.0:
                pZs += mean_pZs * 0.01
            else:
                pZs = np.ones_like(pZs)
        lZs = 1.0 / pZs  # likelihood for sampling in order to become uniform in Z space: 2x more samples, 2x unlikely
        lZs /= np.sum(lZs)  # normalize for np.random.choice()

        z_choices = np.random.choice(len(dataset_gauss_Z_only), p=lZs, replace=True, size=numsamples_z)

        def worker_condmi(zc):
            try:
                return conditional_mi_knn_null(zc, kz, mz, kdtree_Z_only, dataset_gauss_XY_only, kxy=kxy)
            except KeyboardInterrupt as e:
                # solution to interrupt problem as per bottom of
                # http://bryceboe.com/2010/08/26/python-multiprocessing-and-keyboardinterrupt/
                pass

        if nprocs == 1:
            condmi_knn = [worker_condmi(zc) for zc in z_choices]
        else:
            # use multiple processors, most likely a very good idea!
            pool = mp.Pool(nprocs)

            try:
                condmi_knn = pool.map(worker_condmi, z_choices)
            except KeyboardInterrupt as e:
                condmi_knn = None
            pool.close()
            pool.terminate()

        # del kde_Z  # don't know if this is memory intensive but should not be used anymore accidentally anyway
        del pZs, lZs, z_choices

    return condmi_knn


def generate_causal_model(b=0.75, bZ=0.15, numZ=5, bW=0.15, numW=0, k=5, num_pdf_trials=20,
                          tol_rel=0.1, verbose=True):
    """
    Generate a jointpdf object (Z_1, ..., Z_n, X, Y) where X causes Y with relative strength `b` and
    the Z's together causally influence X with relative strength `bZ` (generally lower than `b` so
    that it is a 'nudge').

    If numW>0 then also some W_i's are appended which depend on Y but have no causal influence on anything.
    :rtype: jointpdf.JointProbabilityMatrix
    """

    pdf = None

    for attempt in range(num_pdf_trials):  # try generating a 'correct' pdf 10 times, should only take once
        # generate some X --> Y which satisfies the conditions (b)
        pdfXY = generate_bivariate_pdf(b, k=k, verbose=False)
        pdfY_X = pdfXY.conditional_probability_distributions([0], [1])
        pX = pdfXY[0]

        # generate the Z variables which will act on X
        # note: each Z_i is assumed independent of the other, makes it much faster to generate them
        # (because `numZ` times a low-dimensional optimization is much faster than once a huge-dimensional one)

        if bZ > 0:
            for misZ_trial in range(50):
                misZ = bZ * pX.entropy() / np.random.poisson(numZ, numZ)
                misZ = misZ / np.sum(misZ) * bZ * pX.entropy()

                if np.min(misZ) > 0:
                    break
        else:
            misZ = np.zeros(numZ)

        pdf = pX.copy()
        for zix in range(numZ):
            pdf2 = pX.copy()
            if misZ[zix] > 0:
                pdf2.append_variables_with_target_mi(1, misZ[zix], 'all')  # faster searching independent of other Z's
            else:
                if not bZ == 0:
                    warnings.warn('I am asked to add a Z_i with misZ[zix]=%s which makes little sense' % misZ[zix])
                pdf2.append_independent_variables(jointpdf.JointProbabilityMatrix(1, pdf2.numvalues))
            pZi_X = pdf2.conditional_probability_distributions([0], [1])
            pdf.append_variables_using_conditional_distributions(pZi_X, [0])  # add to X and other Z's
            if verbose:
                print 'note: appended variable Z%s, MI is %s (%s) (aimed at %s)' % (
                zix, pdf2.mutual_information([0], [1]), pdf.mutual_information([0], [zix + 1]), misZ[zix])

        pdf.reorder_variables(range(1, numZ + 1) + [0])
        if verbose:
            print 'note: I(Z0,...,Zn:X) = %s (aimed at %s)' % (
            pdf.mutual_information(range(numZ), [numZ]), bZ * pX.entropy())
        # print 'note: <I(Zi:X)> = %s (aimed at %s)' % (np.mean([pdf.mutual_information([zix], [numZ]) for zix in range(numZ)]), bZ / numZ)

        # finally add Y
        pdf.append_variables_using_conditional_distributions(pdfY_X, [numZ])  # append Y dependent on X

        # now the pdf is numZ+2 variables: (Z[0],...,Z[numZ],X,Y)

        relmi_xy = pdf.mutual_information([numZ], [numZ + 1]) / pdf.entropy([numZ + 1])
        relmi_zx = pdf.mutual_information([numZ], range(numZ)) / pdf.entropy([numZ])

        if verbose:
            print 'note: I(X:Y)/H(Y) = %s (intended: %s)' % (relmi_xy, b)
            print 'note: I(X:Z0,..,Zn)/H(X) = %s (intended: %s)' % (relmi_zx, bZ)

        if abs(relmi_xy - b) / b > tol_rel:
            if verbose:
                print 'note: pdf not good enough (b)'

            pdf = None

            continue  # not good enough
        elif abs(relmi_zx - bZ) / bZ > tol_rel:
            if verbose:
                print 'note: pdf not good enough (bZ)'

            pdf = None

            continue  # not good enough
        else:
            break  # good, let's use it

    if pdf is None:
        warnings.warn(
            'I could not find a good enough pdf: (b, bZ, numZ, bW, numW, k) = %s' % str((b, bZ, numZ, bW, numW, k)))

    if numW > 0:
        if bW > 0:
            for misW_trial in range(50):
                misW = bW * pY.entropy() / np.random.poisson(numW, numW)
                misW = misW / np.sum(misW) * bW * pX.entropy()

                if np.min(misW) > 0:
                    break
        else:
            bW = np.zeros(numW)

        for wi in range(numW):
            if misW[wi] > 0:
                pdf.append_variables_with_target_mi(1, misW[wi], [numZ + 1])
            else:
                pdf.append_independent_variables(jointpdf.JointProbabilityMatrix(1, pdf.numvalues))

    return pdf


def param_sweep_testcases_implicit_nudging(nprocs=1, use_real_Zs=True, numsamples_z=4000, store_datasets=False,
                                           trial_list=None, bZ_list=None, b_list=None, numZ_list=None, k_list=None,
                                           numsamples_list=None, sigma_list=None,
                                           mz_list=None, kz_list=None, kxy_list=None, sample_Z_uniformly=False):
    """
    Do a parameter sweep over generated testcase datasets and see what the distribution of I(X:Y|Z) is and whether
    it admits inferring causality (left skew).
    :param numsamples_z: if True then 'cheat' and directly use all known Z_i's for Z; otherwise
    test many sets (takes longer). (Still the best_Zs will be calculated and returned.)
    :type numsamples_z: object
    :return: pdfs, datasets, best_Zs_per_dataset, real_condmi_dist, estimated_condmi_dist_per_Z, all_param_lists_dict
    :rtype: tuple
    """

    # parameter ranges: currently set to a minimum so that it only takes about 5 hours on 7 cores (my 2015 i7 laptop)
    trial_list = range(2) if trial_list is None else trial_list
    numsamples_list = [100000, 500000] if numsamples_list is None else numsamples_list
    bZ_list = [0.2, 0.4] if bZ_list is None else bZ_list
    b_list = [0.5, 0.75] if b_list is None else b_list
    numZ_list = [3, 5] if numZ_list is None else numZ_list
    k_list = [3, 5] if k_list is None else k_list
    mz_list = [400] if mz_list is None else mz_list
    kz_list = [10] if kz_list is None else kz_list
    kxy_list = [3] if kz_list is None else kz_list
    sigma_list = [0.15] if sigma_list is None else sigma_list

    pdf_param_lists = [trial_list, bZ_list, b_list, numZ_list, k_list]  # params for creating pdfs
    data_param_lists = [numsamples_list, sigma_list]  # additional params for creating a dataset
    mi_param_lists = [mz_list, kz_list, kxy_list]  # additional params for measuring cond. MI distributions

    all_param_lists = pdf_param_lists + data_param_lists + mi_param_lists
    num_param_sets = np.product(map(len, all_param_lists))

    # useful for returning so that the user knows which parameter is which
    param_names = ['trial_list', 'bZ_list', 'b_list', 'numZ_list', 'k_list',
                   'numsamples_list', 'sigma_list',
                   'mz_list', 'kz_list', 'kxy_list']
    assert len(param_names) == len(all_param_lists)
    all_param_lists_dict = {pn: all_param_lists[pni] for pni, pn in enumerate(param_names)}

    # store results, keys are tuples of param values in order of `all_param_lists` (or left subsets of it)
    pdfs = dict()
    datasets = dict()  # store_datasets=True: is left empty because this would take a lot of memory! (see inner loop)
    best_Zs_per_dataset = dict()
    real_condmi_dist = dict()
    estimated_condmi_dist_per_Z = dict()

    time_before_global = time.time()
    num_pdf_param_sets = np.product(map(len, pdf_param_lists))
    pdfix = 0
    for trial, bZ, b, numZ, k in itertools.product(*pdf_param_lists):
        pdf_params_key = (trial, bZ, b, numZ, k)  # for use in dictionary, should change if iterator changes
        pdfix += 1

        print(
        'NOTE: started pdf param set %s/%s after %s hours; (trial, bZ, b, numZ, k) = %s' % (pdfix, num_pdf_param_sets,
                                                                                            (
                                                                                            time.time() - time_before_global) / 3600.,
                                                                                            (trial, bZ, b, numZ, k)))

        ### GENERATE PDF

        if not pdfs.has_key(pdf_params_key):
            pdf = generate_causal_model(b, bZ, numZ, bW=0, numW=0, k=k, num_pdf_trials=20)
            pdfs[pdf_params_key] = pdf
        else:
            pdf = pdfs[pdf_params_key]  # probably previous run was interrupted

        ### GENERATE DATA

        for numsamples, sigma in itertools.product(*data_param_lists):
            data_params_key = pdf_params_key + (numsamples, sigma)  # in case we want to store datasets

            time_before = time.time()
            if not datasets.has_key(data_params_key):
                # because my implementation is inefficient, this step takes VERY long so try to make `nprocs` high!
                # (with nprocs=2 and k=5 and numZ=5 and numsaples=1e6 it took about 7 hours)
                dataset = pdf.generate_samples(int(numsamples),
                                               nprocs=nprocs)  # list of tuples (z_1, ..., z_n, x, y) -- independent samples
                print 'note: generating discrete data took %s seconds' % (time.time() - time_before)
                # add gaussian noise to each datapoint:
                dataset_gauss = pdf.generate_samples_mixed_gaussian(dataset, sigma=sigma)
                dataset_gauss = standardize_columns(dataset_gauss)

                if store_datasets:
                    datasets[data_params_key] = dataset_gauss
                else:
                    pass  # takes too much memory
            else:
                dataset_gauss = datasets[data_params_key]  # reuse from previous computation
            print 'note: generating Gaussian data took %s seconds in total' % (time.time() - time_before)

            pdf2 = jointpdf.JointProbabilityMatrix(1, 2)  # dummy object just to get to the estimator function
            pdf2.estimate_from_data(dataset)  # estimate a categorical pmf from the data (should be similar to 'pdf')

            pZ2 = pdf2.marginalize_distribution(range(numZ))  # get the estimated p(Z')

            for mz, kz, kxy in itertools.product(*mi_param_lists):
                # for storing results per cond. MI estimation
                mi_params_key = data_params_key + (mz, kz, kxy)

                ### ESTIMATE CONDITIONAL MUTUAL INFORMATION DISTRIBUTION ('EXACT')

                if not real_condmi_dist.has_key(mi_params_key):
                    # this will sample I(X:Y|z) values proportionally to p(z), to resemble real-life sampling
                    # note: this is a 'perfect' distribution, which the below sampling using conditional_mi_knn()
                    # can only try to approximate
                    condmis2 = pdf2.conditional_mutual_informations([numZ], [numZ + 1], range(numZ), nprocs=4)
                    condmi2_dist = []
                    n_add = np.power(len(pZ2), k) * 100.
                    for z in pZ2.statespace():
                        condmi2_dist = condmi2_dist + [condmis2[z]] * int(n_add * pZ2(z))

                    real_condmi_dist[mi_params_key] = copy.deepcopy(condmi2_dist)  # store result
                else:
                    condmi2_dist = real_condmi_dist[mi_params_key]  # reuse from previous computation

                ### ESTIMATE CONDITIONAL MUTUAL INFORMATION DISTRIBUTION ('SAMPLING')

                if not best_Zs_per_dataset.has_key(mi_params_key):
                    # find the sequence of sets of non-XY variables to test as Z
                    best_Zs = potential_Zs(dataset_gauss, n=30000, method='backward')
                    best_Zs_real = [range(numZ)]  # this is the real set that was used to generate the data

                    best_Zs_per_dataset[mi_params_key] = best_Zs
                else:
                    best_Zs = best_Zs_per_dataset[mi_params_key]  # reuse from previous computation
                    best_Zs_real = [range(numZ)]  # this is the real set that was used to generate the data

                if not estimated_condmi_dist_per_Z.has_key(mi_params_key):
                    # note: if use_real_Zs=True then condmis_per_Z will consist of only one list
                    condmis_per_Z = []
                    for zs in (best_Zs if not use_real_Zs else best_Zs_real):
                        #                 time_before = time.time()
                        condmis = calculate_condmi_knn(dataset_gauss, setZ=zs, nprocs=nprocs, mz=mz, kz=kz, kxy=kxy,
                                                       sample_Z_uniformly=sample_Z_uniformly)
                        condmis_per_Z.append(condmis)

                    # note: this one can only hope to get close to real_condmi_dist[mi_params_key]
                    estimated_condmi_dist_per_Z[mi_params_key] = condmis_per_Z  # store result
                else:
                    condmis_per_Z = estimated_condmi_dist_per_Z[mi_params_key]  # reuse from previous computation

    print('note: finished, in total took %s hours' % ((time.time() - time_before_global) / 3600.))

    return pdfs, datasets, best_Zs_per_dataset, real_condmi_dist, estimated_condmi_dist_per_Z, all_param_lists_dict


def infer_causality(dataset_gauss, mz=400, kz=10, kxy=3, zs=None, xix=None, yix=None, quit_on_hit=False,
                    n_find_Zs=30000, method_find_Zs='backward', alpha=0.05, nprocs=1, numsamples_z=4000,
                    find_tol=0.2, find_tol_abs=0.5, min_num_indivZs=3,
                    max_num_indivZs=3, min_num_Zs=3, max_num_Zs=7, max_num_setsZ=100, ntrials_skew_diff=2000,
                    sample_Z_uniformly=False, verbose=False):
    """

    :param dataset_gauss:
    :param mz: each individual cond. MI value is estimated using `mz` samples nearest to a point Z=z
    :param kz:
    :param kxy:
    :param zs: trumps method_find_Zs, will just use this single Zs set for computing cond. MI distribution and not
    try to explore different sets of Zs.
    :param n_find_Zs:
    :param method_find_Zs:
    :param alpha:
    :param nprocs:
    :param numsamples_z: this number of cond. MI values will be calculated, one per randomly chosen
    Z=z data point, and each cond. MI estimated using `mz` samples
    :return: best_Zs, condmis_per_Z, causal_per_Z, alpha_multi
    :rtype: tuple
    """
    assert np.ndim(dataset_gauss) == 2, 'dataset should be shaped (#cells, #genes)'
    numcells, numgenes = np.shape(dataset_gauss)
    assert numcells > numgenes, 'dataset should be shaped (#cells, #genes); did you transpose it?'

    # if numsamples_z is None:
    #     numsamples_z = len(dataset_gauss)  # simply use all data, do not subsample

    if zs is None:
        assert not xix is None
        assert not yix is None

        # find the sequence of sets of non-XY variables to test as Z
        best_Zs = potential_Zs(dataset_gauss, n=n_find_Zs, method=method_find_Zs, tol=find_tol, tol_abs=find_tol_abs,
                               min_num_indivZs=min_num_indivZs, max_num_indivZs=max_num_indivZs, min_num_Zs=min_num_Zs,
                               max_num_Zs=max_num_Zs, max_num_setsZ=max_num_setsZ,
                               setZ=np.setdiff1d(range(np.shape(dataset_gauss)[1]), [xix, yix]), nprocs=nprocs)

        if verbose:
            print 'debug: best Z set calculated, length is %s' % len(best_Zs)
    else:
        best_Zs = [zs]  # try only the Z set provided

    alpha_multi = alpha / float(len(best_Zs))  # multiple hypothesis correction

    # note: if use_real_Zs=True then condmis_per_Z will consist of only one list
    condmis_per_Z = []
    mis_per_Z = []  # null-hypothesis of random z's instead of neighboring z's
    causal_per_Z = []
    time_before = time.time()
    for zix2, zs in enumerate(best_Zs):
        assert not xix in zs, 'should not happen: X=%s in Z=%s' % (xix, zs)
        assert not yix in zs, 'should not happen: Y=%s in Z=%s' % (yix, zs)
        condmis = calculate_condmi_knn(dataset_gauss, setZ=zs, nprocs=nprocs, mz=mz, kz=kz, kxy=kxy,
                                       numsamples_z=numsamples_z, xix=xix, yix=yix,
                                       sample_Z_uniformly=sample_Z_uniformly, verbose=verbose)
        condmis_per_Z.append(condmis)

        if verbose:
            print 'debug: done with part of the loop after %s seconds ' \
                  '(after calculate_condmi_knn)' % (time.time() - time_before)

        # NULL-H
        mis = calculate_condmi_knn_null(dataset_gauss, setZ=zs, nprocs=nprocs, mz=mz, kz=kz, kxy=kxy,
                                        numsamples_z=numsamples_z, xix=xix, yix=yix,
                                        sample_Z_uniformly=sample_Z_uniformly, verbose=verbose)
        mis_per_Z.append(mis)

        if ntrials_skew_diff:
            skewz, skewp = st.skewtest(condmis)
            # skewznull, skewpnull = st.skewtest(mis)

            if skewz < 0 and skewp < alpha_multi:
                # this is a relatively quick calculation so no need to store it in the return tuple I guess
                median_skew_diff, ci_low_diff, ci_high_diff = skew_diff(condmis, mis, alpha=alpha_multi,
                                                                        ntrials=ntrials_skew_diff)
                if ci_high_diff < 0.0:
                    causal_per_Z.append(True)

                    if quit_on_hit:
                        if int(quit_on_hit) <= sum(causal_per_Z):
                            break  # found enough causal cases to quit
            else:
                causal_per_Z.append(False)
        else:
            causal_per_Z.append(None)

        if verbose:
            print 'debug: processed loop %s of %s after %s seconds' % (zix2, len(best_Zs), time.time() - time_before)

    return best_Zs, condmis_per_Z, mis_per_Z, causal_per_Z, alpha_multi