__author__ = 'rquax'


import copy
import numpy as np
import scipy.stats as st
import seaborn as sns
import scipy.special as ss

import jointpdf
# import multiprocessing
import pathos.multiprocessing as multiprocessing  # standard multiprocessing package fails to pickle instance functions


class NudgeSamples():

    pdf_orig = None  # the pdf which is being nudged for every entropy in self.causal_mis (for instance), with causality

    causal_mis = []
    noncausal_mis = []

    causal_samples = []
    noncausal_samples = []

    causal_nudges = []
    noncausal_nudges = []

    causal_impacts = []


    def __init__(self):
        self.causal_mis = []
        self.noncausal_mis = []

        self.causal_samples = []
        self.noncausal_samples = []

        self.causal_nudges = []
        self.noncausal_nudges = []

        self.causal_impacts = []


    def extend(self, other):
        """

        :type other: NudgeSamples
        """
        self.causal_mis.extend(other.causal_mis)
        self.noncausal_mis.extend(other.noncausal_mis)

        self.causal_samples.extend(other.causal_samples)
        self.noncausal_samples.extend(other.noncausal_samples)

        self.causal_nudges.extend(other.causal_nudges)
        self.noncausal_nudges.extend(other.noncausal_nudges)

        if hasattr(other, 'causal_impacts'):
            self.causal_impacts.extend(other.causal_impacts)


    def plot(self):
        sns.plt.figure(figsize=(12, 12))
        sns.distplot(self.causal_mis, color='b', label='MI causal', kde=True)
        sns.distplot(self.noncausal_mis, color='g', label='MI non-causal', kde=True)

        # xs = np.linspace(*sns.plt.xlim(), num=50)
        # sns.plt.plot(xs, st.norm.pdf(xs, loc=np.mean(self.causal_mis), scale=np.std(self.causal_mis)), color='b')
        # sns.plt.plot(xs, st.norm.pdf(xs, loc=np.mean(self.noncausal_mis), scale=np.std(self.noncausal_mis)), color='g')

        sns.plt.plot([np.mean(self.causal_mis), np.mean(self.causal_mis)],
                     sns.plt.ylim(), color = 'b', label='Mean causal')
        sns.plt.plot([np.median(self.causal_mis), np.median(self.causal_mis)],
                     sns.plt.ylim(), color = 'b', linestyle='--', label='Median causal')
        sns.plt.plot([np.mean(self.noncausal_mis), np.mean(self.noncausal_mis)],
                     sns.plt.ylim(), color = 'g', label='Mean non-causal')
        sns.plt.plot([np.median(self.noncausal_mis), np.median(self.noncausal_mis)],
                     sns.plt.ylim(), color = 'g', linestyle='--', label='Median con-causal')
        pvalue = test_twosample_diff(self.causal_mis, self.noncausal_mis, st.skew)
        sns.plt.title('Causal skew: %s (%s),\nnon-causal skew: %s (%s)\np-value for rejecting equal median(skew): %s'
                      % (st.skew(np.array(self.causal_mis, dtype=np.float64)),
                         st.skewtest(np.array(self.causal_mis, dtype=np.float64)),
                         st.skew(np.array(self.noncausal_mis, dtype=np.float64)),
                         st.skewtest(np.array(self.noncausal_mis, dtype=np.float64)),
                         pvalue))
        sns.plt.legend()
        sns.plt.show()


def random_walk_bounded(n, numdims, max_norm=0.01, sigma=0.0001, startvec='random'):
    """
    A (potentially) high-dimensional random, uncorrelated walking which has a bounded vector norm. The way in which
    the bound is implement is to reflect points outside this hypersphere, i.e., if they are distance x outside the
    sphere then they become distance x inside the sphere.
    :param n:
    :param numdims:
    :param max_norm:
    :param sigma:
    :param startvec:
    :return:
    """
    if startvec == 'random':
        startvec = np.random.uniform(0.0, max_norm, numdims)

        if np.linalg.norm(startvec) > max_norm:
            startvec = startvec / np.linalg.norm(startvec) * max_norm
    elif startvec == 0:
        startvec = np.zeros(numdims)
    else:
        assert len(startvec) == numdims

        startvec = np.array(startvec)

        if np.linalg.norm(startvec) > max_norm:
            startvec = startvec / np.linalg.norm(startvec) * max_norm

    randvecs = np.random.normal(0.0, sigma, [n, numdims])

    curvec = startvec

    for t, rv in enumerate(randvecs):
        newvec = curvec + rv

        if np.linalg.norm(newvec) > max_norm:
            # newvec = curvec + np.random.normal(0.0, sigma, numdims)  # draw different random vector
            newvec = curvec - rv

            if np.linalg.norm(newvec) > max_norm:  # still? should be very rare, but can happen
                newvec = curvec  # just stay where you are

            assert np.linalg.norm(newvec) <= max_norm, 'norm(start)=%s, t=%s, norm(new)=%s, norm(cur)=%s, norm(at first)=%s' \
                                                       % (np.linalg.norm((startvec)), t, np.linalg.norm(newvec), np.linalg.norm(curvec), np.linalg.norm(curvec + rv))

        randvecs[t] = newvec
        curvec = newvec

    # vecs = startvec + np.cumsum(randvecs, axis=0)
    # norms = np.linalg.norm(vecs, axis=1) / max_norm  # relative norms
    # norms_rem2 = np.remainder(norms, 2.0)
    # # norm_overshoots = np.max([norms - np.ones(n)*max_norm, np.zeros(n)], axis=0) / norms
    # # factor = -1.0 * (1.0 - np.remainder(norms, 1.0))
    # new_norms = -(norms_rem2 - 2.0 * np.remainder(norms_rem2, 1.0)) * max_norm
    # # print 'debug: norms =', norms
    # # print 'debug: factor.shape =', factor.shape
    # factor = np.select([norms_rem2 > 1.0], [new_norms / (norms * max_norm)], 1.0)
    #
    # # print 'debug: factor.shape =', factor.shape
    #
    # vecs = vecs * factor.reshape(n, 1)

    # norm_num_overshoots = np.floor(norm_overshoots)
    # norm_overshoots = norm_overshoots - norm_num_overshoots  # now only decimal part of overshoot
    # vecs = np.multiply(vecs, np.reshape((norms - norm_num_overshoots * max_norm - 2*norm_overshoots) / norms, [n, 1]))

    return randvecs


# # helper function
# def gamma_inverse(x):
#     """
#     Inverse the gamma function.
#     http://mathoverflow.net/questions/12828/inverse-gamma-function
#     """
#     k=1.461632 # the positive zero of the digamma function, scipy.special.psi
#     assert x>=k, 'gamma(x) is strictly increasing for x >= k, k=%1.2f, x=%1.2f' % (k, x)
#     C=np.sqrt(2*np.pi)/np.e - ss.gamma(k) # approximately 0.036534
#     L=np.log((x+C)/np.sqrt(2*np.pi))
#     gamma_inv = 0.5+L/ ss.lambertw(L/np.e)
#     return gamma_inv


# def mean_chi_distr_asymptotic(k):
#     """
#     For large k the real calculation using the gamma functions returns NaN, already at k=1000, but I need it for
#     millions. Empirically I find that the square-root of k is a very good approximation in the large k limit.
#     :param k: number of degrees of freedom
#     :return:
#     """
#     # to see this closeness of the approximation, plot e.g.:
#     # xi = np.linspace(0, 200, 50)
#     # sns.plt.plot(xi, np.sqrt(2) * ss.gamma((xi+1)/2.)/ss.gamma(xi/2.) / np.sqrt(xi)); sns.plt.show()
#     return np.sqrt(k)


def mixing_time(numdims, desired_norm, sigma):
    """

    :param numdims: nudge.size
    :param desired_norm: suggestion: 2*epsilon., or 2*pi*epsilon.
    The desired displacement that you would consider 'mixing'
    :param sigma: each step the random walk displaces by [N(0,sigma)] per dimension independently
    """
    # 1/(sqrt(n)*sigma) * ||X|| ~ ChiSq(k)        [k == numdims)
    # for mixing time we want to know n for a given sigma and desired_norm=||X||
    # so: 1/(sqrt(n)*sigma) * E[||X||] == E[ChiSq(k)]
    # so: 1/sqrt(n) * 1/(sigma) * E[||X||] == E[ChiSq(k)]
    # so: 1/sqrt(n) == E[ChiSq(k)] * (sigma) / E[||X||]
    # so: sqrt(n) == E[||X||] / (E[ChiSq(k)] * sigma)

    return np.power(desired_norm / (st.chi.mean(numdims) * sigma), 2)


def expected_distance_after(numdims, numsteps, sigma):
    """

    :param numdims: nudge.size
    :param desired_norm: suggestion: 2*epsilon., or 2*pi*epsilon.
    The desired displacement that you would consider 'mixing'
    :param sigma: each step the random walk displaces by [N(0,sigma)] per dimension independently
    """
    # 1/(sqrt(n)*sigma) * ||X|| ~ ChiSq(k)        [k == numdims)
    # for mixing time we want to know n for a given sigma and desired_norm=||X||
    # so: 1/(sqrt(n)*sigma) * E[||X||] == E[ChiSq(k)]
    # so: 1/sqrt(n) * 1/(sigma) * E[||X||] == E[ChiSq(k)]
    # so: 1/sqrt(n) == E[ChiSq(k)] * (sigma) / E[||X||]
    # so: sqrt(n) == E[||X||] / (E[ChiSq(k)] * sigma)

    return np.sqrt(numsteps) * (st.chi.mean(numdims) * sigma)


def sigma_for_mixing_time(numdims, desired_mixing_time, desired_mixing_norm):
    """

    :param numdims: nudge.size
    :param desired_mixing_time: probably the length of your time series, or half of it, for instance
    """
    # ...continuing from the derivation notes from mixing_time():
    # sqrt(n) == desired_norm / (E[ChiSq(numdims)] * (sigma))
    # (sigma) == desired_norm / (E[ChiSq(numdims)] * sqrt(n))
    #
    # suppose you compute MI in window of 2000, in which you want the nudge to be roughly the same. How many data
    # ponts in total you need? (for epsilon e)
    # sigma_2000 = 0.1*e / (sqrt(k) * sqrt(2000))
    # mixing_2000 = 2*e / (sqrt(k) * sigma_2000)
    # mixing_2000 = (2*e * sqrt(k) * sqrt(2000)) / (sqrt(k) * 0.1*e)
    # mixing_2000 = (2/0.1 * sqrt(2000))
    # IN WORDS, if you want to keep within 0.1*e during a sliding window for computing MI, but you want the nudge
    # to 'mix' sufficiently during your whole time-series dataset at 2*e norm, then you need 2/0.1 = 20 times 2000
    # dta points, so 40,000. (Though it sounds maybe should be 2*pi*e at least, not just 2*e, for 'mixing...)

    return desired_mixing_norm / (st.chi.mean(numdims) * np.sqrt(desired_mixing_time))


class SlowNudgeResponse():

    ### init values are only intended to show the data type

    pdf = jointpdf.JointProbabilityMatrix(2, 2)

    n = 0
    epsilon = 0.0
    sigma = 0.0

    ### all lists are of length self.n

    samples_vanilla = []
    samples_causal = []
    samples_noncausal = []

    nudges = []
    actual_nudges_causal = []
    actual_nudges_noncausal = []

    nudged_pdfs = []


    def __init__(self, other=None):  # mostly: prevent unintended copy by reference, and unexpected changes across objects
        """

        :type other: SlowNudgeResponse
        """
        if other is None:
            self.pdf = None

            self.n = 0
            self.epsilon = 0.0
            self.sigma = 0.0

            ### all lists are of length self.n

            self.samples_vanilla = []
            self.samples_causal = []
            self.samples_noncausal = []

            self.nudges = []
            self.actual_nudges_causal = []
            self.actual_nudges_noncausal = []

            self.nudged_pdfs = []
        else:
            self.duplicate(other)


    def duplicate(self, other):
        """

        :type other: SlowNudgeResponse
        """
        self.pdf = other.pdf.copy()

        self.n = other.n
        self.epsilon = other.window
        self.sigma = other.sigma

        ### all lists are of length self.n

        self.samples_vanilla = list(other.samples_vanilla)
        self.samples_causal = list(other.samples_causal)
        self.samples_noncausal = list(other.samples_noncausal)

        self.nudges = list(other.nudges)
        self.actual_nudges_causal = list(other.actual_nudges_causal)
        self.actual_nudges_noncausal = list(other.actual_nudges_noncausal)

        self.nudged_pdfs = list(other.nudged_pdfs)




def slow_nudge_samples(pdf, n, epsilon, sigma='auto'):  # todo
    """

    :type pdf: jointpdf.JointProbabilityMatrix
    :param sigma: suggestion: if you want to keep within 0.1*e during a sliding window for computing MI,
    but you want the nudge (with desired norm 'e')
    to 'mix' sufficiently during your whole time-series dataset at 2*e norm, then you need 2/0.1 = 20 times 2000
    data points, so 40,000. (Though it sounds maybe should be 2*pi*e at least, not just 2*e, for 'mixing...)
    """

    assert len(pdf) == 2, 'I assume two variables (X,Y), where X will be nudged and Y is/is not be causally impacted'

    numdims = pdf.numvalues

    if sigma == 'auto':
        sigma = sigma_for_mixing_time(pdf.numvalues, n, 2.0*epsilon)
    else:
        sigma = float(sigma)  # make sure it is a float

    nudges = random_walk_bounded(n, numdims, epsilon, sigma, 'random')

    samples_vanilla = pdf.generate_samples(n)
    samples_causal = []
    samples_noncausal = []

    # these will be the actual nudges performed, starting from the 'nudges' in case that fails
    actual_nudges_causal = []
    actual_nudges_noncausal = []

    nudged_pdfs = []

    # in pdf_indep I will nudge the second variable, which is the input still, but now with no causal impact on output.
    # pre-alloc
    pdf_indep = pdf.copy()
    pdf_indep.reorder_variables([1,0])

    for t in xrange(n):
        pdf_c = pdf.copy()
        pdf_nc = pdf_indep.copy()

        nudge = pdf_c.nudge([0], [1], epsilon=epsilon)  # CAUSAL nudge
        nudge_indep = pdf_nc.nudge([1], [], epsilon=epsilon)  # NON-CAUSAL nudge

        # impact = pdf_c[1].joint_probabilities.joint_probabilities - pdf_output_probs
        nudged_pdfs.append(pdf_c)  # so that e.g. the impact term can be computed

        samples_causal.append(pdf_c.generate_sample())
        samples_noncausal.append(tuple(reversed(pdf_nc.generate_sample())))

        actual_nudges_causal.append(nudge)
        actual_nudges_noncausal.append(nudge_indep)

    resp = SlowNudgeResponse()

    resp.nudges = nudges
    resp.samples_vanilla = samples_vanilla
    resp.samples_causal = samples_causal
    resp.samples_noncausal = samples_noncausal
    resp.actual_nudges_causal = actual_nudges_causal
    resp.actual_nudges_noncausal = actual_nudges_noncausal
    resp.nudged_pdfs = nudged_pdfs

    resp.pdf = pdf
    resp.n = n
    resp.sigma = sigma
    resp.epsilon = epsilon

    return resp



def nudge_samples(numvalues, target_mi_frac=0.5, num_samples=1000, num_data_samples_per_pdf=1, epsilon=0.01,
                  method='random', also_nudge_output=False, output_variable_independent=False, pdf_orig=None,
                  nprocs=1):

    if pdf_orig is None:
        pdf_orig = jointpdf.JointProbabilityMatrix(1, numvalues)
        # try:
        pdf_orig.append_variables_with_target_mi(1, target_mi=target_mi_frac * pdf_orig.entropy([0]))
    else:
        pass

    if nprocs > 1:
        def worker(num):
            np.random.seed()
            return nudge_samples(numvalues=numvalues,
                                 target_mi_frac=target_mi_frac,
                                 num_samples=num,
                                 num_data_samples_per_pdf=num_data_samples_per_pdf,
                                 epsilon=epsilon,
                                 method=method,
                                 also_nudge_output=also_nudge_output,
                                 output_variable_independent=output_variable_independent,
                                 pdf_orig=pdf_orig,
                                 nprocs=1)  # only change

        pool = multiprocessing.Pool(nprocs)

        nudge_list = pool.map(worker, [int(round(num_samples / float(nprocs)))]*nprocs)

        resp = nudge_list[0]
        for n in nudge_list[1:]:
            resp.extend(n)

        return resp

    resp = NudgeSamples()

    if output_variable_independent:
        # this should not be done probably, because it compared two different cases (one with nonzero MI and
        # the other with zero MI, which after nudging has strongly positive skew)

        pdf_indep_orig = pdf_orig[0].copy()
        pdf_indep_orig.append_independent_variables(pdf_orig[1])

        pdf_input_probs_indep = pdf_indep_orig[0].joint_probabilities.joint_probabilities
        pdf_output_probs_indep = pdf_indep_orig[1].joint_probabilities.joint_probabilities
    else:
        # copy so that the same conditional probabilities hold, but reverse so that the causal relation is reversed,
        # so then perturbing pdf[0] (here now pdf_indep[1]) will have no effect on the output variable
        pdf_indep_orig = pdf_orig.copy()
        pdf_indep_orig.reorder_variables([1, 0])

        pdf_input_probs_indep = pdf_indep_orig[1].joint_probabilities.joint_probabilities
        pdf_output_probs_indep = pdf_indep_orig[0].joint_probabilities.joint_probabilities

    pdf_input_probs = pdf_orig[0].joint_probabilities.joint_probabilities
    pdf_output_probs = pdf_orig[1].joint_probabilities.joint_probabilities

    resp.pdf_orig = pdf_orig.copy()

    num_nudge_vec_fails = 0
    max_num_nudge_vec_fails = 1000

    for trial in xrange(num_samples):
        try:
            pdf = pdf_orig.copy()
            pdf_indep = pdf_indep_orig.copy()

            nudge = pdf.nudge([0], [1], epsilon=epsilon, method=method)  # CAUSAL nudge
            # todo: apply the same nudge to pdf_indep, instead of an independently random one?
            if output_variable_independent:
                nudge_indep = pdf_indep.nudge([0], [1], epsilon=epsilon, method=method)  # NON-CAUSAL nudge
            else:
                nudge_indep = pdf_indep.nudge([1], [], epsilon=epsilon, method=method)  # NON-CAUSAL nudge

            if also_nudge_output:
                if output_variable_independent:
                    pdf.nudge([1], [], epsilon=epsilon, method=method)
                    pdf_indep.nudge([1], [], epsilon=epsilon, method=method)
                else:
                    pdf.nudge([1], [], epsilon=epsilon, method=method)
                    # note: no idea of this works! so nudging variable 0 while variable 1 is ignored... will there be
                    # still causal effect on 1? No idea. But should not
                    pdf_indep.nudge([0], [], epsilon=epsilon, method=method)

            samples = pdf.generate_samples(num_data_samples_per_pdf)
            samples_indep = pdf_indep.generate_samples(num_data_samples_per_pdf)

            if not output_variable_independent:
                # in this case pdf_indep is reversed to make sure the nudge has no causal effect, but then the samples
                # are also reversed so here I correct this
                samples_indep = map(lambda a: tuple(reversed(a)), samples_indep)

            # calculate the new MIs
            mi = pdf.mutual_information([0], [1])
            mi_indep = pdf_indep.mutual_information([0], [1])

            # this is the causal impact (array of probability differences) of the nudge
            impact = pdf[1].joint_probabilities.joint_probabilities - pdf_output_probs
            # the pdf_indep should be such that there is no causal impact of a nudge on the output variable
            if output_variable_independent:
                assert np.isclose(np.sum(pdf_indep[1].joint_probabilities.joint_probabilities - pdf_output_probs_indep),
                                  0.0)
            else:
                assert np.isclose(np.sum(pdf_indep[0].joint_probabilities.joint_probabilities - pdf_output_probs_indep),
                                  0.0)
        except UserWarning as e:
            # 'different exception than what I intended to ignore: ' + str(e)
            if not 'was not enough to find a good nudge vector' in str(e):
                raise UserWarning(e)

            num_nudge_vec_fails += 1

            print 'error:', e
            if num_nudge_vec_fails <= max_num_nudge_vec_fails or len(resp.causal_mis) > int(0.1*max_num_nudge_vec_fails):
                print 'error: (ignored)'
                print 'debug: num successful samples so far:', len(resp.causal_mis)
            else:
                print 'error: there were too many failures for finding nudges (%s), and too few successes (%s), so ' \
                      'I will just return what I have so far and give up.' % (num_nudge_vec_fails, len(resp.causal_mis))
                return resp
        except KeyboardInterrupt as e:
            # note: this is not perfect, you may also happen to be in the else-clause appending things when you
            # do keyboard interrupt...
            print 'warning: you interrupted me. I will return what I have so far'

            return resp
        else:
            resp.causal_mis.append(mi)
            resp.noncausal_mis.append(mi_indep)
            resp.causal_samples.extend(samples)
            resp.noncausal_samples.extend(samples_indep)
            resp.causal_nudges.append(nudge)
            resp.noncausal_nudges.append(nudge_indep)
            resp.causal_impacts.append(impact)

    return resp


# this is a TRIAL, not sure if it is useful... especially how pdf_noncausal is constructed is a bit fishy (has zero MI
# whereas co-influenced variables do have MI, and usually the data does as well, so assuming strict independence
# is not a good fit anyway... May have to optimize for finding a common causal ancestor to create a specific
# non-causal MI
def test_nudge_causality(samples, n, repeats=1000, epsilon=0.01, numvalues='auto'):
    pdf = jointpdf.JointProbabilityMatrix(2, 3)
    pdf.estimate_from_data(samples, numvalues=numvalues)

    assert len(pdf) == 2, 'this code assumes two variables (one input one output) currently'

    # in case of causal relation the cond_pdf would not change
    cond_pdf = pdf.conditional_probability_distributions([0])
    pdf_output = pdf[1]

    ll_causal = []
    ll_noncausal = []

    pdf_input = pdf[0].copy()  # pre-alloc

    for r in xrange(repeats):
        subsample = np.take(samples, np.random.choice(range(len(samples)), n, replace=False), axis=0)
        pdf_input.estimate_from_data(subsample, numvalues=numvalues)
        pdf_input = pdf_input[0]  # only need the input part

        pdf_causal = pdf_input + cond_pdf
        pdf_noncausal = pdf_input + pdf_output

        ll_causal.append(pdf_causal.loglikelihood(subsample))
        ll_noncausal.append(pdf_noncausal.loglikelihood(subsample))

    return ll_causal, ll_noncausal


def mi_subsample(samples, n, numvalues='auto', replace=False):
    """
    Sub-sample once ``n'' samples from ``samples'' and use it to compute a MI. Youl should repeat this many times
    to find a distribution for MI for a particular sample size.
    :param samples:
    :param n:
    :param numvalues:
    :return:
    """
    pdf3 = jointpdf.JointProbabilityMatrix(1, 2)
    subsample = np.take(samples, np.random.choice(range(len(samples)), n, replace=replace), axis=0)
    pdf3.estimate_from_data(subsample, numvalues=numvalues)
    return pdf3.mutual_information([0], [1])


def mi_subsample_distr(pdf_orig, n, repeats=1000, epsilon=0.01, output_variable_independent=False, method='fixed',
                       also_nudge_output=False):
    assert len(pdf_orig) == 2, 'assumed in the code for now, input and output'

    if output_variable_independent:
        # this should not be done probably, because it compared two different cases (one with nonzero MI and
        # the other with zero MI, which after nudging has strongly positive skew)

        pdf_indep_orig = pdf_orig[0].copy()
        pdf_indep_orig.append_independent_variables(pdf_orig[1])

        pdf_input_probs_indep = pdf_indep_orig[0].joint_probabilities.joint_probabilities
        pdf_output_probs_indep = pdf_indep_orig[1].joint_probabilities.joint_probabilities
    else:
        # copy so that the same conditional probabilities hold, but reverse so that the causal relation is reversed,
        # so then perturbing pdf[0] (here now pdf_indep[1]) will have no effect on the output variable
        pdf_indep_orig = pdf_orig.copy()
        pdf_indep_orig.reorder_variables([1, 0])

        pdf_input_probs_indep = pdf_indep_orig[1].joint_probabilities.joint_probabilities
        pdf_output_probs_indep = pdf_indep_orig[0].joint_probabilities.joint_probabilities

    pdf_fit_causal = pdf_orig.copy()  # pre-alloc
    pdf_fit_noncausal = pdf_orig.copy()  # pre-alloc

    causal_mis = []
    noncausal_mis = []

    for trial in xrange(repeats):
        try:
            pdf = pdf_orig.copy()
            pdf_indep = pdf_indep_orig.copy()

            nudge = pdf.nudge([0], [1], epsilon=epsilon, method=method)  # CAUSAL nudge
            # todo: apply the same nudge to pdf_indep, instead of an independently random one?
            if output_variable_independent:
                nudge_indep = pdf_indep.nudge([0], [1], epsilon=epsilon, method=method)  # NON-CAUSAL nudge
            else:
                nudge_indep = pdf_indep.nudge([1], [], epsilon=epsilon, method=method)  # NON-CAUSAL nudge

            if also_nudge_output:
                if output_variable_independent:
                    pdf.nudge([1], [], epsilon=epsilon, method=method)
                    pdf_indep.nudge([1], [], epsilon=epsilon, method=method)
                else:
                    pdf.nudge([1], [], epsilon=epsilon, method=method)
                    # note: no idea of this works! so nudging variable 0 while variable 1 is ignored... will there be
                    # still causal effect on 1? No idea. But should not
                    pdf_indep.nudge([0], [], epsilon=epsilon, method=method)

            samples = pdf.generate_samples(n)
            samples_indep = pdf_indep.generate_samples(n)

            if not output_variable_independent:
                # in this case pdf_indep is reversed to make sure the nudge has no causal effect, but then the samples
                # are also reversed so here I correct this
                samples_indep = map(lambda a: tuple(reversed(a)), samples_indep)

            # calculate the new MIs, but after a sampling step
            pdf_fit_causal.estimate_from_data(samples)
            mi = pdf_fit_causal.mutual_information([0], [1])
            pdf_fit_noncausal.estimate_from_data(samples_indep)
            mi_indep = pdf_indep.mutual_information([0], [1])
        except UserWarning as e:
            # 'different exception than what I intended to ignore: ' + str(e)
            if not 'was not enough to find a good nudge vector' in str(e):
                raise UserWarning(e)

            print 'error:', e
            print 'error: (ignored)'
            print 'debug: num successful samples so far:', len(causal_mis)
        except KeyboardInterrupt as e:
            # note: this is not perfect, you may also happen to be in the else-clause appending things when you
            # do keyboard interrupt...
            print 'warning: you interrupted me. I will return what I have so far'

            return causal_mis, noncausal_mis
        else:
            causal_mis.append(mi)
            noncausal_mis.append(mi_indep)

    return causal_mis, noncausal_mis


def test_twosample_diff(sample1, sample2, statistic=None, resamples=1000):
    """
    Helper function.
    :param sample1:
    :param sample2:
    :param statistic:
    :param resamples:
    :return: The p-value for the null-hypothesis "The sample 'statistic' in both populations have the same median"
    """
    import numpy as np

    if statistic is None:
        statistic = np.mean

    stats1 = np.ones(resamples) * np.nan  # pre-alloc
    stats2 = np.ones(resamples) * np.nan

    for i in xrange(resamples):
        resample1 = np.random.choice(sample1, len(sample1), replace=True)
        resample2 = np.random.choice(sample2, len(sample2), replace=True)

        stats1[i] = statistic(np.array(resample1, dtype=np.float64))
        stats2[i] = statistic(np.array(resample2, dtype=np.float64))

    # if the statistic is on average the same for the two populations then I expect a binomial distribution with p=0.5
    # for the number of times stats1 is larger than stats2
    np = np.sum(stats1 > stats2)

    pvalue = 1.0 - st.binom.cdf(resamples * 0.5 + abs(np - resamples * 0.5), resamples, 0.5)
    pvalue += st.binom.cdf(resamples * 0.5 - abs(np - resamples * 0.5), resamples, 0.5)

    return pvalue


def sort_vectors(vecs):
    new_ixs = [np.random.randint(len(vecs))]

    # def dist(vec):
    #     return np.sum(np.abs(vec - new_vecs[-1]))

    vecs = np.array(vecs)

    new_vecs = copy.deepcopy(vecs)
    new_vecs[0] = vecs[new_ixs[-1]]

    while len(new_ixs) < len(vecs):
        dists = np.sum(np.abs(vecs - new_vecs[len(new_ixs)-1]), axis=1)
        for ix in new_ixs:  # ignore previously found vectors
            dists[ix] = np.inf
        argmin = np.argmin(dists, axis=0)
        new_ixs.append(argmin)
        vec = vecs[argmin]
        try:
            new_vecs[len(new_ixs)-1] = copy.deepcopy(vec)
        except IndexError as e:
            print 'error: argmin =', argmin
            print 'error: len(new_ixs) =', len(new_ixs)
            print 'error: len(vecs) =', len(vecs)
            print 'error: len(new_vecs) =', len(new_vecs)
            raise IndexError(e)

    return np.array(new_ixs), np.array(new_vecs)


def mi_sliding_window(samples, n, ix1=(0,), ix2=(1,)):
    pdf = jointpdf.JointProbabilityMatrix(2, 3)  # parameters do not matter
    mis = []
    samples = np.array(samples)
    for startix in xrange(len(samples) - (n - 1)):
        pdf.estimate_from_data(samples[range(startix, startix+n)])
        mis.append(pdf.mutual_information(ix1, ix2))
    return mis