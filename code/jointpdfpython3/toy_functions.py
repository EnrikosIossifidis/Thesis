from operator import sub
from numba.cuda import test
import numpy as np
import warnings
import time
import itertools
import copy
import multiprocessing as mp

from scipy.optimize import minimize
from scipy.sparse import data
from JointProbabilityMatrix import JointProbabilityMatrix, ConditionalProbabilityMatrix,ConditionalProbabilities
from params_matrix import matrix2params_incremental,params2matrix_incremental
from measures import append_variables_using_state_transitions_table,synergistic_entropy_upper_bound,synergistic_information_naive

def append_synergistic_variables(self, num_synergistic_variables, initial_guess_summed_modulo=False, verbose=False,
                                    subject_variables=None, agnostic_about=None, num_repeats=3, minimize_method=None,
                                    tol_nonsyn_mi_frac=0.05, tol_agn_mi_frac=0.05):
    """
    Append <num_synergistic_variables> variables in such a way that they are agnostic about any individual
    existing variable (one of self.numvariables thus) but have maximum MI about the set of self.numvariables
    variables taken together.
    :param minimize_method:
    :param tol_nonsyn_mi_frac: set to None for computational speed
    :param tol_agn_mi_frac: set to None for computational speed
    :return:
    :param agnostic_about: a list of variable indices to which the new synergistic variable set should be
    agnostic (zero mutual information). This can be used to find a 'different', second SRV after having found
    already a first one. If already multiple SRVs have been found then you can choose between either being agnostic
    about these previous SRVs jointly (so also possibly any synergy among them), or being 'pairwise' agnostic
    to each individual one, in which case you can pass a list of lists, then I will compute the added cost for
    each sublist and sum it up.
    :param num_repeats:
    :param subject_variables: the list of variables the <num_synergistic_variables> should be synergistic about;
    then I think the remainder variables the <num_synergistic_variables> should be agnostic about. This way I can
    support adding many UMSRVs (maybe make a new function for that) which then already are orthogonal among themselves,
    meaning I do not have to do a separate orthogonalization for the MSRVs as in the paper's theoretical part.
    :param num_synergistic_variables:
    :param initial_guess_summed_modulo:
    :param verbose:
    :return:
    """

    if not agnostic_about is None:
        if len(agnostic_about) == 0:
            agnostic_about = None  # treat as if not supplied

    parameter_values_before = list(matrix2params_incremental(self))

    # a pdf with XORs as appended variables (often already MSRV for binary variables), good initial guess?
    pdf_with_srvs = self.copy()
    append_variables_using_state_transitions_table(pdf_with_srvs,
        state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_synergistic_variables)

    assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

    parameter_values_after = matrix2params_incremental(pdf_with_srvs)

    assert len(parameter_values_after) > len(parameter_values_before), 'should be additional free parameters'
    # a = np.random.random()
    # print(a)
    if np.random.random() < 0.1:  # reduce slowdown from this
        # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
        # have to optimize the latter part of parameter_values_after
        np.testing.assert_array_almost_equal(parameter_values_before,
                                                parameter_values_after[:len(parameter_values_before)])

    # this many parameters (each in [0,1]) must be optimized
    num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

    assert num_synergistic_variables == 0 or num_free_parameters > 0

    if initial_guess_summed_modulo:
        # note: this is xor for binary variables
        initial_guess = parameter_values_after[len(parameter_values_before):]
    else:
        initial_guess = np.random.random(num_free_parameters)

    if verbose:
        debug_pdf_with_srvs = pdf_with_srvs.copy()
        params2matrix_incremental(debug_pdf_with_srvs,list(parameter_values_before) + list(initial_guess))

        # store the synergistic information before the optimization procedure (after procedure should be higher...)
        debug_before_syninfo = synergistic_information_naive(debug_pdf_with_srvs,
            variables_SRV=range(self.numvariables, pdf_with_srvs.numvariables),
            variables_X=range(self.numvariables))

    assert len(initial_guess) == num_free_parameters

    pdf_with_srvs_for_optimization = pdf_with_srvs.copy()

    if not subject_variables is None:
        pdf_subjects_syns_only = pdf_with_srvs_for_optimization.marginalize_distribution(
            list(subject_variables) + range(len(pdf_with_srvs) - num_synergistic_variables, len(pdf_with_srvs)))

        pdf_subjects_syns_only = pdf_with_srvs_for_optimization.marginalize_distribution(
            list(subject_variables) + list(range(len(pdf_with_srvs) - num_synergistic_variables, len(pdf_with_srvs))))

        pdf_subjects_only = pdf_subjects_syns_only.marginalize_distribution(range(len(subject_variables)))
        if __debug__ and np.random.random() < 0.01:
            debug_pdf_subjects_only = pdf_with_srvs.marginalize_distribution(subject_variables)

            assert debug_pdf_subjects_only == pdf_subjects_only

        num_free_parameters_synonly = len(matrix2params_incremental(pdf_subjects_syns_only)) \
                                        - len(matrix2params_incremental(pdf_subjects_only))

        parameter_values_static = matrix2params_incremental(pdf_subjects_only)

        initial_guess = np.random.random(num_free_parameters_synonly)

        # pdf_subjects_syns_only should be the new object that fitness_func operates on, instead of
        # pdf_with_srvs_for_optimization
    else:
        # already like this, so simple renaming
        pdf_subjects_syns_only = pdf_with_srvs_for_optimization

        parameter_values_static = parameter_values_before

        num_free_parameters_synonly = num_free_parameters

        # subject_variables = range(len(self))

    upper_bound_synergistic_information = synergistic_entropy_upper_bound(self,subject_variables)
    if not agnostic_about is None:
        # upper_bound_agnostic_information is only used to normalize the cost term for non-zero MI with
        # the agnostic_variables (evidently a SRV is sought that has zero MI with these)
        if np.ndim(agnostic_about) == 1:
            upper_bound_agnostic_information = self.entropy(agnostic_about)
        elif np.ndim(agnostic_about) == 2:
            # in this case the cost term is the sum of MIs of the SRV with the sublists, so max cost term is this..
            upper_bound_agnostic_information = sum([self.entropy(ai) for ai in agnostic_about])
    else:
        upper_bound_agnostic_information = 0  # should not even be used...

    # todo: should lower the upper bounds if the max possible entropy of the SRVs is less...

    assert upper_bound_synergistic_information != 0.0, 'can never find any SRV!'

    # in the new self, these indices will identify the synergistic variables that will be added
    synergistic_variables = range(len(self), len(self) + num_synergistic_variables)

    # todo: shouldn't the cost terms in this function not be squared for better convergence?
    def cost_func_subjects_only(free_params, parameter_values_before, extra_cost_rel_error=True):
        """
        This cost function searches for a Pr(S,Y,A,X) such that X is synergistic about S (subject_variables) only.
        This fitness function also taxes any correlation between X and A (agnostic_variables), but does not care
        about the relation between X and Y.
        :param free_params:
        :param parameter_values_before:
        :return:
        """
        assert len(free_params) == num_free_parameters_synonly

        if min(free_params) < -0.00001 or max(free_params) > 1.00001:
            warnings.warn('scipy\'s minimize() is violating the parameter bounds 0...1 I give it: '
                            + str(free_params))

            # high cost for invalid parameter values
            # note: maximum cost normally from this function is about 2.0
            return 10.0 + 100.0 * np.sum([p - 1.0 for p in free_params if p > 1.0]
                                            + [np.abs(p) for p in free_params if p < 0.0])

        # assert max(free_params) <= 1.00001, \
        #     'scipy\'s minimize() is violating the parameter bounds 0...1 I give it: ' + str(free_params)

        free_params = [min(max(fp, 0.0), 1.0) for fp in free_params]  # clip small roundoff errors

        params2matrix_incremental(pdf_subjects_syns_only,list(parameter_values_before) + list(free_params))

        # this if-block will add cost for the estimated amount of synergy induced by the proposed parameters,
        # and possible also a cost term for the ratio of synergistic versus non-synergistic info as extra
        if not subject_variables is None:
            assert pdf_subjects_syns_only.numvariables == len(subject_variables) + num_synergistic_variables

            # this can be considered to be in range [0,1] although particularly bad solutions can go >1
            if not extra_cost_rel_error:
                cost = (upper_bound_synergistic_information - synergistic_information_naive(pdf_subjects_syns_only,
                    variables_SRV=range(len(subject_variables), len(pdf_subjects_syns_only)),
                    variables_X=range(len(subject_variables)))) / upper_bound_synergistic_information
            else:
                tot_mi = pdf_subjects_syns_only.mutual_information(
                    range(len(subject_variables), len(pdf_subjects_syns_only)),
                    range(len(subject_variables)))

                indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                        range(len(subject_variables),
                                                                                len(pdf_subjects_syns_only)))
                                for var in range(len(subject_variables))]

                syninfo_naive = tot_mi - sum(indiv_mis)

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                cost = (upper_bound_synergistic_information - syninfo_naive) \
                        / upper_bound_synergistic_information

                # add an extra cost term for the fraction of 'individual' information versus the total information
                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if tot_mi != 0:
                    cost += sum(indiv_mis) / tot_mi
                else:
                    cost += sum(indiv_mis)
        else:
            assert pdf_subjects_syns_only.numvariables == len(self) + num_synergistic_variables

            if not extra_cost_rel_error:
                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                cost = (upper_bound_synergistic_information - synergistic_information_naive(pdf_subjects_syns_only,
                    variables_SRV=range(len(self), len(pdf_subjects_syns_only)),
                    variables_X=range(len(self)))) / upper_bound_synergistic_information
            else:
                tot_mi = pdf_subjects_syns_only.mutual_information(
                    range(len(self), len(pdf_subjects_syns_only)),
                    range(len(self)))

                indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                        range(len(self),
                                                                                len(pdf_subjects_syns_only)))
                                for var in range(len(self))]

                syninfo_naive = tot_mi - sum(indiv_mis)

                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                cost = (upper_bound_synergistic_information - syninfo_naive) \
                        / upper_bound_synergistic_information

                # add an extra cost term for the fraction of 'individual' information versus the total information
                # this can be considered to be in range [0,1] although particularly bad solutions can go >1
                if tot_mi != 0:
                    cost += sum(indiv_mis) / tot_mi
                else:
                    cost += sum(indiv_mis)

        # this if-block will add a cost term for not being agnostic to given variables, usually (a) previous SRV(s)
        if not agnostic_about is None:
            assert not subject_variables is None, 'how can all variables be subject_variables and you still want' \
                                                    ' to be agnostic about certain (other) variables? (if you did' \
                                                    ' not specify subject_variables, do so.)'

            # make a conditional distribution of the synergistic variables conditioned on the subject variables
            # so that I can easily make a new joint pdf object with them and quantify this extra cost for the
            # agnostic constraint
            cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                range(len(subject_variables))
            )

            assert type(cond_pdf_syns_on_subjects) == dict \
                    or isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)

            pdf_with_srvs_for_agnostic = self.copy()
            pdf_with_srvs_for_agnostic.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                        subject_variables)

            if np.ndim(agnostic_about) == 1:
                # note: cost term for agnostic is in [0,1]
                cost += (pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agnostic_about)) \
                        / upper_bound_agnostic_information
            else:
                assert np.ndim(agnostic_about) == 2, 'expected list of lists, not more... made a mistake?'

                assert False, 'I don\'t think this case should happen, according to my 2017 paper should be just ' \
                                'I(X:A) so ndim==1'

                for agn_i in agnostic_about:
                    # note: total cost term for agnostic is in [0,1]
                    cost += (1.0 / float(len(agnostic_about))) * \
                            pdf_with_srvs_for_agnostic.mutual_information(synergistic_variables, agn_i) \
                            / upper_bound_agnostic_information

        assert np.isscalar(cost)
        assert np.isfinite(cost)

        return float(cost)

    param_vectors_trace = []

    # these options are altered mainly to try to lower the computation time, which is considerable.
    minimize_options = {'ftol': 1e-6}

    if True:
    # if num_repeats == 1:
    #     optres = minimize(cost_func_subjects_only, initial_guess, bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
    #                       callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
    #                       args=(parameter_values_static,), method=minimize_method, options=minimize_options)
    # else:
        optres_list = []

        for ix in range(num_repeats):
            optres_ix = minimize(cost_func_subjects_only,
                                    np.random.random(num_free_parameters_synonly) if ix > 0 else initial_guess,
                                    bounds=[(0.0, 1.0)]*num_free_parameters_synonly,
                                    callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                                    args=(parameter_values_static,), method=minimize_method, options=minimize_options)

            if verbose:
                print('note: finished a repeat. success=' + str(optres_ix.success) + ', cost=' \
                        + str(optres_ix.fun))

            if not tol_nonsyn_mi_frac is None:
                params2matrix_incremental(pdf_subjects_syns_only,list(parameter_values_static) + list(optres_ix.x))

                if subject_variables is None:
                    print('debug: will set subject_variables=%s' % (range(len(self))))
                    subject_variables = range(len(self))

                assert not pdf_subjects_syns_only is None

                tot_mi = pdf_subjects_syns_only.mutual_information(
                            range(len(subject_variables), len(pdf_subjects_syns_only)),
                            range(len(subject_variables)))

                indiv_mis = [pdf_subjects_syns_only.mutual_information([var],
                                                                        range(len(subject_variables),
                                                                                len(pdf_subjects_syns_only)))
                                for var in range(len(subject_variables))]

                if sum(indiv_mis) / float(tot_mi) > tol_nonsyn_mi_frac:
                    if verbose:
                        print('debug: in iteration %s I found an SRV but with total MI %s and indiv. MIs %s it ' \
                                'violates the tol_nonsyn_mi_frac=%s' % (ix, tot_mi, indiv_mis, tol_nonsyn_mi_frac))

                    continue  # don't add this to the list of solutions

            if not tol_agn_mi_frac is None and not agnostic_about is None:
                if len(agnostic_about) > 0:
                    # note: could reuse the one above, saves a bit of computation
                    params2matrix_incremental(pdf_subjects_syns_only,list(parameter_values_static)
                                                                        + list(optres_ix.x))

                    cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
                                                    range(len(subject_variables))
                                                )

                    # I also need the agnostic variables, which are not in pdf_subjects_syns_only, so construct
                    # the would-be final result (the original pdf with the addition of the newly found SRV)
                    pdf_with_srvs = self.copy()
                    pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                                    given_variables=subject_variables)

                    agn_mi = pdf_with_srvs.mutual_information(agnostic_about, range(len(self), len(pdf_with_srvs)))
                    agn_ent = self.entropy(agnostic_about)

                    if agn_mi / agn_ent > tol_agn_mi_frac:
                        if verbose:
                            print('debug: in iteration %s I found an SRV but with agn_mi=%s and agn_ent=%s it ' \
                                    'violates the tol_agn_mi_frac=%s' % (ix, agn_mi, agn_ent, tol_nonsyn_mi_frac))

                        continue  # don't add this to the list of solutions

            optres_list.append(optres_ix)

        if verbose and __debug__:
            print('debug: num_repeats=' + str(num_repeats) + ', all cost values were: ' \
                    + str([resi.fun for resi in optres_list]))
            print('debug: successes =', [resi.success for resi in optres_list])

        optres_list = [resi for resi in optres_list if resi.success]  # filter out the unsuccessful optimizations

        if len(optres_list) == 0:
            raise UserWarning('all ' + str(num_repeats) + ' optimizations using minimize() failed...?!')

        costvals = [res.fun for res in optres_list]
        min_cost = min(costvals)
        optres_ix = costvals.index(min_cost)

        assert optres_ix >= 0 and optres_ix < len(optres_list)

        optres = optres_list[optres_ix]

    if subject_variables is None:
        assert len(optres.x) == num_free_parameters
    else:
        assert len(optres.x) == num_free_parameters_synonly

    assert max(optres.x) <= 1.0000001, 'parameter bound significantly violated, ' + str(max(optres.x))
    assert min(optres.x) >= -0.0000001, 'parameter bound significantly violated, ' + str(min(optres.x))

    # todo: reuse the .append_optimized_variables (or so) instead, passing the cost function only? would also
    # validate that function.

    # clip to valid range
    optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

    # optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)
    # pdf_with_srvs.params2matrix_incremental(optimal_parameters_joint_pdf)

    assert len(matrix2params_incremental(pdf_subjects_syns_only)) == len(parameter_values_static) + len(optres.x)

    params2matrix_incremental(pdf_subjects_syns_only,list(parameter_values_static) + list(optres.x))

    if not subject_variables is None:
        cond_pdf_syns_on_subjects = pdf_subjects_syns_only.conditional_probability_distributions(
            range(len(subject_variables))
        )

        assert isinstance(cond_pdf_syns_on_subjects, ConditionalProbabilities)
        assert cond_pdf_syns_on_subjects.num_given_variables() > 0, 'conditioned on 0 variables?'

        # if this hits then something is unintuitive with conditioning on variables...
        assert cond_pdf_syns_on_subjects.num_given_variables() == len(subject_variables)

        pdf_with_srvs = self.copy()
        pdf_with_srvs.append_variables_using_conditional_distributions(cond_pdf_syns_on_subjects,
                                                                        given_variables=subject_variables)
    else:
        pdf_with_srvs = pdf_subjects_syns_only  # all variables are subject

    assert pdf_with_srvs.numvariables == self.numvariables + num_synergistic_variables

    if verbose:
        parameter_values_after2 = matrix2params_incremental(pdf_with_srvs)

        assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'

        if not 1.0 in parameter_values_after2 and not 0.0 in parameter_values_after2:
            # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
            # have to optimize the latter part of parameter_values_after
            np.testing.assert_array_almost_equal(parameter_values_before,
                                                    parameter_values_after2[:len(parameter_values_before)])
            np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                    optres.x)
        else:
            # it can happen that some parameters are 'useless' in the sense that they defined conditional
            # probabilities in the case where the prior (that is conditioned upon) has zero probability. The
            # resulting pdf is then always the same, no matter this parameter value. This can only happen if
            # there is a 0 or 1 in the parameter list, (sufficient but not necessary) so skip the test then...
            pass

        # store the synergistic information before the optimization procedure (after procedure should be higher...)
        debug_after_syninfo = pdf_with_srvs.synergistic_information_naive(variables_SRV=range(self.numvariables,
                                                                                        pdf_with_srvs.numvariables),
                                                                            variables_X=range(self.numvariables))

        if verbose:
            print('debug: append_synergistic_variables: I started from synergistic information =', \
                'at initial guess. After optimization it became', debug_after_syninfo, \
                '(should be higher). Optimal params:', \
                parameter_values_after2[len(parameter_values_before):])

    self.duplicate(pdf_with_srvs)

def append_independent_variables(self, joint_pdf):
    """
    :type joint_pdf: JointProbabilityMatrix
    """
    assert not type(joint_pdf) in (int, float, str), 'should pass a JointProbabilityMatrix object'

    self.append_variables_using_conditional_distributions(ConditionalProbabilityMatrix({(): joint_pdf}))

    def append_variables_with_target_mi(self, num_appended_variables, target_mi, relevant_variables='all',
                                        verbose=False, num_repeats=None):

        # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

        if relevant_variables in ('all', 'auto'):
            relevant_variables = range(len(self))
        else:
            assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
            assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'

        if target_mi == 0.0:
            raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
                              ' Therefore you should call append_independent_variables instead and specify explicitly'
                              ' which PDFs you want to add independently.')

        parameter_values_before = list(self.matrix2params_incremental())

        pdf_new = self.copy()
        pdf_new.append_variables(num_appended_variables)

        assert pdf_new.numvariables == self.numvariables + num_appended_variables

        parameter_values_after = pdf_new.matrix2params_incremental()

        # this many parameters (each in [0,1]) must be optimized
        num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

        def cost_func_target_mi(free_params, parameter_values_before):

            assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
            # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

            pdf_new.params2matrix_incremental(list(parameter_values_before) + list(free_params))

            mi = pdf_new.mutual_information(relevant_variables, range(len(self), len(pdf_new)))

            return np.power((target_mi - mi) / target_mi, 2)

        self.append_optimized_variables(num_appended_variables, cost_func=cost_func_target_mi,
                                        initial_guess=np.random.random(num_free_parameters),
                                        verbose=verbose, num_repeats=num_repeats)

        return  # nothing, in-place

def append_variables_with_target_mi(self, num_appended_variables, target_mi, relevant_variables='all',
                                    verbose=False, num_repeats=None):

    # input_variables = [d for d in xrange(self.numvariables) if not d in output_variables]

    if relevant_variables in ('all', 'auto'):
        relevant_variables = range(len(self))
    else:
        assert len(relevant_variables) <= len(self), 'cannot be relative to more variables than I originally had'
        assert max(relevant_variables) <= len(self) - 1, 'relative to new variables...?? should not be'

    if target_mi == 0.0:
        raise UserWarning('you set target_mi but this is ill-defined: any independent variable(s) will do.'
                            ' Therefore you should call append_independent_variables instead and specify explicitly'
                            ' which PDFs you want to add independently.')

    parameter_values_before = list(matrix2params_incremental(self))

    pdf_new = self.copy()
    pdf_new.append_variables(num_appended_variables)

    assert pdf_new.numvariables == self.numvariables + num_appended_variables

    parameter_values_after = matrix2params_incremental(pdf_new)

    # this many parameters (each in [0,1]) must be optimized
    num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

    def cost_func_target_mi(free_params, parameter_values_before):

        assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
        # assert np.all(np.isfinite(parameter_values_before))  # todo: remove

        params2matrix_incremental(pdf_new,list(parameter_values_before) + list(free_params))

        mi = pdf_new.mutual_information(relevant_variables, range(len(self), len(pdf_new)))

        return np.power((target_mi - mi) / target_mi, 2)

    append_optimized_variables(self,num_appended_variables, cost_func=cost_func_target_mi,
                                    initial_guess=np.random.random(num_free_parameters),
                                    verbose=verbose, num_repeats=num_repeats)

    return  # nothing, in-place

def append_optimized_variables(self, num_appended_variables, cost_func, initial_guess=None, verbose=True,
                                num_repeats=None):
    """
    Append variables in such a way that their conditional pdf with the existing variables is optimized in some
    sense, for instance they can be synergistic (append_synergistic_variables) or orthogonalized
    (append_orthogonalized_variables). Use the cost_func to determine the relation between the new appended
    variables and the pre-existing ones.
    :param num_appended_variables:
    :param cost_func: a function cost_func(free_params, parameter_values_before) which returns a float.
    The parameter set list(parameter_values_before) + list(free_params) defines a joint pdf of the appended
    variables together with the pre-existing ones, and free_params by itself defines completely the conditional
    pdf of the new variables given the previous. Use params2matrix_incremental to construct a joint pdf from the
    parameters and evaluate whatever you need, and then return a float. The higher the return value of cost_func
    the more desirable the joint pdf induced by the parameter set list(parameter_values_before) + list(free_params).
    :param initial_guess: initial guess for 'free_params' where you think cost_func will return a relatively
    low value. It can also be None, in which case a random point in parameter space will be chosen. It can also
    be an integer value like 10, in which case 10 optimizations will be run each starting from a random point
    in parameter space, and the best solution is selected.
    :param verbose:
    :rtype: scipy.optimize.OptimizeResult
    """

    # these parameters should be unchanged and the first set of parameters of the resulting pdf_new
    parameter_values_before = list(matrix2params_incremental(self))

    assert min(parameter_values_before) >= -0.00000001, \
        'minimum of %s is < 0, should not be.' % parameter_values_before
    assert max(parameter_values_before) <= 1.00000001, \
        'minimum of %s is < 0, should not be.' % parameter_values_before

    if __debug__:
        debug_params_before = copy.deepcopy(parameter_values_before)

    # a pdf with XORs as appended variables (often already MSRV for binary variables), good initial guess?
    # note: does not really matter how I set the pdf of this new pdf, as long as it has the correct number of
    # paarameters for optimization below
    pdf_new = self.copy()
    append_variables_using_state_transitions_table(pdf_new,
        state_transitions=lambda vals, mv: [int(np.mod(np.sum(vals), mv))]*num_appended_variables)

    assert pdf_new.numvariables == self.numvariables + num_appended_variables

    parameter_values_after = matrix2params_incremental(pdf_new)

    assert num_appended_variables > 0, 'makes no sense to add 0 variables'
    assert len(parameter_values_after) > len(parameter_values_before), 'should be >0 free parameters to optimize?'
    # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
    # have to optimize the latter part of parameter_values_after
    np.testing.assert_array_almost_equal(parameter_values_before,
                                            parameter_values_after[:len(parameter_values_before)])

    # this many parameters (each in [0,1]) must be optimized
    num_free_parameters = len(parameter_values_after) - len(parameter_values_before)

    assert num_appended_variables == 0 or num_free_parameters > 0

    # if initial_guess is None:
    #     initial_guess = np.random.random(num_free_parameters)  # start from random point in parameter space

    param_vectors_trace = []  # storing the parameter vectors visited by the minimize() function

    if num_repeats is None:
        if type(initial_guess) == int:
            num_repeats = int(initial_guess)
            initial_guess = None

            assert num_repeats > 0, 'makes no sense to optimize zero times?'
        else:
            num_repeats = 1

    optres = None

    def cost_func_wrapper(free_params, parameter_values_before):
        # note: jezus CHRIST not only does minimize() ignore the bounds I give it, it also suggests [nan, ...]!
        # assert np.all(np.isfinite(free_params)), 'looking for bug 23142'
        if not np.all(np.isfinite(free_params)):
            return np.power(np.sum(np.isfinite(free_params)), 2) * 10.
        else:
            clipped_free_params = np.max([np.min([free_params, np.ones(np.shape(free_params))], axis=0),
                                            np.zeros(np.shape(free_params))], axis=0)
            # penalize going out of bounds
            extra_cost = np.power(np.sum(np.abs(np.subtract(free_params, clipped_free_params))), 2)
            return cost_func(clipped_free_params, parameter_values_before) + extra_cost

    for rep in range(num_repeats):
        if initial_guess is None:
            initial_guess_i = np.random.random(num_free_parameters)  # start from random point in parameter space
        else:
            initial_guess_i = initial_guess  # always start from supplied point in parameter space

        assert len(initial_guess_i) == num_free_parameters
        assert np.all(np.isfinite(initial_guess_i)), 'looking for bug 55142'
        assert np.all(np.isfinite(parameter_values_before)), 'looking for bug 44142'

        if verbose:
            print('debug: starting minimize() #' + str(rep) \
                    + ' at params=' + str(initial_guess_i) + ' at cost_func=' \
                    + str(cost_func_wrapper(initial_guess_i, parameter_values_before)))

        optres_i = minimize(cost_func_wrapper,
                            initial_guess_i, bounds=[(0.0, 1.0)]*num_free_parameters,
                            # callback=(lambda xv: param_vectors_trace.append(list(xv))) if verbose else None,
                            args=(parameter_values_before,))

        if optres_i.success:
            if verbose:
                print('debug: successfully ended minimize() #' + str(rep) \
                        + ' at params=' + str(optres_i.x) + ' at cost_func=' \
                        + str(optres_i.fun))

            if optres is None:
                optres = optres_i
            elif optres.fun > optres_i.fun:
                optres = optres_i
            else:
                pass  # this solution is worse than before, so do not change optres

    if optres is None:
        # could never find a good solution, in all <num_repeats> attempts
        raise UserWarning('always failed to successfully optimize: increase num_repeats')

    assert len(optres.x) == num_free_parameters
    assert max(optres.x) <= 1.0001, 'parameter bound significantly violated: ' + str(optres.x)
    assert min(optres.x) >= -0.0001, 'parameter bound significantly violated: ' + str(optres.x)

    # clip the parameters within the allowed bounds
    optres.x = [min(max(xi, 0.0), 1.0) for xi in optres.x]

    optimal_parameters_joint_pdf = list(parameter_values_before) + list(optres.x)

    assert min(optimal_parameters_joint_pdf) >= 0.0, \
        'minimum of %s is < 0, should not be.' % optimal_parameters_joint_pdf
    assert min(optimal_parameters_joint_pdf) <= 1.0, \
        'minimum of %s is > 1, should not be.' % optimal_parameters_joint_pdf
    assert min(parameter_values_before) >= 0.0, \
        'minimum of %s is < 0, should not be.' % parameter_values_before
    assert min(parameter_values_before) <= 1.0, \
        'minimum of %s is > 1, should not be.' % parameter_values_before

    params2matrix_incremental(pdf_new,optimal_parameters_joint_pdf)

    assert len(pdf_new) == len(self) + num_appended_variables

    if __debug__:
        parameter_values_after2 = matrix2params_incremental(pdf_new)

        assert len(parameter_values_after2) > len(parameter_values_before), 'should be additional free parameters'
        # see if the first part of the new parameters is exactly the old parameters, so that I know that I only
        # have to optimize the latter part of parameter_values_after
        np.testing.assert_array_almost_equal(parameter_values_before,
                                                parameter_values_after2[:len(parameter_values_before)])
        # note: for the if see the story in params2matrix_incremental()
        if not (0.000001 >= min(self.scalars_up_to_level(parameter_values_after2)) or \
                        0.99999 <= max(self.scalars_up_to_level(parameter_values_after2))):
            try:
                np.testing.assert_array_almost_equal(parameter_values_after2[len(parameter_values_before):],
                                                        optres.x)
            except AssertionError as e:
                # are they supposed to be equal, but in different order?
                print('debug: sum params after 1 =', np.sum(parameter_values_after2[len(parameter_values_before):]))
                print('debug: sum params after 2 =', optres.x)
                print('debug: parameter_values_before (which IS equal and correct) =', parameter_values_before)
                # does this one below have a 1 or 0 in it? because then the error could be caused by the story in
                # params2matrix_incremental()
                print('debug: parameter_values_after2 =', parameter_values_after2)

                raise AssertionError(e)
    if __debug__:
        # unchanged, not accidentally changed by passing it as reference? looking for bug
        np.testing.assert_array_almost_equal(debug_params_before, parameter_values_before)

    self.duplicate(pdf_new)

    return optres
