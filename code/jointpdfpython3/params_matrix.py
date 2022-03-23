import numpy as np
from numba import njit
from numba import vectorize
from .JointProbabilityMatrix import JointProbabilityMatrix
from .utils import *

def matrix2params(d):
    vector_probs = d.joint_probabilities.joint_probabilities
    remaining_prob_mass = 1.0

    parameters = [-1.0]*(len(vector_probs) - 1)

    for pix in range(len(parameters)):
        if remaining_prob_mass > 0:
            parameters[pix] = vector_probs[pix] / remaining_prob_mass

        elif remaining_prob_mass == 0:
            parameters[pix] = 0
        else:
            if not remaining_prob_mass > -0.000001:
                print('debug: remaining_prob_mass =', remaining_prob_mass)

                # todo: if just due to floating operation error, so small, then clip to zero and go on?
                raise ValueError('remaining_prob_mass = ' + str(remaining_prob_mass)
                                 + ' < 0, which should not happen?')
            else:
                # seems that it was intended to reach zero but due to floating operation roundoff it got just
                # slightly under. Clip to 0 will do the trick.
                remaining_prob_mass = 0.0  # clip to zero, so it will stay that way

            parameters[pix] = 0  # does not matter

            assert -0.1 <= parameters[pix] <= 1.1, \
                'parameters should be in [0, 1]: ' + str(parameters[pix]) \
                + ', sum probs = ' + str(np.sum(vector_probs))

        # sometimes this happens I think due to rounding errors, but when I sum the probabilities they
        # still seem to sum to exactly 1.0 so probably is due to some parameters being 0 or 1, so clip here
        parameters[pix] = max(min(parameters[pix], 1.0), 0.0)

        # parameters[pix] = min(max(parameters[pix], 0.0), 1.0)
        remaining_prob_mass -= remaining_prob_mass * parameters[pix]
    return parameters

# Works for XOR example, but not random distributions
def matrix2params_incremental(self, return_flattened=True, verbose=False):
    if self.numvariables > 1:
        # get the marginal pdf for the first variable
        pdf1 = self.marginalize_distribution([0])
        # first sequence of parameters, rest is added below here
        parameters = matrix2params(pdf1)

        pdf_conds = self.conditional_probability_distributions([0])
        # assert len(pdf_conds) == self.numvalues, 'should be one pdf for each value of first variable'
        for val in range(self.numvalues):
            pdf_cond = pdf_conds[tuple([val])]
            added_params = matrix2params_incremental(pdf_cond,return_flattened=False, verbose=verbose)
            if verbose:
                print('debug: matrix2params_incremental: recursed: for val=' + str(val) + ' I got added_params=' \
                      + str(added_params) + '.')
                print('debug: matrix2params_incremental: old parameters =', parameters)

            # instead of returning a flat list of parameters I make it nested, so that the structure (e.g. number of
            # variables and number of values) can be inferred, and also hopefully it can be inferred to which
            # variable which parameters belong.
            # CHANGE123
            parameters.append(added_params)
            # print(added_params,val)
            if verbose:
                print('debug: matrix2params_incremental: new parameters =', parameters)

        if return_flattened:
            # flatten the tree structure to a list of scalars, which is sorted on the variable id
            parameters = self.scalars_up_to_level(parameters)

        return parameters
    elif self.numvariables == 1:
        return matrix2params(self)
    else:
        raise ValueError('no parameters for 0 variables')

_debug_params2matrix = False  # internal variable, used to debug a debug statement, can be removed in a while


@vectorize(['float64(float64, float64, float64)'])
def truncate(a, amin, amax):
    if a < amin:
        a = amin
    elif a > amax:
        a = amax
    return a

@njit
def numba_p2m(values, variables, parameters,newshape):
    vector_probs = [-1.0]*(np.power(values, variables))
    remaining_prob_mass = 1.0

    for pix in range(len(parameters)):
        # clip the parameter to the allowed range. If a rounding error is fixed by this in the parameters then
        # possibly a rounding error will appear in the probabilities?... Not sure though
        parameters[pix] = min(max(parameters[pix], 0.0), 1.0)
        vector_probs[pix] = remaining_prob_mass * parameters[pix]
        remaining_prob_mass = remaining_prob_mass * (1.0 - parameters[pix])
    
    vector_probs[-1] = remaining_prob_mass 
    vector_probs = np.asarray(vector_probs)
    vector_probs = truncate(vector_probs, 0.0, 1.0)
    vector_probs.reshape(newshape)
    return vector_probs

def vector2matrix(self, list_probs):
    # np.testing.assert_almost_equal(np.sum(list_probs), 1.0)

    # assert np.ndim(list_probs) == 1
    self.joint_probabilities.reset(np.reshape(list_probs, [self.numvalues]*self.numvariables))

    self.clip_all_probabilities()

def params2matrix(self, parameters):
    # assert len(parameters) == np.power(self.numvalues, self.numvariables) - 1

    vector_probs = [-1.0]*(np.power(self.numvalues, self.numvariables))

    remaining_prob_mass = 1.0

    for pix in range(len(parameters)):
        # note: small rounding errors will be fixed below by clipping
        # assert -0.000001 <= parameters[pix] <= 1.000001, 'parameters should be in [0, 1]: ' + str(parameters[pix])

        # clip the parameter to the allowed range. If a rounding error is fixed by this in the parameters then
        # possibly a rounding error will appear in the probabilities?... Not sure though
        parameters[pix] = min(max(parameters[pix], 0.0), 1.0)

        vector_probs[pix] = remaining_prob_mass * parameters[pix]

        remaining_prob_mass = remaining_prob_mass * (1.0 - parameters[pix])

    # assert vector_probs[-1] < 0.0, 'should still be unset by the above loop'

    # last parameter is irrelevant, must always be 1.0 is also a way to look at it
    vector_probs[-1] = remaining_prob_mass

    if __debug__:
        np.testing.assert_almost_equal(np.sum(vector_probs), 1.0)

    vector2matrix(self,vector_probs)
    
def params2matrix_incremental(self, parameters,func='normal'):
    """
    Takes in a row of floats in range [0.0, 1.0] and changes <self> to a new PDF which is characterized by the
    parameters. Benefit: np.random.random(M**N - 1) results in an unbiased sample of PDF, wnere M is numvalues
    and N is numvariables.
    :param parameters: list of floats, length equal to what matrix2params_incrmental() returns (M**N - 1)
    :type parameters: list of float
    """
    if __debug__:
        # store the original provided list of scalars
        original_parameters = list(parameters)

    # I suspect that both a tree-like input and a list of scalars should work... (add to unit test?)
    if np.all(list(map(np.isscalar, parameters))):
        
        assert min(parameters) > -0.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                          + str(parameters)
        assert min(parameters) < 1.0001, 'parameter(s) significantly out of allowed bounds [0,1]: ' \
                                         + str(parameters)

        # clip each parameter value to the allowed range. above I check already whether the error is not too large
        parameters = [min(max(pi, 0.0), 1.0) for pi in parameters]

        parameters = imbalanced_tree_from_scalars(self,parameters, self.numvalues)

        # verify that the procedure to make the tree out of the list of scalars is reversible and correct
        # (looking for bug)
        if __debug__ and _debug_params2matrix:
            original_parameters2 = self.scalars_up_to_level(parameters)

            np.testing.assert_array_almost_equal(original_parameters, original_parameters2)

    if self.numvariables > 1:
        # first (numvalues - 1) values in the parameters tree structure should be scalars, as they will be used
        # to make the first variable's marginal distribution
        assert np.all(map(np.isscalar, parameters[:(self.numvalues - 1)]))

        ### start already by setting the pdf of the first variable...
        pdf_1 = JointProbabilityMatrix(1, self.numvalues)

        if not func == 'normal':
            pdf_1.joint_probabilities.joint_probabilities = numba_p2m(pdf_1.numvalues,pdf_1.numvariables,parameters[:(len(pdf_1.joint_probabilities.flatiter()) - 1)],
                                        (pdf_1.numvalues,)*pdf_1.numvariables)
        else:
            params2matrix(pdf_1,parameters[:(len(pdf_1.joint_probabilities.flatiter()) - 1)])

        assert (len(pdf_1.joint_probabilities.flatiter()) - 1) == (self.numvalues - 1), 'assumption directly above'

        assert len(pdf_1.joint_probabilities.flatiter()) == self.numvalues

        assert len(flatten(parameters)) == len(self.joint_probabilities.flatiter()) - 1, \
            'more or fewer parameters than needed: ' \
            'need ' + str(len(self.joint_probabilities.flatiter()) - 1) + ', got ' + str(len(flatten(parameters))) \
            + '; #vars, #vals = ' + str(self.numvariables) + ', ' + str(self.numvalues)

        if __debug__ and _debug_params2matrix:
            # remove this (expensive) check after it seems to work a few times?
            # note: for the conditions of no 1.0 or 0.0 prior probs, see the note in params2matrix_incremental
            if not 0.0 in matrix2params(pdf_1) and not 1.0 in matrix2params(pdf_1):
                np.testing.assert_array_almost_equal(matrix2params(pdf_1),
                                                     self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))
                np.testing.assert_array_almost_equal(matrix2params_incremental(pdf_1),
                                                     self.scalars_up_to_level(parameters[:(self.numvalues - 1)]))

        # remove the used parameters from the list
        parameters = parameters[(len(pdf_1.joint_probabilities.flatiter()) - 1):]
        assert len(parameters) == self.numvalues  # one subtree per conditional pdf

        pdf_conds = dict()

        ### now add other variables...

        for val in range(self.numvalues):
            # set this conditional pdf recursively as defined by the next sequence of parameters
            pdf_cond = JointProbabilityMatrix(self.numvariables - 1, self.numvalues)

            # note: parameters[0] is a sublist
            assert not np.isscalar(parameters[0])

            assert not np.isscalar(parameters[0])

            # todo: changing the parameters list is not necessary, maybe faster if not?

            # pdf_cond.params2matrix_incremental(parameters[:(len(pdf_cond.joint_probabilities.flatiter()) - 1)])
            params2matrix_incremental(pdf_cond,parameters[0])

            # conditional pdf should have the same set of parameters as the ones I used to create it
            # (todo: remove this expensive check if it seems to work for  while)
            if __debug__:  # seemed to work for a long time...
                try:
                    if np.random.randint(20) == 0:
                        np.testing.assert_array_almost_equal(matrix2params_incremental(pdf_cond),
                                                             self.scalars_up_to_level(parameters[0]))
                except AssertionError as e:
                    # print 'debug: parameters[0] =', parameters[0]
                    # print 'debug: len(pdf_cond) =', len(pdf_cond)
                    # print 'debug: pdf_cond.joint_probabilities =', pdf_cond.joint_probabilities

                    pdf_1_duplicate1 = pdf_cond.copy()
                    pdf_1_duplicate2 = pdf_cond.copy()

                    pdf_1_duplicate1._debug_params2matrix = False  # prevent endless recursion
                    pdf_1_duplicate2._debug_params2matrix = False  # prevent endless recursion

                    params2matrix_incremental(pdf_1_duplicate1,self.scalars_up_to_level(parameters[0]))
                    params2matrix_incremental(pdf_1_duplicate2,matrix2params_incremental(pdf_cond))

                    pdf_1_duplicate1._debug_params2matrix = True
                    pdf_1_duplicate2._debug_params2matrix = True

                    assert pdf_1_duplicate1 == pdf_cond
                    assert pdf_1_duplicate2 == pdf_cond

                    del pdf_1_duplicate1, pdf_1_duplicate2

                    # note: the cause seems to be as follows. If you provide the parameters e.g.
                    # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                    # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                    # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                    # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                    # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                    # parameter values I may very well get a different parameter list from that pdf, even though
                    # the pdf built is the one intended. I don't see a way around this because even if this class
                    # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                    # but then still a user or optimization procedure can provide any list of parameters, so
                    # then also the uniformized parameter list will differ from the user-supplied.

                    # raise AssertionError(e)

                    # later add check. If this check fails then for sure there is something wrong. See also the
                    # original check below.
                    assert 0.0 in self.scalars_up_to_level(parameters) or \
                           1.0 in self.scalars_up_to_level(parameters), 'see story above. ' \
                                                                        'self.scalars_up_to_level(parameters) = ' \
                                                                        + str(self.scalars_up_to_level(parameters))

                    # original check. This check failed once, but the idea was to see if there are 0s or 1s in the
                    # prior probability distribution, which precedes the conditional probability distribution for which
                    # apparently the identifying parameter values have changed. But maybe I am wrong in that
                    # parameters[0] is not the prior only, and some prior prob. information is in all of parameters,
                    # I am not sure anymore so I added the above check to see whether that one is hit instead
                    # of this one (so above check is of course more stringent than this one....)
                    # assert 0.0 in self.scalars_up_to_level(parameters[0]) or \
                    #        1.0 in self.scalars_up_to_level(parameters[0]), 'see story above. ' \
                    #                                                        'self.scalars_up_to_level(parameters[0]) = ' \
                    #                                                        + str(self.scalars_up_to_level(parameters[0]))

            if __debug__:
                np.testing.assert_almost_equal(pdf_cond.joint_probabilities.sum(), 1.0)

            parameters = parameters[1:]

            # add the conditional pdf
            pdf_conds[(val,)] = pdf_cond.copy()

        assert len(parameters) == 0, 'all parameters should be used to construct joint pdf'
        pdf_1.append_variables_using_conditional_distributions(pdf_conds)

        if __debug__ and _debug_params2matrix:
            # remove this (expensive) check after it seems to work a few times?
            try:
                np.testing.assert_array_almost_equal(matrix2params_incremental(pdf_1),
                                                     self.scalars_up_to_level(original_parameters))
            except AssertionError as e:
                ### I have the hunch that the above assertion is hit but that it is only if a parameter is 1 or 0,
                ### so that the parameter may be different but that it does not matter. still don't understand
                ### why it happens though...

                pdf_1_duplicate = pdf_1.copy()

                pdf_1_duplicate._debug_params2matrix = False  # prevent endless recursion

                params2matrix_incremental(pdf_1_duplicate,self.scalars_up_to_level(original_parameters))

                pdf_1_duplicate._debug_params2matrix = True

                if not pdf_1_duplicate == pdf_1:
                    print('error: the pdfs from the two different parameter lists are also not equivalent')

                    del pdf_1_duplicate

                    raise AssertionError(e)
                else:
                    # warnings.warn('I found that two PDF objects can have the same joint prob. matrix but a'
                    #               ' different list of identifying parameters. This seems to be due to a variable'
                    #               ' having 0.0 probability on a certain value, making the associated conditional'
                    #               ' PDF of other variables 0 and therefore those associated parameters irrelevant.'
                    #               ' Find a way to make these parameters still uniform? Seems to happen in'
                    #               ' "pdf_1.append_variables_using_conditional_distributions(pdf_conds)"...')

                    # note: (duplicated) the cause seems to be as follows. If you provide the parameters e.g.
                    # [0.0, [0.37028884415935004], [0.98942830522914993]] then the middle parameter is superfluous,
                    # because it defines a conditional probability p(b|a) for which its prior p(a)=0. So no matter
                    # what parameter is here, the joint prob p(a,b) will be zero anyway. In other words, many
                    # parameter lists map to the same p(a,b). This makes the inverse problem ambiguous:
                    # going from a p(a,b) to a parameter list. So after building a pdf from the above example of
                    # parameter values I may very well get a different parameter list from that pdf, even though
                    # the pdf built is the one intended. I don't see a way around this because even if this class
                    # makes it uniform, e.g. always making parameter values 0.0 in case their prior is zero,
                    # but then still a user or optimization procedure can provide any list of parameters, so
                    # then also the uniformized parameter list will differ from the user-supplied.

                    del pdf_1_duplicate

        assert pdf_1.numvariables == self.numvariables
        assert pdf_1.numvalues == self.numvalues

        self.duplicate(pdf_1)  # make this object (self) be the same as pdf_1

    elif self.numvariables == 1:
        if func == 'normal':
            params2matrix(self,parameters)
        else:
            self.joint_probabilities.joint_probabilities = numba_p2m(self.numvalues,
                                        self.numvariables,parameters,
                                        (self.numvalues,)*self.numvariables)

    else:
        assert len(parameters) == 0, 'at the least 0 parameters should be given for 0 variables...'

        raise ValueError('no parameters for 0 variables')

def imbalanced_tree_from_scalars(self, list_of_scalars, numvalues):
    """
    Helper function.
    Consider e.g. tree =
                    [0.36227870614214747,
                     0.48474422004766832,
                     [0.34019329926554265,
                      0.40787146599658614,
                      [0.11638879037422999, 0.64823088842780996],
                      [0.33155311703042312, 0.11398958845340294],
                      [0.13824154613818085, 0.42816388506114755]],
                     [0.15806602176772611,
                      0.32551465875945773,
                      [0.25748947995256499, 0.35415524846620511],
                      [0.64896559115417218, 0.65575802084978507],
                      [0.36051945555508391, 0.40134903827671109]],
                     [0.40568439663760192,
                      0.67602830725264651,
                      [0.35103999983495449, 0.59577145940649334],
                      [0.38917741342947187, 0.44327101890582132],
                      [0.075034425516081762, 0.59660319391007388]]]

    If you first call scalars_up_to_level on this you get a list [0.36227870614214747, 0.48474422004766832,
    0.34019329926554265, 0.40787146599658614, 0.15806602176772611, ...]. If you pass this flattened list through
    this function then you should get the above imbalanced tree structure back again.

    At each level in the resulting tree there will be <numvalues-1> scalars and <numvalues> subtrees (lists).
    :type list_of_scalars: list
    :type numvalues: int
    :rtype: list
    """

    num_levels = int(np.round(np.log2(len(list_of_scalars) + 1) / np.log2(numvalues)))

    all_scalars_at_level = dict()

    list_of_scalars_remaining = list(list_of_scalars)

    for level in range(num_levels):
        num_scalars_at_level = np.power(numvalues, level) * (numvalues - 1)

        scalars_at_level = list_of_scalars_remaining[:num_scalars_at_level]

        all_scalars_at_level[level] = scalars_at_level

        list_of_scalars_remaining = list_of_scalars_remaining[num_scalars_at_level:]

    def tree_from_levels(all_scalars_at_level):
        if len(all_scalars_at_level) == 0:
            return []
        else:
            assert len(all_scalars_at_level[0]) == numvalues - 1

            if len(all_scalars_at_level) > 1:
                assert len(all_scalars_at_level[1]) == numvalues * (numvalues - 1)
            if len(all_scalars_at_level) > 2:
                assert len(all_scalars_at_level[2]) == (numvalues*numvalues) * (numvalues - 1), \
                    'len(all_scalars_at_level[2]) = ' + str(len(all_scalars_at_level[2])) + ', ' \
                                                                                            '(numvalues*numvalues) * (numvalues - 1) = ' + str((numvalues*numvalues) * (numvalues - 1))
            if len(all_scalars_at_level) > 3:
                assert len(all_scalars_at_level[3]) == (numvalues*numvalues*numvalues) * (numvalues - 1)
            # etc.

            tree = list(all_scalars_at_level[0][:(numvalues - 1)])

            if len(all_scalars_at_level) > 1:
                # add <numvalues> subtrees to this level
                for subtree_id in range(numvalues):
                    all_scalars_for_subtree = dict()

                    for level in range(len(all_scalars_at_level) - 1):
                        num_scalars_at_level = len(all_scalars_at_level[level + 1])

                        assert np.mod(num_scalars_at_level, numvalues) == 0, 'should be divisible nu <numvalues>'

                        num_scalars_for_subtree = int(num_scalars_at_level / numvalues)

                        all_scalars_for_subtree[level] = \
                            all_scalars_at_level[level + 1][subtree_id * num_scalars_for_subtree
                                                            :(subtree_id + 1) * num_scalars_for_subtree]

                    subtree_i = tree_from_levels(all_scalars_for_subtree)

                    if len(all_scalars_for_subtree) > 1:
                        # numvalues - 1 scalars and numvalues subtrees
                        assert len(subtree_i) == (numvalues - 1) + numvalues, 'len(subtree_i) = ' \
                                                                              + str(len(subtree_i)) \
                                                                              + ', expected = ' \
                                                                              + str((numvalues - 1) + numvalues)
                    elif len(all_scalars_for_subtree) == 1:
                        assert len(subtree_i) == numvalues - 1

                    tree.append(subtree_i)

            return tree

    tree = tree_from_levels(all_scalars_at_level)

    assert maximum_depth(tree) == len(all_scalars_at_level)  # should be numvariables if the scalars are parameters
    assert len(flatten(tree)) == len(list_of_scalars), 'all scalars should end up in the tree, and not duplicate'

    return tree