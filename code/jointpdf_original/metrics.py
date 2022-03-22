__author__ = 'rquax'

'''
In this file I will create functions that implement 'metrics' which operate on a pdf object and return
a number of (nested) list of numbers, depending on the metric.
'''

from jointpdf import JointProbabilityMatrix

from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import numbers
from scipy.optimize import brentq

# standard labels for standard concepts
class standard_labels:
    # for a time variable (guess should usually only be one of these)
    time = 'time'

    # can be a spin's state, or a voter's opinion in a voter model, or a bird's position and direction in a
    # bird flocking model, or the concentration of a chemical, etc.
    variable = 'variable'

    # (usually in combination with a 'time' variable.) The time-dependent PDFs of the variables (with label .variable)
    # may depend on a further initial condition, such as a snapshot system state that is used to generate system
    # trajectories from, or otherwise the initial concentrations of chemicals for instance. An initial condition
    # is distinct from a 'parameter' in the sense that it is assumed to influence the time-dependent PDFs of variables
    # only at time=0, and subsequently it only exterts an indirect influence by dissipation of information or e.g.
    # in case if hysteresis.
    initial_condition = 'initial_condition'


'''
Metrics.
'''


def shannon_entropy_simple(pdf, dict_of_args={}):  # instructive example, intended to be as small and simple as possible
    pdf = pdf.marginalize_distribution_retaining_only_labels([standard_labels.time])

    return pdf.entropy()


def shannon_equilibrium_entropy_of_system_state(pdf, dict_of_args={}):
    """
    More advanced example.
    :type pdf: JointProbabilityMatrix
    :type dict_of_args: dict
    :rtype: float
    """

    unsupported_labels = set(pdf.get_labels()).difference({standard_labels.variable, standard_labels.time})

    if len(unsupported_labels) > 0:
        print 'debug: unsupported_labels =', unsupported_labels

    for uns_lbl in unsupported_labels:
        # these are variable indices with labels that I don't understand
        variables_with_label = pdf.get_variables_with_label_in([uns_lbl])

        # I assume that these variables with unsupported labels are parameters, like 'temperature', for which
        # the user provides me with a specific value for which I need to compute my metric
        # todo: recurse over all values for all unsupported labels and return a dict of metric values?
        if not dict_of_args.has_key(uns_lbl):
            raise ValueError('PDF contains variable(s) with label=' + str(uns_lbl) + ' which makes no sense for'
                             + ' computing Shannon entropy. If it is a parameter like \'temperature\' then you '
                               'should provide a specific value for me to compute entropy for')

        # note: I suppose it can also be possible for the user to first have the option to select specific
        # values for parameters (such as a temperature value) before he/she decides to pass the joint pdf
        # to a metric... but I can also imagine that a well-implemented metric tries its best to handle
        # unexpected additional dimensions, which I try here. A default behavior of iterating over all possible
        # values of an unexpected (and unspecified in dict_of_args) dimension could make a lot of sense, then it
        # should not be possible for any metric to crash in case extra dimensions are passed, but perhaps it could
        # be that the memory runs out if a dimension has a gazillion possible values... Perhaps this standard behavior
        # can be standardized. Like a function call
        # "I_expect_only_dimensions(['variable', 'time'], metric_i, dict_of_args)" which does what I do above:
        # try to find specific values for unexpected dimensions, and if not specified then iterate over its
        # possible values. Or so.

        # keep only a subpart of the pdf, namely where the unsupported label has the provided value
        pdf = pdf.conditional_probability_distribution(variables_with_label, dict_of_args[uns_lbl])

        assert not pdf is None

    assert len(set(pdf.get_labels())) in (1, 2), 'I expect either "variable" or "variable" and "time" as labels, ' \
                                                 'no more or less'

    # sum out over time, only keep pdf of variables (states)
    pdf = pdf.marginalize_distribution_retaining_only_labels([standard_labels.variable])

    return pdf.entropy()


def system_idt(pdf, dict_of_args={}):
    """
    This is the IDT at the level of the entire system state, so the characteristic decay of I(S[0], S[t]) as function
     of t, where S[i] are considered stochastic variables of the joint state of all variables in the <pdf> with label
     <standard_labels.variable>.

     Note: to compute the IDT per individual variable (like each spin in an Ising network) use individual_idt() instead.
    :type pdf: JointProbabilityMatrix
    :param dict_of_args: dict of arbitrary arguments you want to pass, depending on the metric.
    Like a specific value for unsupported dimensions such as 'temperature', if any.
    """

    # todo: make a separate class ConditionalProbabilityistribution, wrapping the dict?

    pdf.retain_only_labels([standard_labels.time, standard_labels.variable, standard_labels.initial_condition])

    # these three types of variables are needed for computing idt
    assert standard_labels.time in pdf.get_labels()
    assert standard_labels.variable in pdf.get_labels()
    assert standard_labels.initial_condition in pdf.get_labels()

    # todo: implement a mutual_information which takes labels as arguments

    time_variables = pdf.marginalize_distribution_retaining_only_labels([standard_labels.time])
    assert len(time_variables) == 1, 'should have exactly one time parameter'

    pdf_per_timestep = pdf.conditional_probability_distributions(time_variables)

    time_steps = sorted(pdf_per_timestep.keys())
    mutual_info_over_time = []

    assert len(time_steps) > 0, 'there are no time steps in the joint pdf?'
    assert len(time_steps) > 1, 'there are too few time steps in the joint pdf'

    assert isinstance(time_steps[0], numbers.Number), 'time step values should be numeric (not wrong per se, but weird)'

    for time_step in time_steps:
        mutual_info = pdf_per_timestep[time_step].mutual_information_labels(standard_labels.variable,
                                                                            standard_labels.initial_condition)

        mutual_info_over_time.append(mutual_info)

    e_const = 2.718281828459045

    max_mi = max(mutual_info_over_time)
    t_max = mutual_info_over_time.index(max_mi)

    # clip the mutual information curve to after the peak, because it can be a unimodal curve and then there will be
    # more than one root below for brentq, and it does not like that
    mutual_info_over_time = mutual_info_over_time[t_max:]

    # value of the mutual information curve at the IDT point. We here use the definition of a decay to 1/e of the
    # maximum value
    mi_at_idt = (max_mi - min(mutual_info_over_time)) * (1.0 / e_const) + min(mutual_info_over_time)

    # curve of mutual information over time (after peak), but shifted along y-axis so that MI=0 is the IDT point
    func_mi_curve_around_zero = InterpolatedUnivariateSpline(time_steps, np.subtract(mutual_info_over_time, mi_at_idt),
                                                             k=1)

    t_left = min(time_steps)
    t_right = max(time_steps)

    max_iter_root_finding = 3000
    idt = brentq(func_mi_curve_around_zero, t_left, t_right, rtol=1e-6, maxiter=max_iter_root_finding)

    idt = float(idt)  # make sure it is a float

    assert t_left <= idt <= t_right, 'found invalid IDT=' + str(idt) + ', not in ' + str(t_left) + ' ... ' \
                                     + str(t_right)

    return idt

'''
Unit testing.
'''


def run_equilibrium_entropy_test():
    data = [[1,2],[0,3],[0,1],[0,1]]

    pdf = JointProbabilityMatrix(2, 2)  # todo: enable this in one call
    pdf.estimate_from_data(data)

    assert pdf.entropy() == 1.5

    pdf.append_variables(1)
    pdf.set_label(2, 'time')  # this is a parameter that will be summed out to get the equilibrium entropy

    assert shannon_equilibrium_entropy_of_system_state(pdf) == 1.5
    assert shannon_entropy_simple(pdf) == 1.5


def run_all_tests():
    run_equilibrium_entropy_test()
    print 'note: run_equilibrium_entropy_test successful.'