class CausalImpactResponse(object):
    perturbed_variables = None

    mi_orig = None
    mi_nudged_list = None
    mi_diffs = None
    avg_mi_diff = None
    std_mi_diff = None

    impacts_on_output = None
    avg_impact = None
    std_impact = None

    correlations = None
    avg_corr = None
    std_corr = None

    residuals = None
    avg_residual = None
    std_residual = None

    nudges = None
    upper_bounds_impact = None

    def __init__(self):
        self.perturbed_variables = None

        self.mi_orig = None
        self.mi_nudged_list = None
        self.mi_diffs = None
        self.avg_mi_diff = None
        self.std_mi_diff = None

        self.impacts_on_output = None
        self.avg_impact = None
        self.std_impact = None

        self.correlations = None
        self.avg_corr = None
        self.std_corr = None

        self.residuals = None
        self.avg_residual = None
        self.std_residual = None

        self.nudges = None
        self.upper_bounds_impact = None
