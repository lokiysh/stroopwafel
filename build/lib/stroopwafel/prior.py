import numpy as np
from .constants import *
"""
This class defines the birth priors of the dimensions
"""

def uniform(dimension, value):
    """
    Usual method to calculate priors probability.
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
    OUT:
        (float) : The prior probability
    """
    return 1.0 / (dimension.max_value - dimension.min_value)

def flat_in_log(dimension, value):
    #return 1 / (value * (np.log(dimension.max_value) - np.log(dimension.min_value)))
    #This will assume that all values are already the log values
    return 1.0 / (dimension.max_value - dimension.min_value)

def kroupa(dimension, value):
    """
    method to calculate priors probability for kroupa distributions
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    norm_const = (ALPHA_IMF + 1) / (np.power(dimension.max_value, ALPHA_IMF + 1) - np.power(dimension.min_value, ALPHA_IMF + 1))
    return norm_const * np.power(value, ALPHA_IMF)

def uniform_in_sine(dimension, value):
    #norm_const = 2
    #return np.abs(np.cos(value)) / norm_const
    #This will assume that all values are already in sin
    return 1.0 / (dimension.max_value - dimension.min_value)

def uniform_in_cosine(dimension, value):
    #norm_const = 2
    #return np.abs(np.cos(value)) / norm_const
    #This will assume that all values are already in sin
    return 1.0 / (dimension.max_value - dimension.min_value)
