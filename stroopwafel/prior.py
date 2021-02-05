import numpy as np
from .constants import *
"""
This class defines the prior/birth distributions of the dimensions
"""

def uniform(dimension, value):
    """
    method to calculate prior probabilities from a uniform distribution
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    return 1.0 / (dimension.max_value - dimension.min_value)

def flat_in_log(dimension, value):
    """
    method to calculate prior probabilities from a flat-in-log distribution
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    return 1.0 / (dimension.max_value - dimension.min_value) # this will assume that all values are already the log values

def kroupa(dimension, value):
    """
    method to calculate prior probabilities from the kroupa mass function
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    norm_const = (ALPHA_IMF + 1) / (np.power(dimension.max_value, ALPHA_IMF + 1) - np.power(dimension.min_value, ALPHA_IMF + 1))
    return norm_const * np.power(value, ALPHA_IMF)

def uniform_in_sine(dimension, value):
    """
    method to calculate prior probabilities from a uniform sine distribution
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    return 1.0 / (dimension.max_value - dimension.min_value) # this will be a normalising constant equal to 1/2, assuming the values are in between -1 and 1

def uniform_in_cosine(dimension, value):
    """
    method to calculate prior probabilities from a uniform cosine distribution
    IN:
        dimension (Dimension) : The dimension to calculate the prior for
        value (float) : The value at which we are calculating prior probability
    OUT:
        (float) : The prior probability
    """
    return 1.0 / (dimension.max_value - dimension.min_value) # this will be a normalising constant equal to 1/2, assuming the values are in between -1 and 1
