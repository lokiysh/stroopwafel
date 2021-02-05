import numpy as np
from .constants import *
"""
Defines different kinds of sampling distributions for the initial properties
"""
def uniform(num_samples, **kwargs):
    """
     method to sample from a uniform distribution in the range [x, y)
    IN:
        num_samples : number of samples to be returned
        x (float) : starting value
        y (float) : ending value
    OUT:
        a list of samples in the range [x, y) considering uniform sampling distribution
    """
    x = kwargs['x']
    y = kwargs['y']
    return np.random.uniform(x, y, num_samples)

def flat_in_log(num_samples, **kwargs):
    """
    method to sample from a flat_in_log distribution in the range[x, y]
    IN:
        num_samples : number of samples to be returned
        x (float) : starting value
        y (float) : ending value
    OUT:
        a list of samples in the range [x, y) considering power law distribution
    """
    x = kwargs['x']
    y = kwargs['y']
    return np.random.uniform(x, y, num_samples)   # assumes that the values x and y are already in log-scale

def flat(num_samples, **kwargs):
    """
    method to sample from a flat distribution
    IN:
        num_samples : number of samples to be returned
        x (float) : The flat value for sampling
    OUT:
        a list of num_samples zeros
    """
    x = kwargs['x']
    return np.ones(num_samples) * x   # in a flat distribution each sample has an equal probability

def kroupa(num_samples = 1, **kwargs):
    """
    method to sample from the kroupa mass function in the range [x, y)
    IN:
        num_samples (int) : number of samples to be returned
        x (float) : starting value
        y (float) : ending value
    OUT:
        a list of samples in the range [x, y) considering kroupa sampling distribution
    """
    x = kwargs['x']
    y = kwargs['y']
    return np.power(np.random.uniform(0, 1, num_samples) * (np.power(y, 1 + ALPHA_IMF) - np.power(x, 1 + ALPHA_IMF)) + np.power(x, 1 + ALPHA_IMF), 1 / (1 + ALPHA_IMF))

def uniform_in_sine(num_samples, **kwargs):
    """
    method to sample from a uniform sine distribution. useful for example in solid angles sampling
    IN:
        num_samples (int) : number of samples to be returned
        x (float) : starting value
        y (float) : ending value
    OUT:
        a list of samples in the range [x, y) considering uniformly in sine sampling distribution
    """
    x = kwargs['x']
    y = kwargs['y']
    return np.random.uniform(x, y, num_samples)  # assumes values to range from -1 and 1


def uniform_in_cosine(num_samples, **kwargs):
    """
    method to sample from a uniform cosine distribution. useful for example in solid angles sampling
    IN:
        num_samples (int) : number of samples to be returned
        x (float) : starting value
        y (float) : ending value
    OUT:
        a list of samples in the range [x, y) considering uniformly in sine sampling distribution
    """
    x = kwargs['x']
    y = kwargs['y']
    return np.random.uniform(x, y, num_samples)  # assumes value to range from -1 to 1
