import sampler as sp
import numpy as np

class Dimension:
    """
    A class that represents one of the dimensions that we will stroopwafelize
    name (string): The name of this dimension. This will be used to print in the file
    min_value (float): The minimum value this dimension can take
    max_value (float): The maximum value this dimension can take
    sampler (Sampler): type of sampling for this dimension during the exploration phase. It should be from the class Sampler
    prior (Prior): A function that will be used to calculate the prior for this dimension
    should_print (bool): If this dimension should be printed to the file
    """
    def __init__(self, name, min_value = 0, max_value = 0, sampler = None, prior = None, should_print = True):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.sampler = sampler
        self.prior = prior
        self.should_print = should_print
        if self.sampler.__name__ == sp.flat_in_log.__name__:
            self.min_value = np.log10(min_value)
            self.max_value = np.log10(max_value)
        elif self.sampler.__name__ == sp.uniform_in_sine.__name__:
            self.min_value = np.sin(min_value)
            self.max_value = np.sin(max_value)
    """
    Function that samples the variable based on the given sampler class
    IN:
        num_samples (int) : number of samples required
    OUT:
        list(float) : samples
    """
    def run_sampler(self, num_samples):
        return self.sampler(num_samples, x = self.min_value, y = self.max_value)

    """
    Function that returns which samples are within the bounds of this variables
    IN:
        sample (list(float)) : a list of points which we need to check if in bounds
    OUT:
        list(bool) : which of the given samples are within the bounds
    """
    def is_sample_within_bounds(self, samples):
        return (samples >= self.min_value) & (samples <= self.max_value)

    def __str__(self):
        return self.name

class Location:
    """
    Describe a point in N-Dimensional space
    dimensions (dict(Dimension : float)) : dictionary of mapping between class Dimension and its float value
    properties (dict) : A location could have more properties which we dont want to stroopwafelize but still store. For example, metallicity_2 (same as metallicity_1) or ID from Compas
    hit_score (float) : The hit value of the location in the range [0.0, 1.0], describes how interesting is this location [Currently unused]
    weight (float) : A field to store the weightage of this location
    """
    def __init__(self, dimensions, properties, hit_score = 0, weight = 1):
        self.dimensions = dimensions
        self.hit_score = hit_score
        self.properties = properties
        self.weight = weight

    def __key(self):
        return str(self.dimensions.items())

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Location):
            return self.__key() == other.__key()
        return NotImplemented

    def to_array(self):
        array = []
        for dimension in sorted(self.dimensions.keys(), key = lambda d: d.name):
            array.append(self.dimensions[dimension])
        return array

    def __str__(self):
        string = ''
        for dimension in self.dimensions.keys():
            string += dimension.name + " : " + str(self.dimensions[dimension]) + ","
        return string

    def revert_variables_to_original_scales(self):
        for dimension, value in self.dimensions.items():
            if dimension.sampler.__name__ == sp.flat_in_log.__name__:
                self.dimensions[dimension] = np.power(10, value)
            elif dimension.sampler.__name__ == sp.uniform_in_sine.__name__:
                self.dimensions[dimension] = np.arcsin(value)

    def transform_variables_to_new_scales(self):
        for dimension, value in self.dimensions.items():
            if dimension.sampler.__name__ == sp.flat_in_log.__name__:
                self.dimensions[dimension] = np.log10(value)
            elif dimension.sampler.__name__ == sp.uniform_in_sine.__name__:
                self.dimensions[dimension] = np.sin(value)

