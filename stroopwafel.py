#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod

class Location:
    """
    Describe a point in space, using a dict of Dimension : to its value
    value (dict(Dimension : float)) : dictionary that describes a point in space
    hit_score (float) : The hit value of the location in the range [0.0, 1.0]
    """
    def __init__(self, value = {}, hit_score = 0, compas_id = -1, weight = 1):
        self.value = value
        self.hit_score = 0
        self.compas_id = compas_id
        self.weight = weight
    
    def __key(self):
        return str(self.value.items())

    def __hash__(self):
        return hash(self.__key())
    
    def __eq__(self, other):
        if isinstance(other, Location):
            return self.__key() == other.__key()
        return NotImplemented
    
    def __str__(self):
        string = ''
        for key in self.value.keys():
            string += str(key) + " " + str(self.value[key]) + "\n"
        return string

class Dimension:
    """
    How a variable for stroopwafeling should look like
    name (string): The name of this variable
    min_value (float): The minimum value this variable can take
    max_value (float): The maximum value this variable can take
    sampler (Samplers): type of sampling for this variable
    sigma_calculation_method (SigmaCalculationMethod) : defines the function to use to calculate sigma for this variable
    should_refine (bool) : Should we draw samples for this variables
    should_print (bool) : Should we print this variable to the Grid.txt
    """
    def __init__(self, name, min_value = 0, max_value = 0, sampler = None, sigma_calculation_method = None, prior = None, should_refine = False, should_print = True):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.sampler = sampler
        self.sigma_calculation_method = sigma_calculation_method
        self.prior = prior
        self.should_refine = should_refine
        self.should_print = should_print
    """
    Function that samples the variable based on the given sampler
    IN:
        num_samples (int) : number of samples required
    OUT:
        list(float) : samples
    """
    def run_sampler(self, num_samples = 1):
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

class NDimensionalDistribution:
    
    def __init__(self, dimensions = []):
        self.dimensions = dimensions
    
    @abstractmethod
    def run_sampler(self, num_samples):
        #This method must be implemented by all the sub classes
        pass

class InitialDistribution(NDimensionalDistribution):
    
    def __init__(self, dimensions = []):
        self.dimensions = dimensions
        
    def run_sampler(self, num_samples):
        assert num_samples > 0, "number of samples must be positive"
        locations = [Location({}) for i in range(num_samples)]
        mask = np.ones(num_samples, dtype = bool)
        for dimension in self.dimensions:
            if dimension.sampler != None:
                current_samples = dimension.run_sampler(num_samples)
                mask &= dimension.is_sample_within_bounds(current_samples)
                for index, sample in enumerate(current_samples):
                    locations[index].value[dimension] = sample
        return (locations, mask)
    
class Gaussian(NDimensionalDistribution):
    
    def __init__(self, mean, sigma, kappa = 1.0):
        self.mean = mean
        self.sigma = sigma
        self.kappa = kappa
        (locations, mask) = self.run_sampler(100000)
        self.rejection_rate = 1 - np.sum(mask) / len(mask)
        
    def run_sampler(self, num_samples, consider_rejection_rate = False):
        assert num_samples > 0, "number of samples must be positive"
        if consider_rejection_rate == True:
            if (self.rejection_rate == 1.0):
                return ([], [])
            num_samples = int(np.ceil(num_samples / (1 - self.rejection_rate)))
        locations = [Location({}) for i in range(num_samples)]
        mask = np.ones(num_samples, dtype = bool)
        for dimension in self.mean.value.keys():
            if dimension.should_refine:
                current_samples = multivariate_normal.rvs(mean = self.mean.value[dimension], cov = self.sigma.value[dimension], size= num_samples)
            else:
                current_samples = np.ones(num_samples) * self.mean.value[dimension]
            mask &= dimension.is_sample_within_bounds(current_samples)
            for index, sample in enumerate(current_samples):
                locations[index].value[dimension] = sample
        return (locations, mask)

class Samplers:
    """
    Defines different kinds of samplers
    """
    #static method allows accessing this method without creating objects of the class
    @staticmethod
    def uniform(num_samples = 1, **kwargs):
        """
        static method to run a uniform sampling for the range [x, y)
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
    
    @staticmethod
    def power_law(num_samples = 1, **kwargs):
        """
        static method to run a power law for the range [x, y), assuming power is -1
        #TODO : make it generic for other powers if required
        IN:
            num_samples : number of samples to be returned
            x (float) : starting value
            y (float) : ending value
        OUT:
            a list of samples in the range [x, y) considering power law distribution
        """
        x = kwargs['x']
        y = kwargs['y']
        return np.exp(np.random.uniform(0, 1, num_samples) * np.log(y / x)) * x
    
    @staticmethod
    def zero(num_samples = 1, **kwargs):
        """
        static method to run a sampling with zero (useful for eccentricity for example)
        IN:
            num_samples : number of samples to be returned
        OUT:
            a list of num_samples zeros
        """
        return np.zeros(num_samples)
    
    @staticmethod
    def kroupa(num_samples = 1, **kwargs):
        """
        static method to run a kroupa sampling for the range [x, y)
        ALPHA_IMF is assumed to be -2.3
        IN:
            num_samples (int) : number of samples to be returned
            x (float) : starting value
            y (float) : ending value
        OUT:
            a list of samples in the range [x, y) considering kroupa sampling distribution
        """
        ALPHA_IMF = -2.3
        x = kwargs['x']
        y = kwargs['y']
        return np.power(np.random.uniform(0, 1, num_samples) * (np.power(y, 1 + ALPHA_IMF) - np.power(x, 1 + ALPHA_IMF)) + np.power(x, 1 + ALPHA_IMF), 1 / (1 + ALPHA_IMF))

class SigmaCalculationMethod:
    """
    Defines various ways to calculate sigmas for gaussians
    """
    @staticmethod
    def usual(dimension, average_density_one_dim, kappa = 1.0):
        """
        Usual method for calculating the sigma
        IN:
            dimension (Dimension) : The dimension to find the sigma for
            average_density_one_dim (float) : The expected number of average density of hits in this dimension
            kappa (float) : free parameter for the width of the gaussian
        OUT:
            float : The calculated value of variance
        """
        cov = (dimension.max_value - dimension.min_value) * average_density_one_dim
        return np.power(kappa, 2) * np.power(cov, 2)
    
    @staticmethod
    def kroupa(dimension, average_density_one_dim, mean, kappa = 1.0):
        # Below looks ugly, but probably the only way to do it
        """
        kroupa method for calculating the sigma
        IN:
            dimension (Dimension) : The dimension to find the sigma for
            average_density_one_dim (float) : The expected number of average density of hits in this dimension
            kappa (float) : free parameter for the width of the gaussian
            mean (float) : the mean for the gaussian
        OUT:
            float : The calculated value of sigma
        """
        ALPHA_IMF = -2.3
        K1 = (ALPHA_IMF + 1) / (np.power(dimension.max_value, ALPHA_IMF + 1) - np.power(dimension.min_value, ALPHA_IMF + 1))
        inverse = (np.power(K1 / mean, 1 / -ALPHA_IMF) - np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF)) / (np.power(K1 / dimension.max_value, 1 / -ALPHA_IMF) - np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF))
        #Find average distance on the right side
        right = inverse + average_density_one_dim
        right_average_distance = K1 / np.power(right * (np.power(K1 / dimension.max_value, 1 / -ALPHA_IMF) - np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF)) + np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF), -ALPHA_IMF)
        #Do it for the left side
        left = inverse - average_density_one_dim
        left_average_distance = K1 / np.power(left * (np.power(K1 / dimension.max_value, 1 / -ALPHA_IMF) - np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF)) + np.power(K1 / dimension.min_value, 1 / -ALPHA_IMF), -ALPHA_IMF)
        cov = np.maximum(np.abs(right_average_distance - mean), np.abs(left_average_distance - mean))
        return np.power(kappa, 2) * np.power(cov, 2)

class Priors:
    """
    Defines the birth distribution of the dimensions
    """
    @staticmethod
    def usual(dimension, locations):
        return 1.0 / (dimension.max_value - dimension.min_value)

    @staticmethod
    def kroupa(dimension, locations):
        ALPHA_IMF = -2.3
        norm_const = (ALPHA_IMF + 1) / (np.power(dimension.max_value, ALPHA_IMF + 1) - np.power(dimension.min_value, ALPHA_IMF + 1))
        return norm_const * np.power(locations, ALPHA_IMF)

def generate_grid(locations = [], filename = 'grid.txt'):
    """
    IN:
        locations (list[Location]) : list of locations
        filename (string) : filename to save
    OUT:
        generates file with name filename with the given location
    """
    header = []
    grid = []
    for location in locations:
        current_location = []
        for key, value in location.value.items():
            if key.should_print == True:
                if len(grid) == 0:
                    header.append(key.name)
                current_location.append(value)
        grid.append(current_location)
    DELIMITER = ', '
    np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments='')

def update_fraction_explored(fraction_exploration_phase, num_binaries, num_explored, num_hits):
    weight_unidentified_region = 1.0 / (fraction_exploration_phase * num_binaries)
    target_rate = num_hits / num_explored
    numerator = target_rate * (np.sqrt(1. - target_rate) - np.sqrt(weight_unidentified_region))
    denominator = np.sqrt(1. - target_rate) * (np.sqrt(weight_unidentified_region * (1. - target_rate)) + target_rate)
    fraction_exploration_phase = 1 - numerator / denominator
    return fraction_exploration_phase

def should_continue_exploring(num_explored, num_binaries, fraction_exploration_phase):
    return num_explored / num_binaries < fraction_exploration_phase

def mark_location_as_hits(points = [], hit_locations = []):
    """
    Function to mark locations as hits (gives a score in the range [0, 1])
    IN:
        points (list(Location)) : List of Location that was generated initially
        hit_locations (list(Location)) : list of hits (interesting points)
    """
    hit_locations = set(hit_locations)
    for point in points:
        # TODO : Can we think of some way to give some score to all the points? An algorithm to see how distant it is from an interesting point
        if point in hit_locations:
            point.hit_score = 1

def draw_gaussians(hit_locations, average_density_one_dim):
    """
    Function that draws gaussians at the hits
    IN:
        hit_locations (list(Location)) : the locations which have hits to draw the gaussians
        average_density_one_dim (float) : The average density of hits to be expected per dimension
    OUT:
        gaussians (list(Gaussian)) : list of gaussians drawn
    """
    gaussians = []
    for hit in hit_locations:
        sigma = dict()
        for variable, val in hit.value.items():
            if variable.sigma_calculation_method != None:
                sigma[variable] = variable.sigma_calculation_method(variable, average_density_one_dim, hit.value[variable])
        gaussians.append(Gaussian(hit, Location(sigma)))
    return gaussians

def calculate_weights_of_gaussians(hit_locations, gaussians, num_dimensions, fraction_exploration_phase):
    num_gaussians = len(gaussians)
    num_hits = len(hit_locations)
    samples = np.zeros((num_hits, num_dimensions))
    pi_x = np.ones(num_hits)
    q_x = np.zeros((num_gaussians, num_hits))
    for counter, location in enumerate(hit_locations):
        index = 0
        for dimension in sorted(location.value.keys(), key = lambda d: d.name):
            if dimension.should_refine == True:
                samples[counter][index] = location.value[dimension]
                pi_x[counter] *= dimension.prior(dimension, location.value[dimension])
                index = index + 1
    for counter, gaussian in enumerate(gaussians):
        mean = np.zeros(num_dimensions)
        variance = np.zeros((num_dimensions, num_dimensions))
        index = 0
        for dimension in sorted(gaussian.mean.value.keys(), key = lambda d: d.name):
            if dimension.should_refine == True:
                mean[index] = gaussian.mean.value[dimension]
                variance[index][index] = gaussian.sigma.value[dimension]
                index = index + 1
        q_x[counter, :] = (multivariate_normal.pdf(samples, mean, variance, allow_singular = True)) / (1 - gaussian.rejection_rate)
    q_pdf = np.sum(q_x, axis = 0) / num_gaussians
    for counter, location in enumerate(hit_locations):
        Q = (fraction_exploration_phase * pi_x[counter]) + ((1 - fraction_exploration_phase) * q_pdf[counter])
        location.weight = pi_x[counter] / Q

def print_hits(hit_locations, filename = 'hits.txt'):
    header = ['ID', 'weight']
    grid = []
    for hit in hit_locations:
        current_hit = [int(hit.compas_id), hit.weight]
        for key, value in hit.value.items():
            if key.should_print == True:
                if len(grid) == 0:
                    header.append(key.name)
                current_hit.append(value)
        grid.append(current_hit)
    DELIMITER = ', '
    np.savetxt(filename, grid, fmt = '%i' + ' %f' * (len(header) - 1), delimiter = DELIMITER, header = DELIMITER.join(header), comments='')

def determine_rate(hit_locations, num_binaries):
    phi = np.ones(len(hit_locations))
    for index, location in enumerate(hit_locations):
        phi[index] = location.weight
    stroopwafel_rate = np.round(np.sum(phi) / num_binaries, 4)
    uncertainity = np.round(np.std(phi, ddof = 1) / np.sqrt(num_binaries), 6)
    return (stroopwafel_rate, uncertainity)

#copied from stack overflow
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', autosize = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        autosize    - Optional  : automatically resize the length of the progress bar to the terminal window (Bool)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % (prefix, fill, percent, suffix)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s' % styling.replace(fill, bar), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()