#!/usr/bin/env python
# coding: utf-8
"""
STROOPWAFEL: Simulating The Rare Outcomes Of Populations With AIS For Efficient Learning.
Based on Broekgaarden et al. 2019
"""

import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
import subprocess
import os

ALPHA_IMF = -2.3

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
        if self.sampler == Sampler.uniform or self.sampler == Sampler.kroupa:
            return (samples >= self.min_value) & (samples <= self.max_value)
        elif self.sampler == Sampler.flat_in_log:
            return (samples >= np.power(10, float(self.min_value))) & (samples <= np.power(10, float(self.max_value)))

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

class NDimensionalDistribution:
    """
    This is a parent class for any NDimensional distribution. Any subclass of this class, must implement the run_sampler method.
    Examples of NDimensionalDistribution are Gaussian class and InitialDistribution class. (See below)
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    @abstractmethod
    def run_sampler(self, num_samples):
        #This method must be implemented by all the sub classes
        pass

class InitialDistribution(NDimensionalDistribution):
    """
    This class inherits from NDimensionalDistribution. It will be used during the exploration phase to draw from Sampler class of Dimensions.
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    def run_sampler(self, num_samples):
        """
        This function tells the class how to draw the samples.
        IN:
            num_samples (int) : How many samples to draw
        OUT:
            (list(Location), list(bool)) : A pair of lists (both of size num_samples), the first describing the samples drawn, and the second telling which of the samples are within the bounds of the respective variables
        """
        locations = [Location({}, {}) for i in range(num_samples)]
        mask = np.ones(num_samples, dtype = bool)
        for dimension in self.dimensions:
            if dimension.sampler != None:
                current_samples = dimension.run_sampler(num_samples)
                mask &= dimension.is_sample_within_bounds(current_samples)
                for index, sample in enumerate(current_samples):
                    locations[index].dimensions[dimension] = sample
        return (locations, mask)

class Gaussian(NDimensionalDistribution):
    """
    This class inherits from NDimensionalDistribution. It will be used during the refinement phase to draw adapted distributions.
    mean (Location) : The mean location of the gaussian. It is of Location type and therefore stores all the dimension
    sigma (Location) : The sigma of the gaussian. Also of Location class type
    kappa (float) : An independent scaling factor describing how wide the gaussian can be.
    rejection_rate (float) : A number which tells how often samples drawn are rejected because they are outside of the boundary.
    """
    def __init__(self, mean, sigma, kappa = 1, biased_weight = 1):
        self.mean = mean
        self.sigma = sigma
        self.kappa = kappa
        self.biased_weight = biased_weight
        (locations, mask) = self.run_sampler(10000)
        self.rejection_rate = 1 - np.sum(mask) / len(mask)

    """
        This function tells the class how to draw the samples for this class
        IN:
            num_samples (int) : How many samples to draw
            consider_rejection_rate (bool) : If we should consider rejection rate while drawing samples. If True, it will draw more samples (based on rejection_rate), such that the num_samples are within the bounds
        OUT:
            (list(Location), list(bool)) : A pair of lists (both of size num_samples), the first describing the samples drawn, and the second telling which of the samples are within the bounds of the respective variables
        """
    def run_sampler(self, num_samples, consider_rejection_rate = False):
        if consider_rejection_rate == True:
            if (self.rejection_rate == 1.0):
                return ([], [])
            num_samples = int(np.ceil(num_samples / (1 - self.rejection_rate)))
        num_samples = int(num_samples * self.biased_weight)
        locations = []
        mask = np.ones(num_samples, dtype = bool)
        current_samples = multivariate_normal.rvs(mean = self.mean.to_array(), cov = np.power(self.kappa * np.asarray(self.sigma.to_array()), 2), size = num_samples)
        for i, sample in enumerate(current_samples):
            current_location = dict()
            for index, dimension in enumerate(sorted(self.mean.dimensions.keys(), key = lambda d: d.name)):
                current_location[dimension] = sample[index]
                mask[i] &= dimension.is_sample_within_bounds(sample[index])
            locations.append(Location(current_location, {}))
        return (locations, mask)

    @classmethod
    def draw_distributions(self, hit_locations, average_density_one_dim):
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
            for variable, val in hit.dimensions.items():
                sigma[variable] = (average_density_one_dim * self.__bound_factor(variable, val)) / variable.prior(variable, val)
            gaussians.append(Gaussian(hit, Location(sigma, {})))
        return gaussians

    @staticmethod
    def __bound_factor(dimension, value):
        max_value = dimension.max_value
        min_value = dimension.min_value
        if dimension.prior == Prior.flat_in_log:
            max_value = np.power(10, float(dimension.max_value))
            min_value = np.power(10, float(dimension.min_value))
        return np.minimum(max_value - value, value - min_value) / (max_value - min_value)

class Sampler:
    """
    Defines different kinds of samplers
    """
    #static method allows accessing this method without creating objects of the class
    @staticmethod
    def uniform(num_samples, **kwargs):
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
    def flat_in_log(num_samples, **kwargs):
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
        return np.power(10, np.random.uniform(x, y, num_samples))

    @staticmethod
    def flat(num_samples, **kwargs):
        """
        static method to run a sampling for a flat distribution
        IN:
            num_samples : number of samples to be returned
            x (float) : The flat value for sampling
        OUT:
            a list of num_samples zeros
        """
        x = kwargs['x']
        return np.ones(num_samples) * x

    @staticmethod
    def kroupa(num_samples = 1, **kwargs):
        """
        static method to run a kroupa sampling for the range [x, y)
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

    @staticmethod
    def uniform_in_sine(num_samples, **kwargs):
        """
        static method to run a uniform sine sampling  useful for example in solid angles sampling
        IN:
            num_samples (int) : number of samples to be returned
        OUT:
            a list of samples in the range [x, y) considering uniformly in sine sampling distribution
        """
        x = np.sin(kwargs['x'])
        y = np.sin(kwargs['y'])
        return np.arcsin(np.random.uniform(x, y, num_samples))

"""
## This class will be deprecated soon, please do not use it
"""
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
            float : The calculated value of sigma
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

class Prior:
    """
    This class defines the birth priors of the dimensions
    """
    @staticmethod
    def uniform(dimension, value):
        """
        Usual method to calculate priors probability.
        IN:
            dimension (Dimension) : The dimension to calculate the prior for
        OUT:
            (float) : The prior probability
        """
        return 1.0 / (dimension.max_value - dimension.min_value)

    @staticmethod
    def flat_in_log(dimension, value):
        max_value = np.power(10, float(dimension.max_value))
        min_value = np.power(10, float(dimension.min_value))
        return 1 / (value * (np.log(max_value) - np.log(min_value)))

    @staticmethod
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

    @staticmethod
    def uniform_in_sine(dimension, value):
        norm_const = 2
        return np.abs(np.cos(value)) / norm_const

class Stroopwafel:

    def __init__(self, num_dimensions, num_systems = 100, num_batches = 1, num_samples_per_batch = 100, debug = False, run_on_helios = True, mc_only = False):
        self.ALPHA_IMF = -2.3 #Initial Mass Functions alpha value
        self.fraction_explored = 1
        self.num_dimensions = num_dimensions
        self.num_systems = num_systems
        self.num_batches = num_batches
        self.num_samples_per_batch = num_samples_per_batch
        self.debug = debug
        self.run_on_helios = run_on_helios
        self.mc_only = mc_only

    def generate_grid(self, locations, filename = 'grid.txt'):
        """
        Function which generated a txt file with the locations specified
        IN:
            locations (list[Location]) : list of locations to be printed in the file
            filename (string) : filename to save
        OUT:
            generates file with name filename with the given locations and saves to the disk
        """
        header = []
        grid = []
        for location in locations:
            current_location = []
            for key, value in location.dimensions.items():
                if key.should_print:
                    if len(grid) == 0:
                        header.append(key.name)
                    current_location.append(value)
            for key, value in location.properties.items():
                if len(grid) == 0:
                    header.append(key)
                current_location.append(value)
            grid.append(current_location)
        DELIMITER = ', '
        np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments = '')

    def update_fraction_explored(self):
        """
        Function which updates the fraction of region which is already explored
        """
        num_hits = len(self.hits)
        unidentified_region_weight = 1.0 / (self.fraction_explored * self.num_systems)
        target_rate = num_hits / self.num_explored
        numerator = target_rate * (np.sqrt(1. - target_rate) - np.sqrt(unidentified_region_weight))
        denominator = np.sqrt(1. - target_rate) * (np.sqrt(unidentified_region_weight * (1. - target_rate)) + target_rate)
        self.fraction_explored = 1 - numerator / denominator

    def should_continue_exploring(self):
        """
        Function that estimates if we should continue exploring or are we ready
        OUT:
            bool : boolean value telling If we should continue exploring or not
        """
        if self.mc_only:
            return self.num_explored < self.num_systems
        return self.num_explored / self.num_systems < self.fraction_explored

    def mark_location_as_hits(self, points, hit_locations):
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

    def calculate_weights_of_hits(self, hit_locations, adapted_distributions):
        """
        Function that calculates the weights of the hits based on their priors
        IN:
            hit_locations (list(Location)) : the locations which have hits
            adapted_distributions (list(NDimensionalDistribution)) : list of adpated distributions, for example, gaussians
        """
        num_distributions = len(adapted_distributions)
        num_hits = len(hit_locations)
        samples = np.zeros((num_hits, self.num_dimensions))
        pi_x = np.ones(num_hits)
        q_x = np.zeros((num_distributions, num_hits))
        for counter, location in enumerate(hit_locations):
            index = 0
            for dimension in sorted(location.dimensions.keys(), key = lambda d: d.name):
                samples[counter][index] = location.dimensions[dimension]
                pi_x[counter] *= dimension.prior(dimension, location.dimensions[dimension])
                index = index + 1
        for counter, distribution in enumerate(adapted_distributions):
            mean = np.zeros(self.num_dimensions)
            variance = np.zeros((self.num_dimensions, self.num_dimensions))
            index = 0
            for dimension in sorted(distribution.mean.dimensions.keys(), key = lambda d: d.name):
                mean[index] = distribution.mean.dimensions[dimension]
                variance[index][index] = distribution.sigma.dimensions[dimension]
                index = index + 1
            q_x[counter, :] = (multivariate_normal.pdf(samples, mean, variance, allow_singular = True) * distribution.biased_weight) / (1 - distribution.rejection_rate)
        q_pdf = np.sum(q_x, axis = 0) / num_distributions
        for counter, location in enumerate(hit_locations):
            Q = (self.fraction_explored * pi_x[counter]) + ((1 - self.fraction_explored) * q_pdf[counter])
            location.weight = pi_x[counter] / Q

    def print_hits(self, hit_locations, filename):
        """
        Function that prints all the hits to a file
        IN:
            hit_locations(list(Location)): All the hits that need to be printed
            filename (String) : The filename that will be saved
        """
        header = []
        grid = []
        for hit in hit_locations:
            current_hit = []
            for key, value in hit.properties.items():
                if len(grid) == 0:
                    header.append(key)
                current_hit.append(value)
            for key, value in hit.dimensions.items():
                if len(grid) == 0:
                    header.append(key.name)
                current_hit.append(value)
            current_hit.append(hit.weight)
            grid.append(current_hit)
        header.append('weight')
        DELIMITER = ', '
        np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments = '')

    def determine_rate(self, hit_locations):
        """
        Function that determines the rate of producing the hits from the algorithm
        IN:
            hit_locations(list(Location)): All the hits that need to be printed
        OUT:
            (float, float): A pair of values which has the rate of stroopwafel rate and the uncertainity in the rate
        """
        if len(hit_locations) == 0:
            return (0, 0)
        phi = np.ones(len(hit_locations))
        for index, location in enumerate(hit_locations):
            phi[index] = location.weight
        stroopwafel_rate = np.round(np.sum(phi) / self.num_systems, 4)
        uncertainity = np.round(np.std(phi, ddof = 1) / np.sqrt(self.num_systems), 6)
        return (stroopwafel_rate, uncertainity)

    #copied from stack overflow
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', autosize = False):
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
        if iteration >= total:
            print()

    def generate_slurm_file(self, command):
        filename = os.getcwd() + "/slurm.sh"
        writer = open(filename, 'w')
        writer.write("#!/bin/bash\n")
        writer.write("#SBATCH --output=log.txt\n")
        writer.write(command + "\n")
        writer.close()

    def run_code(self, command):
        """
        Function that runs the command specified on the command shell.
        IN:
            command list(String): A list of commands to be triggered along with the options
        OUT:
            subprocess : An instance of subprocess created after running the command
        """
        if command != None:
            if not self.debug:
                stdout = subprocess.PIPE
                stderr = subprocess.PIPE
            else:
                stdout = stderr = None
            command_to_run = " ".join(str(v) for v in command))
            if self.run_on_helios:
                self.generate_slurm_file(" ".join(str(v) for v in command))
                command_to_run = "sbatch -W -Q slurm.sh"
            process = subprocess.Popen(command_to_run, shell = True, stdout = stdout, stderr = stderr)
            return process

    def wait_for_completion(self, batches):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
        """
        for batch in batches:
            if batch['process']:
                batch['process'].wait()
            if self.interesting_systems_method != None:
                self.hits.extend(self.interesting_systems_method(batch))
            self.finished += self.num_samples_per_batch
            self.printProgressBar(self.finished, self.num_systems, prefix = 'progress', suffix = 'complete', length = 20)
            if self.finished >= self.num_systems:
                break

    def initialize(self, interesting_systems_method, configure_code_run, update_properties_method = None):
        """
        This function is the one which is run only once in the stroopwafel class. It initializes the associated variables and the function calls that user will specify
        IN:
            interesting_system_method: The method provided by the user which will determine what an interesting system is for stroopwafel
            update_properties_method: The method provided by the user which will run to update the properties of each of the location once it is sampled
            configure_code_run: The method provided by the user which will be running for each of the batches to determine the command line args for that batch
        """
        self.interesting_systems_method = interesting_systems_method
        self.update_properties_method = update_properties_method
        self.configure_code_run = configure_code_run
        self.batch_num = 0
        self.num_explored = 0
        self.finished = 0
        self.hits = []
        self.printProgressBar(0, self.num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def explore(self, dimensions, intial_pdf):
        """
        This function is the exploration phase of the stroopwafel
        IN:
            dimensions (list(Dimension)) : A list of user created dimensions, each an instance of Dimension class
            initial_pdf (NDimensionalDistribution) : An instance of NDimensionalDistribution showing how to sample from in the exploration phase
        """
        while self.should_continue_exploring():
            batches = []
            for batch in range(self.num_batches):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                (locations, mask) = intial_pdf.run_sampler(self.num_samples_per_batch)
                if self.update_properties_method != None:
                    self.update_properties_method(locations)
                command = self.configure_code_run(current_batch)
                self.generate_grid(locations, current_batch['grid_filename'])
                current_batch['process'] = self.run_code(command)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.wait_for_completion(batches)
            self.num_explored += self.num_batches * self.num_samples_per_batch
            self.update_fraction_explored()
        print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(len(self.hits), self.num_explored, len(self.hits) / self.num_explored, self.fraction_explored))

    def adapt(self, n_dimensional_distribution_type = Gaussian):
        """
        Adaptive phase of stroopwafel
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        self.num_to_be_refined = self.num_systems - self.num_explored
        if self.num_to_be_refined > 0:
            average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / self.num_dimensions)
            self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(self.hits, average_density_one_dim)

    def refine(self):
        """
        Refinement phase of stroopwafel
        """
        refined = False
        while self.num_to_be_refined > 0:
            batches = []
            for batch in range(self.num_batches):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                locations_ref = []
                for distribution in self.adapted_distributions:
                    (locations, mask) = distribution.run_sampler(self.num_samples_per_batch, True)
                    locations_ref.extend(np.asarray(locations)[mask])
                np.random.shuffle(locations_ref)
                locations_ref = locations_ref[0 : self.num_samples_per_batch]
                if self.update_properties_method != None:
                    self.update_properties_method(locations_ref)
                command = self.configure_code_run(current_batch)
                self.generate_grid(locations_ref, current_batch['grid_filename'])
                current_batch['process'] = self.run_code(command)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.wait_for_completion(batches)
            self.num_to_be_refined -= self.num_batches * self.num_samples_per_batch
            refined = True
        if refined:
            print ("\nRefinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(len(self.hits) - len(self.adapted_distributions), (self.num_systems - self.num_explored), (len(self.hits) - len(self.adapted_distributions)) / (self.num_systems - self.num_explored)))

    def postprocess(self, filename):
        """
        Postprocessing phase of stroopwafel
        IN:
            filename(String): It tells what is the filename to store all the hits and its properties
        """
        try:
            self.calculate_weights_of_hits(self.hits[len(self.adapted_distributions):], self.adapted_distributions)
        except AttributeError:
            pass
        self.print_hits(self.hits, filename)
        (stroopwafel_rate, uncertainity) = self.determine_rate(self.hits)
        print ("Rate of hits = %f with uncertainity = %f" %(stroopwafel_rate, uncertainity))
