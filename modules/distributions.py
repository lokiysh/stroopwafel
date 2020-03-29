from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from classes import Location
import numpy as np
from utils import *
import sampler
import prior
import json

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
        self.rejection_rate = 0
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
            if self.rejection_rate == 1:
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
        return np.minimum(max_value - value, value - min_value) / (max_value - min_value)

    @classmethod
    def calculate_rejection_rate(self, gaussians, num_batches, output_folder, debug, run_on_helios):
        dimension_ranges = []
        for dimension in sorted(gaussians[0].mean.dimensions.keys(), key = lambda d: d.name):
            dimension_ranges.append([dimension.min_value, dimension.max_value])
        batch_num = 0
        batches = []
        for index, gaussian in enumerate(gaussians):
            current_batch = dict()
            current_batch['batch_num'] = "gauss_" + str(batch_num)
            param = dict()
            param["mean"] = gaussian.mean.to_array()
            param["cov"] = [(i * gaussian.kappa)**2 for i in gaussian.sigma.to_array()]
            param["dimension_ranges"] = dimension_ranges
            command = ["python " + os.getcwd() + "/modules/find_rejection_rate.py '" + json.dumps(param) + "'"]
            current_batch['process'] = run_code(command, current_batch['batch_num'], output_folder, debug, run_on_helios)
            current_batch['gaussian'] = gaussian
            batches.append(current_batch)
            batch_num = batch_num + 1
            if len(batches) == num_batches or index == len(gaussians) - 1:
                for batch in batches:
                    batch['process'].wait()
                    batch['gaussian'].rejection_rate = float(get_slurm_output(output_folder, batch['batch_num']))
                batches = []
