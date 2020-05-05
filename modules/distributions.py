from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from classes import Location
import numpy as np
from utils import *
import json

class NDimensionalDistribution:
    """
    This is a parent class for any NDimensional distribution. Any subclass of this class, must implement all the methods
    defined in this class, for example the run_sampler method.
    Examples of NDimensionalDistribution are Gaussian class and InitialDistribution class. (See below)
    """
    def __init__(self, dimensions):
        self.dimensions = dimensions

    @abstractmethod
    def run_sampler(self, num_samples):
        #This method must be implemented by all the sub classes
        pass

    @abstractmethod
    def calculate_probability_of_locations_from_distribution(self, locations):
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
        mask = np.ones(num_samples, dtype = bool)
        headers = sorted(self.dimensions, key = lambda d: d.name)
        samples = []
        for dimension in headers:
            current_samples = dimension.run_sampler(num_samples)
            mask &= dimension.is_sample_within_bounds(current_samples)
            samples.append(current_samples)
        samples = list(map(list, zip(*samples)))
        locations = [Location(dict(zip(headers, row)), {}) for row in samples]
        return (locations, mask)

    def calculate_probability_of_locations_from_distribution(self, locations):
        pass

class Gaussian(NDimensionalDistribution):
    """
    This class inherits from NDimensionalDistribution. It will be used during the refinement phase to draw adapted distributions.
    mean (Location) : The mean location of the gaussian. It is of Location type and therefore stores all the dimension
    sigma (Location) : The sigma of the gaussian. Also of Location class type
    kappa (float) : An independent scaling factor describing how wide the gaussian can be.
    """
    def __init__(self, mean, sigma, kappa = 1):
        self.mean = mean
        self.sigma = sigma
        self.kappa = kappa
        self.__bound_factor(True)
        self.rejection_rate = 0
        self.biased_weight = 1
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
        mask = np.ones(num_samples, dtype = bool)
        current_samples = multivariate_normal.rvs(mean = self.mean.to_array(), cov = self.cov, size = num_samples)
        headers = sorted(self.mean.dimensions.keys(), key = lambda d: d.name)
        for index, dimension in enumerate(headers):
            mask &= dimension.is_sample_within_bounds(current_samples[:, index])
        locations = [Location(dict(zip(headers, row)), {}) for row in current_samples]
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
                sigma[variable] = average_density_one_dim / variable.prior(variable, val)
            gaussians.append(Gaussian(hit, Location(sigma, {})))
        return gaussians

    """
    This function is used to vary the width of the gaussian depending on how close it is to the edge of the dimension
    """
    def __bound_factor(self, consider = True):
        original_cov = np.power(np.asarray(self.sigma.to_array()), 2)
        cov = []
        for dimension in sorted(self.mean.dimensions.keys(), key = lambda d: d.name):
            mean = self.mean.dimensions[dimension]
            sigma = self.sigma.dimensions[dimension]
            value = min([dimension.max_value - mean, mean - dimension.min_value]) / 2
            if sigma < value:
                value = sigma
            self.sigma.dimensions[dimension] = value
            cov.append(value**2)
        cov = np.asarray(cov)
        if not consider:
            self.cov = original_cov
        else:
            self.cov = cov

    """
    Calculates the rejection rate of each of the gaussians in batches
    IN:
        gaussians (List(Gaussian)) : The gaussians to calculate the rejection rate
        num_batches (int) : The number of batches to run in parallel
        output_folder (Path) : Path to the output folder
        debug (Boolean) : If debug mode is on or off
        run_on_helios (Boolean) : If running on helios clusters or not
    """
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
            param["cov"] = gaussian.cov.tolist()
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

    """
    Given a list of locations, calculates the probability of drawing each location from the given gaussian
    IN:
        locations (list(Location)) : list of locations
    """
    @abstractmethod
    def calculate_probability_of_locations_from_distribution(self, locations):
        mean = self.mean.to_array()
        variance = np.diagflat(self.cov)
        samples = []
        for location in locations:
            samples.append(location.to_array())
        pdf = multivariate_normal.pdf(samples, mean, variance, allow_singular = True)
        pdf = (pdf * self.biased_weight) * (1 - self.rejection_rate)
        for index, location in enumerate(locations):
            location.properties['q_pdf'] = location.properties.get('q_pdf', 0) + pdf[index]
