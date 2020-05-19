from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from classes import Location
import numpy as np
from utils import *
import json
from constants import *
import sampler as sp

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

    def calculate_rejection_rate(self, num_batches, output_folder, debug, run_on_helios):
        batch_num = 0
        num_systems = 0
        num_rejected = 0
        batches = []
        while num_systems < TOTAL_REJECTION_SAMPLES * 10:
            current_batch = dict()
            current_batch['batch_num'] = "initial_" + str(batch_num)
            param = dict()
            param['exploration'] = 1
            command = ["python " + os.getcwd() + "/modules/find_rejection_rate.py '" + json.dumps(param) + "'"]
            current_batch['process'] = run_code(command, current_batch['batch_num'], output_folder, debug, run_on_helios)
            batches.append(current_batch)
            batch_num = batch_num + 1
            num_systems += REJECTION_SAMPLES_PER_BATCH
            if len(batches) == num_batches or num_systems >= TOTAL_REJECTION_SAMPLES * 10:
                for batch in batches:
                    batch['process'].wait()
                    num_rejected += float(get_slurm_output(output_folder, batch['batch_num']))
                batches = []
        return num_rejected / num_systems

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
    def run_sampler(self, num_samples, dimensions, consider_rejection_rate = False):
        if consider_rejection_rate == True:
            if self.rejection_rate == 1:
                return ([], [])
            num_samples = int(2 * np.ceil(num_samples / (1 - self.rejection_rate)))
        num_samples = int(num_samples * self.biased_weight)
        mask = np.ones(num_samples, dtype = bool)
        current_samples = multivariate_normal.rvs(mean = self.mean.to_array(), cov = self.cov, size = num_samples)
        headers = sorted(dimensions, key = lambda d: d.name)
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
                sigma[variable] = self.__calculate_sigma(average_density_one_dim, variable, val)
            gaussians.append(Gaussian(hit, Location(sigma, {})))
        return gaussians

    """
    This function is used to vary the width of the gaussian depending on how close it is to the edge of the dimension
    """
    def __bound_factor(self, consider = True):
        self.cov = np.power(np.asarray(self.sigma.to_array()), 2)
        if not consider:
            return
        cov = []
        for dimension in sorted(self.mean.dimensions.keys(), key = lambda d: d.name):
            mean = self.mean.dimensions[dimension]
            sigma = self.sigma.dimensions[dimension]
            value = min([dimension.max_value - mean, mean - dimension.min_value]) / 2
            if sigma < value:
                value = sigma
            self.sigma.dimensions[dimension] = value
            cov.append(value**2)
        self.cov = np.asarray(cov)

    @classmethod
    def __calculate_sigma(self, average_density_one_dim, dimension, value):
        if dimension.sampler.__name__ == sp.kroupa.__name__:
            norm_factor = (ALPHA_IMF + 1) / (pow(dimension.max_value, ALPHA_IMF + 1) - pow(dimension.min_value, ALPHA_IMF + 1))
            inverse_value = (pow(norm_factor / value, 1 / -ALPHA_IMF) - pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF)) / \
            (pow(norm_factor / dimension.max_value, 1 / -ALPHA_IMF) - pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF))
            right_distance = abs(inverse_back(dimension, inverse_value + average_density_one_dim) - value)
            left_distance = abs(inverse_back(dimension, inverse_value - average_density_one_dim) - value)
            return max(right_distance, left_distance)
        else:
            return average_density_one_dim / dimension.prior(dimension, value)

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
        batch_num = 0
        num_systems = np.zeros(len(gaussians))
        num_rejected = np.zeros(len(gaussians))
        batches = []
        for index, gaussian in enumerate(gaussians):
            while num_systems[index] < TOTAL_REJECTION_SAMPLES:
                current_batch = dict()
                current_batch['batch_num'] = "gauss_" + str(batch_num)
                current_batch['number'] = index
                param = dict()
                param['filename'] = output_folder + '/distributions.csv'
                param['number'] = index
                param['exploration'] = 0
                command = ["python " + os.getcwd() + "/modules/find_rejection_rate.py '" + json.dumps(param) + "'"]
                current_batch['process'] = run_code(command, current_batch['batch_num'], output_folder, debug, run_on_helios)
                batches.append(current_batch)
                batch_num = batch_num + 1
                num_systems[index] += REJECTION_SAMPLES_PER_BATCH
                if len(batches) == num_batches or (index == len(gaussians) - 1 and num_systems[index] >= TOTAL_REJECTION_SAMPLES):
                    for batch in batches:
                        batch['process'].wait()
                        num_rejected[batch['number']] += float(get_slurm_output(output_folder, batch['batch_num']))
                    batches = []
        (total_rejected, total_sampled) = (0, 0)
        for index, gaussian in enumerate(gaussians):
            total_rejected += num_rejected[index]
            total_sampled += num_systems[index]
        return total_rejected / total_sampled

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
        pdf = multivariate_normal.pdf(samples, mean, variance)
        pdf = pdf * self.biased_weight
        for index, location in enumerate(locations):
            location.properties['q_pdf'] += pdf[index]
