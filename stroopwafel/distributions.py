from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal
from .classes import Location
import numpy as np
from .utils import *
import json
from .constants import *
from . import sampler as sp

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
        # For each dimension, draw samples and check whether they are within their given minimum and maximum value bounds
        for dimension in headers:
            current_samples = dimension.run_sampler(num_samples)
            mask &= dimension.is_sample_within_bounds(current_samples) # mask the systems that are within the bounds
            samples.append(current_samples)
        samples = list(map(list, zip(*samples)))
        locations = [Location(dict(zip(headers, row)), {}) for row in samples]
        return (locations, mask)

    def calculate_probability_of_locations_from_distribution(self, locations):
        pass

    def calculate_rejection_rate(self, update_properties, rejected_systems, dimensions):
        """
        This function samples a small population to calculate what fraction of the samples should be rejected.
        Samples can be either rejected because one or more of their properties lie outside the property bounds,
        or because the combination of initial properties doesn't allow the system to be evolved.
        IN:
            update_properties (function) : method that defines the properties on which we do not perform adaptive importance sampling
            rejected_systems (function) : method that calculates which systems can not evolve from birth and should therefore be rejected
            dimensions (dict(Dimension : float)) : dictionary of mapping between class Dimension and its float value
        OUT:
            (float) : fraction of rejected systems
        """
        num_samples = int(TOTAL_REJECTION_SAMPLES) # found in constants.py
        (locations, mask) = self.run_sampler(num_samples) # mask masks out all systems not within property bounds
        rejected = num_samples - np.sum(mask) # samples rejected by sampling outside the property bounds
        locations = np.asarray(locations)[mask]
        [location.revert_variables_to_original_scales() for location in locations]
        update_properties(locations, dimensions) # the update_properties() function is defined in the interface file
        rejected += rejected_systems(locations, dimensions) # samples rejected because the systems couldn't be evolved with the initial properties
        return rejected / num_samples

class Gaussian(NDimensionalDistribution):
    """
    This class inherits from NDimensionalDistribution. It will be used during the refinement phase to draw adapted distributions.
    mean (Location) : The mean location of the gaussian. It is of Location type and therefore stores all the dimensions
    sigma (Location) : The standard deviation of the gaussian. Also of Location class type
    kappa (float) : An independent scaling factor describing how wide the gaussian can be.
    """
    def __init__(self, mean, sigma = None, rejection_rate = 0, alpha = 1, kappa = 1, cov = None):
        self.mean = mean
        self.sigma = sigma
        self.kappa = kappa
        # Since we are using a N-dimensional Gaussian distribution, we need to use the covariance matrix
        # If only the standard deviations of the dimensions are given, turn them into a diagonal covariance matrix
        if cov == None:
            if sigma != None:
                self.cov = np.diagflat(np.power(np.asarray(self.sigma.to_array()), 2))
            else:
                raise Exception("Either sigma or cov value is needed")
        else:
            self.cov = cov
        self.rejection_rate = rejection_rate
        self.biased_weight = 1 # factor that allows us to bias towards drawing more samples from certain adapted Gaussian distributions

    def run_sampler(self, num_samples, dimensions, consider_rejection_rate = False):
        """
        This function tells the class how to draw the samples for this class
        IN:
            num_samples (int) : How many samples to draw
            dimensions (dict(Dimension : float)) : dictionary of mapping between class Dimension and its float value
            consider_rejection_rate (bool) : If we should consider rejection rate while drawing samples. If True, it will draw more samples (based on rejection_rate), such that the num_samples are within the bounds
        OUT:
            (list(Location), list(bool)) : A pair of lists (both of size num_samples), the first describing the samples drawn, and the second telling which of the samples are within the bounds of the respective variables
        """
        if consider_rejection_rate == True:
            # If we know all systems will be rejected, we shouldn't bother sampling at all
            if self.rejection_rate == 1:
                return ([], [])
            num_samples = int(2 * np.ceil(num_samples / (1 - self.rejection_rate))) # we want to have more samples if we are rejecting some of them
        # For each dimension, draw samples and check whether they are within their given min and max value
        num_samples = int(num_samples * self.biased_weight) # bias sampling towards certain adapted distributions
        mask = np.ones(num_samples, dtype = bool)
        current_samples = multivariate_normal.rvs(mean = self.mean.to_array(), cov = self.cov, size = num_samples) # draws samples randomly from the adapted distribution
        headers = sorted(dimensions, key = lambda d: d.name)
        for index, dimension in enumerate(headers):
            mask &= dimension.is_sample_within_bounds(current_samples[:, index])
        locations = [Location(dict(zip(headers, row)), {}) for row in current_samples]
        return (locations, mask)

    @classmethod
    def draw_distributions(self, hit_locations, average_density_one_dim, kappa = 1):
        """
        Function that creates Gaussian distributions around the target systems
        IN:
            hit_locations (list(Location)) : The locations which have hits to draw the gaussians
            average_density_one_dim (float) : The average density of hits to be expected per dimension
            kappa (float) : An independent scaling factor describing how wide the gaussian can be.
        OUT:
            gaussians (list(Gaussian)) : list of gaussians drawn
        """
        gaussians = []
        # Calculate the standard deviation for each Gaussian. The mean does not
        # have to be calculated as this will be the value of the property
        for hit in hit_locations:
            sigma = dict()
            for variable, val in hit.dimensions.items():
                sigma[variable] = self.__calculate_sigma(average_density_one_dim, variable, val)
            gaussians.append(Gaussian(hit, sigma = Location(sigma, {}), kappa = kappa))
        return gaussians

    @classmethod
    def __calculate_sigma(self, average_density_one_dim, dimension, value):
        """
        Function that determines the standard deviation for the adapted distributions. The standard deviation is
        derived from the average distance to the next sample in each dimension, which depends on the number of
        effective samples drawn per dimension during the exploration phase and prior value of the sampled systems.
        See eq. 11 from Broekgaarden et al. 2019
        IN:
            average_density_one_dim (hit_locations (list(Location)) : The average density of hits to be expected per dimension (the inverse of number of effective samples drawn)
            dimension (dict(Dimension : float)) : dictionary of mapping between class Dimension and its float value
            value (float) : The value of this dimensions
        OUT:
            (float) : standard deviation for the adapted distribution created around the target system
        """
        # The Kroupa mass function is more complicated.
        # STEPHEN, DO YOU KNOW WHAT HAPPENS HERE? I FORGOT
        if dimension.sampler.__name__ == sp.kroupa.__name__:
            norm_factor = (ALPHA_IMF + 1) / (pow(dimension.max_value, ALPHA_IMF + 1) - pow(dimension.min_value, ALPHA_IMF + 1)) # normalizing constant
            inverse_value = (pow(norm_factor / value, 1 / -ALPHA_IMF) - pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF)) / \
            (pow(norm_factor / dimension.max_value, 1 / -ALPHA_IMF) - pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF))
            right_distance = abs(inverse_back(dimension, inverse_value + average_density_one_dim) - value)
            left_distance = abs(inverse_back(dimension, inverse_value - average_density_one_dim) - value)
            return max(right_distance, left_distance)
        # For prior distributions that are simple functions we can directly use the equation 11
        else:
            return average_density_one_dim / dimension.prior(dimension, value)

    @classmethod
    def calculate_rejection_rate(self, gaussians, update_properties, rejected_systems, dimensions):
        """
        This function samples a small population to calculate what fraction of the samples should be rejected.
        Samples can be either rejected because one or more of their properties lie outside the property bounds,
        or because the combination of initial properties doesn't allow the system to be evolved.
        IN:
            gaussians (List(Gaussian)) : The gaussians to calculate the rejection rate
            update_properties (function) : method that defines the properties on which we do not perform adaptive importance sampling
            rejected_systems (function) : method that calculates which systems can not evolve from birth and should therefore be rejected
            dimensions (dict(Dimension : float)) : dictionary of mapping between class Dimension and its float value
        OUT:
            (float) : fraction of rejected systems
        """
        rejected = 0
        num_samples = int(REJECTION_SAMPLES_PER_BATCH) # found in constants.py
        # Calculate the rejection rate seperately for each adapted gaussian distribution
        for index, gaussian in enumerate(gaussians):
            (locations, mask) = gaussian.run_sampler(num_samples, dimensions) # mask masks out all systems not within property bounds
            rejected += num_samples - np.sum(mask) # samples rejected by sampling outside the property bounds
            locations = np.asarray(locations)[mask]
            [location.revert_variables_to_original_scales() for location in locations]
            update_properties(locations, dimensions) # the update_properties() function is defined in the interface file
            rejected += rejected_systems(locations, dimensions) # samples rejected because the systems couldn't be evolved with the initial properties
        return rejected / (num_samples * len(gaussians))

    @classmethod
    def calculate_probability_of_locations_from_distribution(self, locations, gaussians):
        """
        Given a list of locations, calculates the probability of drawing each location from the given gaussian.
        These probabilities are used to compute the mixture weights after the end of the refinement phase in sw.py and genais.py
        IN:
            locations (list(Location)) : list of locations
            gaussians (List(Gaussian)) : The gaussians to calculate the rejection rate
        """
        samples = []
        for location in locations:
            samples.append(location.to_array())
        pdfs = np.zeros(len(locations))
        # For each gaussian, check what the probabilities are for each sample being drawn from this gaussian
        # and combine to get the effective adapted sampling distribution
        for index, gaussian in enumerate(gaussians):
            pdf = multivariate_normal.pdf(samples, gaussian.mean.to_array(), gaussian.cov, allow_singular = True)
            pdf = pdf * gaussian.biased_weight # if we were biased towards certain gaussian, we were more or less likely to draw samples from this gaussian
            pdfs += pdf
        for index, location in enumerate(locations):
            location.properties['q_pdf'] = pdfs[index]
