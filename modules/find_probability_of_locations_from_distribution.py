#!/usr/bin/env python

from scipy.stats import multivariate_normal
import json
import sys
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from interface import create_dimensions
import csv
from distributions import Gaussian
from classes import Location
from utils import *

def read_distribution(filename, distribution_number):
    dimensions = create_dimensions()
    with open(filename, 'r') as file:
        distributions = csv.DictReader(file)
        for index, distribution in enumerate(distributions):
            if index == distribution_number:
                means = dict()
                sigma = dict()
                for dimension in dimensions:
                    means[dimension] = float(distribution[dimension.name + '_mean'])
                    sigma[dimension] = float(distribution[dimension.name + '_sigma'])
                return Gaussian(Location(means, {}), Location(sigma, {}))

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        dimensions = create_dimensions()
        params = json.loads(sys.argv[1])
        distribution = read_distribution(params['distribution_filename'], params['number'])
        mean = distribution.mean.to_array()
        variance = np.diagflat(distribution.cov)
        locations = read_samples(params['output_filename'], dimensions)
        [location.transform_variables_to_new_scales() for location in locations]
        samples = []
        for location in locations:
            samples.append(location.to_array())
        pdf = multivariate_normal.pdf(samples, mean, variance, allow_singular = True)
        pdf = pdf * distribution.biased_weight
        for p in pdf:
            print (p)