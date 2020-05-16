#!/usr/bin/env python

import numpy as np
import json
import sys
from scipy.stats import multivariate_normal
from distributions import Gaussian
from classes import Location
import os
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from interface import create_dimensions, rejected_systems
import csv

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
        distribution = read_distribution(params['filename'], params['number'])
        total_samples = 0
        total_rejected = 0
        for i in range(100):
            num_samples = int(1e5)
            (locations, mask) = distribution.run_sampler(num_samples, dimensions)
            rejected = num_samples - np.sum(mask)
            locations = np.asarray(locations)[mask]
            [location.revert_variables_to_original_scales() for location in locations]
            rejected += rejected_systems(locations, dimensions)
            total_samples += num_samples
            total_rejected += rejected
        print (total_rejected / total_samples)
