#!/usr/bin/env python

import numpy as np
import json
import sys
from scipy.stats import multivariate_normal
if (len(sys.argv) > 1):
    params = json.loads(sys.argv[1])
    mean = params['mean']
    cov = params['cov']
    dimension_ranges = params['dimension_ranges']
    num_samples = 1000000

    mask = np.ones(num_samples, dtype = bool)
    current_samples = multivariate_normal.rvs(mean = mean, cov = cov, size = num_samples)
    for i, sample in enumerate(current_samples):
        for index in range(len(dimension_ranges)):
            dimension = dimension_ranges[index]
            mask[i] &= (sample[index] >= dimension[0]) & (sample[index] <= dimension[1])
    print(1 - np.sum(mask) / len(mask))