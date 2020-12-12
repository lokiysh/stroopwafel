"""
This file will contain all the constants
"""
ALPHA_IMF = -2.3

R_COEFF =\
    [[1.71535900,    0.62246212,     -0.92557761,    -1.16996966,    -0.30631491],\
    [6.59778800,    -0.42450044,    -12.13339427,   -10.73509484,   -2.51487077],\
    [10.08855000,   -7.11727086,    -31.67119479,   -24.24848322,   -5.33608972],\
    [1.01249500,    0.32699690,     -0.00923418,    -0.03876858,    -0.00412750],\
    [0.07490166,    0.02410413,     0.07233664,     0.03040467,     0.00197741],\
    [0.01077422,    0.00000000,     0.00000000,     0.00000000,     0.00000000],\
    [3.08223400,    0.94472050,     -2.15200882,    -2.49219496,    -0.63848738],\
    [17.84778000,   -7.45345690,    -48.96066856,   -40.05386135,   -9.09331816],\
    [0.00022582,    -0.00186899,    0.00388783,     0.00142402,     -0.00007671]]
MINIMUM_SECONDARY_MASS = 0.1
R_SOL_TO_AU = 0.00465047
METALLICITY_SOL = 0.0142
ZSOL = 0.02
REJECTION_SAMPLES_PER_BATCH = 1e4
TOTAL_REJECTION_SAMPLES = 1e6
NUM_GENERATIONS = 10
MIN_ENTROPY_CHANGE = 0.01
KAPPA = 1.0
