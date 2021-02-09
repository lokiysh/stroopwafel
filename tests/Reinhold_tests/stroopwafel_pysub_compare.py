#!/usr/bin/env python 
import os, sys
import numpy as np
import pandas as pd
import h5py as h5
import shutil
import argparse
import subprocess

sys.path.append(os.environ.get('SW_ROOT')) # FOR TESTING ONLY - TODO: delete this line and the line below after, restore following line
from stroopwafel_dev import sw, classes, prior, sampler, distributions, constants, utils, run_sw
#from stroopwafel import sw, classes, prior, sampler, distributions, constants, utils, run_sw  

#######################################################
### 
### For User Instructions, see 'docs/sampling.md'
### 
#######################################################
xx = 9
### Set default stroopwafel inputs - these are overwritten by any command-line arguments

num_systems = 100000                # Number of binary systems to evolve                                  
output_folder = '/home/rwillcox/compas/COMPAS/outputs/test_pysub_in_sw/stroop/stroop'+str(xx) # Location of output folder (relative to cwd)                         
random_seed_base = xx                # The initial random seed to increment from                           
num_cores = 25                      # Number of cores to parallelize over 
mc_only = True                      # Exclude adaptive importance sampling (currently not implemented, leave set to True)
run_on_hpc = True                   # Run on slurm based cluster HPC
time_request = None                 # Request HPC time-per-cpu in DD-HH:MM:SS - default is .15s/binary/cpu (only valid for HPC)
debug = True                        # Show COMPAS output/errors
num_per_batch = int(np.ceil(num_systems/num_cores)) # Number of binaries per batch, default num systems per num cores. If mc_only = False, it is highly recommended to change it to a lower value (e.g. int(np.ceil(num_cores/100.)))

### User probably does not need to change these

compas_root = os.environ.get('COMPAS_ROOT_DIR')                              # Location of COMPAS installation
executable = os.path.join(compas_root, 'src/COMPAS')                         # Location of COMAS executable 
h5copyFile = os.path.join(compas_root, 'postProcessing/Folders/H5/PythonScripts/h5copy.py') # Location of COMPAS h5copy File
output_filename = 'samples.csv'                                              # Output filename for the stroopwafel samples
np.random.seed(random_seed_base)                                             # Fix the random seed for the numpy calls


### Command-line only arguments (those that do not work in the grid file)
command_line_args = {
        '--mode' : 'BSE',
        '--logfile-type' : 'HDF5',
        '--rlof-printing' : 'TRUE',
        '--number-of-systems' : 10,
        '--maximum-evolution-time' : 13700.0,
        '--maximum-number-timestep-iterations' : 99999,
        '--timestep-multiplier' : 1.0,
        '--log-level' : 0,
        '--debug-level' : 0,
        '--hdf5-chunk-size' : 100000,
        '--hdf5-buffer-size' : 1
        }


##############################################################################################################
###
### User should set their desired parameters and distributions in the functions below. 
### See function descriptions, or 'docs/sampling.md' for further details.
###
### Any parameter which is not set will fallback to the COMPAS default. COMPAS defaults can be viewed using:
###     ./COMPAS --help
### or for just the parameter names:
###     ./COMPAS -h
###

def create_dimensions():
    """
    This function creates Stroopwafel Dimensions, which handle the sampling and prior evaluation for any parameter
    which should be "stroopwafelized". Any parameter here which will be included as a COMPAS argument should be named
    as it will appear in the COMPAS gridfile. 

    Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in classes.py for more.
    It takes the name of the dimension, its max and min value. 
    The Sampler class will tell how to sample this dimension. Similarly, prior tells it how it calculates the prior. You can find more info in their respective modules.
    OUT:
        As Output, this should return a list containing all the instances of Dimension class.
    """
    m1 = classes.Dimension('--initial-mass-1', 5, 150, sampler.kroupa, prior.kroupa)
    q = classes.Dimension('q', 0.01, 1, sampler.uniform, prior.uniform, should_print = False)
    a = classes.Dimension('--semi-major-axis', .01, 1000, sampler.flat_in_log, prior.flat_in_log) 
    return [m1, q, a ]

def update_properties(locations, dimensions):
    """
    This function creates Stroopwafel Locations, which are all the parameters which do not need to be "stroopwafelized",
    but which should still be included in the gridfile. This includes constants, distribution choices, or random numbers
    which do not contribute to the AIS resampling. This also includes dependent variables, such as the mass of the secondary
    (when we would like to stroopwafelize the primary mass and mass ratio). 

    Values here can be constants, strings, or numpy random variables (of size 1). 
    IN:
        locations (list(Location)) : A list containing objects of Location class in classes.py. 
        You can play with them and update whatever fields you like or add more in the property (which is a dictionary)
    OUT: Not Required
    """
    m1 = dimensions[0]
    q = dimensions[1]
    for location in locations:
        location.properties['--initial-mass-2'] = location.dimensions[m1] * location.dimensions[q]
        location.properties['--metallicity'] = constants.METALLICITY_SOL
        location.properties['--eccentricity'] = 0
        location.properties['--kick-theta-1'] = np.arccos(np.random.uniform(-1, 1)) - np.pi / 2   
        location.properties['--kick-theta-2'] = np.arccos(np.random.uniform(-1, 1)) - np.pi / 2   
        location.properties['--kick-phi-1'] = np.random.uniform(0, 2 * np.pi)
        location.properties['--kick-phi-2'] = np.random.uniform(0, 2 * np.pi)
        location.properties['--kick-mean-anomaly-1'] = np.random.uniform(0, 2 * np.pi)
        location.properties['--kick-mean-anomaly-2'] = np.random.uniform(0, 2 * np.pi)

        #location.properties['--kick-magnitude-1'] =  # (default = 0.000000 km s^-1 )
        #location.properties['--kick-magnitude-2'] =  # (default = 0.000000 km s^-1 )
        #location.properties['--kick-magnitude-random-1'] =  # (default = uniform random number [0.0, 1.0))
        #location.properties['--kick-magnitude-random-2'] =  # (default = uniform random number [0.0, 1.0))




        ### COMPAS Fiducial Model, from pythonSubmit.py
        location.properties['--use-mass-loss'] = 'TRUE'
        location.properties['--mass-transfer'] = 'TRUE'
        location.properties['--pair-instability-supernovae'] = 'TRUE'
        location.properties['--pulsational-pair-instability'] = 'TRUE'
        location.properties['--common-envelope-allow-main-sequence-survive'] = 'TRUE'
        location.properties['--allow-rlof-at-birth'] = 'TRUE'
        #location.properties['--metallicity'] = 0.0142
        location.properties['--common-envelope-alpha'] = 1.0
        location.properties['--common-envelope-lambda'] = 0.1
        location.properties['--common-envelope-slope-kruckow'] = -0.8333333333333334
        location.properties['--common-envelope-alpha-thermal'] = 1.0
        location.properties['--common-envelope-lambda-multiplier'] = 1.0
        location.properties['--luminous-blue-variable-multiplier'] = 1.5
        location.properties['--overall-wind-mass-loss-multiplier'] = 1.0
        location.properties['--wolf-rayet-multiplier'] = 1.0
        location.properties['--cool-wind-mass-loss-multiplier'] = 1.0
        location.properties['--mass-transfer-fa'] = 0.5
        location.properties['--mass-transfer-jloss'] = 1.0
        #location.properties['--initial-mass-min'] = 5.0
        #location.properties['--initial-mass-max'] = 150.0
        #location.properties['--initial-mass-power'] = 0.0
        #location.properties['--semi-major-axis-min'] = 0.01
        #location.properties['--semi-major-axis-max'] = 1000.0
        #location.properties['--mass-ratio-min'] = 0.01
        #location.properties['--mass-ratio-max'] = 1.0
        location.properties['--minimum-secondary-mass'] = 0.1
        #location.properties['--eccentricity-min'] = 0.0
        #location.properties['--eccentricity-max'] = 1.0
        #location.properties['--metallicity-min'] = 0.0001
        #location.properties['--metallicity-max'] = 0.03
        location.properties['--pulsar-birth-magnetic-field-distribution-min'] = 11.0
        location.properties['--pulsar-birth-magnetic-field-distribution-max'] = 13.0
        location.properties['--pulsar-birth-spin-period-distribution-min'] = 10.0
        location.properties['--pulsar-birth-spin-period-distribution-max'] = 100.0
        location.properties['--pulsar-magnetic-field-decay-timescale'] = 1000.0
        location.properties['--pulsar-magnetic-field-decay-massscale'] = 0.025
        location.properties['--pulsar-minimum-magnetic-field'] = 8.0
        #location.properties['--orbital-period-min'] = 1.1
        #location.properties['--orbital-period-max'] = 1000
        location.properties['--kick-magnitude-sigma-CCSN-NS'] = 265.0
        location.properties['--kick-magnitude-sigma-CCSN-BH'] = 265.0
        location.properties['--fix-dimensionless-kick-magnitude'] = -1
        location.properties['--kick-direction-power'] = 0.0
        #location.properties['--random-seed'] = 0
        location.properties['--mass-transfer-thermal-limit-C'] = 10.0
        location.properties['--eddington-accretion-factor'] = 1
        location.properties['--pisn-lower-limit'] = 60.0
        location.properties['--pisn-upper-limit'] = 135.0
        location.properties['--ppi-lower-limit'] = 35.0
        location.properties['--ppi-upper-limit'] = 60.0
        location.properties['--maximum-neutron-star-mass'] = 2.5
        location.properties['--kick-magnitude-sigma-ECSN'] = 30.0
        location.properties['--kick-magnitude-sigma-USSN'] = 30.0
        location.properties['--kick-scaling-factor'] = 1.0
        location.properties['--maximum-mass-donor-nandez-ivanova'] = 2.0
        location.properties['--common-envelope-recombination-energy-density'] = 15000000000000.0
        location.properties['--common-envelope-mass-accretion-max'] = 0.1
        location.properties['--common-envelope-mass-accretion-min'] = 0.04
        location.properties['--zeta-main-sequence'] = 2.0
        location.properties['--zeta-radiative-envelope-giant'] = 6.5
        location.properties['--kick-magnitude-max'] = -1.0
        location.properties['--muller-mandel-kick-multiplier-BH'] = 200.0
        location.properties['--muller-mandel-kick-multiplier-NS'] = 400.0
        location.properties['--neutrino-mass-loss-BH-formation-value'] = 0.1
        location.properties['--case-BB-stability-prescription'] = 'ALWAYS_STABLE'
        location.properties['--chemically-homogeneous-evolution'] = 'PESSIMISTIC'
        location.properties['--luminous-blue-variable-prescription'] = 'BELCZYNSKI'
        location.properties['--mass-loss-prescription'] = 'VINK'
        location.properties['--mass-transfer-angular-momentum-loss-prescription'] = 'ISOTROPIC'
        location.properties['--mass-transfer-accretion-efficiency-prescription'] = 'THERMAL'
        location.properties['--mass-transfer-rejuvenation-prescription'] = 'STARTRACK'
        #location.properties['--initial-mass-function'] = 'KROUPA'
        #location.properties['--semi-major-axis-distribution'] = 'FLATINLOG'
        #location.properties['--orbital-period-distribution'] = 'FLATINLOG'
        #location.properties['--mass-ratio-distribution'] = 'FLAT'
        #location.properties['--eccentricity-distribution'] = 'ZERO'
        #location.properties['--metallicity-distribution'] = 'ZSOLAR'
        location.properties['--rotational-velocity-distribution'] = 'ZERO'
        location.properties['--remnant-mass-prescription'] = 'FRYER2012'
        location.properties['--fryer-supernova-engine'] = 'DELAYED'
        location.properties['--black-hole-kicks'] = 'FALLBACK'
        location.properties['--kick-magnitude-distribution'] = 'MAXWELLIAN'
        location.properties['--kick-direction'] = 'ISOTROPIC'
        #location.properties['--output-path'] = /home/rwillcox/astro/compas/COMPAS/defaults
        location.properties['--common-envelope-lambda-prescription'] = 'LAMBDA_NANJING'
        location.properties['--stellar-zeta-prescription'] = 'SOBERMAN'
        location.properties['--mass-transfer-thermal-limit-accretor'] = 'CFACTOR'
        location.properties['--pulsational-pair-instability-prescription'] = 'MARCHANT'
        location.properties['--neutron-star-equation-of-state'] = 'SSE'
        location.properties['--pulsar-birth-magnetic-field-distribution'] = 'ZERO'
        location.properties['--pulsar-birth-spin-period-distribution'] = 'ZERO'
        location.properties['--common-envelope-mass-accretion-prescription'] = 'ZERO'
        location.properties['--envelope-state-prescription'] = 'LEGACY'
        location.properties['--neutrino-mass-loss-BH-formation'] = 'FIXED_MASS'



##############################################################################################################
###
### If using full Adaptive Importance Sampling, user should set their desired systems of interest,
### selection effects, and rejected systems, and below
###

def interesting_systems(batch):
    """
    This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
    IN:
        batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
    OUT:
        Number of interesting systems
        In the below example, I define all the NSs as interesting, so I read the files, get the SEED from the system_params file and define the key is_hit in the end for all interesting systems 
    """
    try:
        folder = os.path.join(output_folder, batch['output_container'])
        BSE_file = h5.File(folder + '/' + batch['output_container'] + '.h5')
        system_parameters = BSE_file['BSE_System_Parameters']
        seeds = system_parameters['SEED'][()]
        for index, sample in enumerate(batch['samples']):
            seed = seeds[index]
            sample.properties['SEED'] = seed
            sample.properties['is_hit'] = 0
            sample.properties['batch'] = batch['number']
        double_compact_objects = BSE_file['BSE_Double_Compact_Objects']
        dco = np.logical_and(double_compact_objects['Merges_Hubble_Time'][()] == 1, \
            np.logical_and(double_compact_objects['Stellar_Type(1)'][()] == 14, double_compact_objects['Stellar_Type(2)'][()] == 14))
        interesting_systems_seeds = set(double_compact_objects['SEED'][()][dco])
        for sample in batch['samples']:
            if sample.properties['SEED'] in interesting_systems_seeds:
                sample.properties['is_hit'] = 1
        BSE_file.close()
        return len(dco)
    except KeyError as error:
        return 0

def selection_effects(sw):
    """
    This is not a mandatory function, it was written to support selection effects
    Fills in selection effects for each of the distributions
    IN:
        sw (Stroopwafel) : Stroopwafel object
    """
    if hasattr(sw, 'adapted_distributions'):
        biased_masses = []
        rows = []
        for distribution in sw.adapted_distributions:
            folder = os.path.join(output_folder, 'batch_' + str(int(distribution.mean.properties['batch'])))
            h5_file = h5.File(folder + '/' + 'batch_' + str(int(distribution.mean.properties['batch'])) + '.h5')
            dco = h5_file['BSE_Double_Compact_Objects']
            row = dco['SEED'][()] == distribution.mean.properties['SEED']
            rows.append([dco['Mass(1)'][()][row], dco['Mass(2)'][()][row]])
            biased_masses.append(np.power(max(rows[-1]), 2.2))
        # update the weights
        mean = np.mean(biased_masses)
        for index, distribution in enumerate(sw.adapted_distributions):
            distribution.biased_weight = np.power(max(rows[index]), 2.2) / mean
        
def rejected_systems(locations, dimensions):
    """
    This method takes a list of locations and marks the systems which can be
    rejected by the prior distribution
    IN:
        locations (List(Location)): list of location to inspect and mark rejected
    OUT:
        num_rejected (int): number of systems which can be rejected
    """
    m1 = dimensions[0]
    q = dimensions[1]
    a = dimensions[2]
    mass_1 = [location.dimensions[m1] for location in locations]
    mass_2 = [location.properties['--initial-mass-2'] for location in locations]
    metallicity = [location.properties['--metallicity'] for location in locations]
    eccentricity = [location.properties['--eccentricity'] for location in locations]
    num_rejected = 0
    for index, location in enumerate(locations):
        radius_1 = utils.get_zams_radius(mass_1[index], metallicity[index])
        radius_2 = utils.get_zams_radius(mass_2[index], metallicity[index])
        roche_lobe_tracker_1 = radius_1 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_1[index], mass_2[index]))
        roche_lobe_tracker_2 = radius_2 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_2[index], mass_1[index]))
        location.properties['is_rejected'] = 0
        if (mass_2[index] < constants.MINIMUM_SECONDARY_MASS) or (location.dimensions[a] <= (radius_1 + radius_2)) \
        or roche_lobe_tracker_1 > 1 or roche_lobe_tracker_2 > 1:
            location.properties['is_rejected'] = 1
            num_rejected += 1
    return num_rejected

##############################################################################################################
###
### Run the stroopwafel interface with all the specified parameters
###

if __name__ == '__main__':

    ### Import and assign input parameters for stroopwafel 
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_systems', help = 'Total number of systems', type = int, default = num_systems)  
    parser.add_argument('--num_cores', help = 'Number of cores to run in parallel', type = int, default = num_cores)
    parser.add_argument('--num_per_batch', help = 'Number of systems to generate in one core', type = int, default = num_per_batch)
    parser.add_argument('--debug', help = 'If debug of COMPAS is to be printed', type = bool, default = debug)
    parser.add_argument('--mc_only', help = 'If run in MC simulation mode only', type = bool, default = mc_only)
    parser.add_argument('--run_on_hpc', help = 'If we are running on a (slurm-based) HPC', type = bool, default = run_on_hpc)
    parser.add_argument('--output_filename', help = 'Output filename', default = output_filename)
    parser.add_argument('--output_folder', help = 'Output folder name', default = output_folder)
    namespace, extra_params = parser.parse_known_args()

    ### Define the parameters to the constructor of stroopwafel
    TOTAL_NUM_SYSTEMS = namespace.num_systems                          # Total number of systems to evolve
    NUM_CPU_CORES = namespace.num_cores                                # Number of cpu cores you want to run in parallel
    NUM_SYSTEMS_PER_BATCH = namespace.num_per_batch                    # Number of systems generated per run per cpu 
    debug = namespace.debug                                            # Print the logs given by the external program (like COMPAS)
    run_on_hpc = namespace.run_on_hpc                                  # Run on a clustered system helios, rather than your pc
    mc_only = namespace.mc_only                                        # If you dont want to do the refinement phase and just do random mc exploration
    output_filename = namespace.output_filename                        # The name of the output file
    output_folder = os.path.join(os.getcwd(), namespace.output_folder) # Name of the output folder
    [extra_params.extend([key, val]) for key, val in command_line_args.items()]

    ### Run Stroopwafel with all the specified arguments and parameter distributions
    run_sw.run_stroopwafel(output_folder, output_filename, random_seed_base, 
        executable, extra_params, 
        TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_BATCH, 
        time_request, debug, run_on_hpc, mc_only,
        create_dimensions, update_properties, interesting_systems,
        selection_effects, rejected_systems)

    ### Convert output to h5 format
    subprocess.call("python3 " + h5copyFile + ' -o ' + output_folder+'/COMPAS_Output.h5 ' + output_folder+'/*', shell=True)

