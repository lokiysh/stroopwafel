#!/usr/bin/env python 
import os, sys
import shutil
import argparse
import subprocess
import h5py as h5
import numpy as np
import pandas as pd
import scipy.stats as ss    # Only necessary for some distributions

sys.path.append(os.environ.get('SW_ROOT')) # FOR TESTING ONLY - TODO: delete this line and the line below after, restore following line
from stroopwafel_dev import sw, classes, prior, sampler, distributions, constants, utils, run_sw
#from stroopwafel import sw, classes, prior, sampler, distributions, constants, utils, run_sw  

#######################################################
### 
### For User Instructions, see 'docs/sampling.md'
### 
#######################################################

### Set default stroopwafel inputs - these are overwritten by any command-line arguments

num_systems = 40000                 # Number of binary systems to evolve                                  
output_folder = 'output/'           # Location of output folder (relative to cwd)                         
random_seed_base = 0                # The initial random seed to increment from                           
num_cores = 4                       # Number of cores to parallelize over 
mc_only = True                      # Exclude adaptive importance sampling (currently not implemented, leave set to True)
run_on_hpc = False                  # Run on slurm based cluster HPC
time_request = None                 # Request HPC time-per-cpu in DD-HH:MM:SS - default is .15s/binary/cpu (only valid for HPC)
debug = True                        # Show COMPAS output/errors
num_per_batch = int(np.ceil(num_systems/num_cores)) # Number of binaries per batch, default num systems per num cores. If mc_only = False, it is highly recommended to change it to a lower value (e.g. int(np.ceil(num_cores/100.)))

### User probably does not need to change these

compas_root = os.environ.get('COMPAS_ROOT_DIR')                              # Location of COMPAS installation
executable = os.path.join(compas_root, 'src/COMPAS')                         # Location of COMAS executable 
h5copyFile = os.path.join(compas_root, 'postProcessing/Folders/H5/PythonScripts/h5copy.py') # Location of COMPAS h5copy File
output_filename = 'samples.csv'                                              # Output filename for the stroopwafel samples
np.random.seed(random_seed_base)                                             # Fix the random seed for the numpy calls

### Get the Deller kicks
dellerData = np.loadtxt('pulsarPosteriors.txt')

### Command-line only arguments 
"""
Include here any COMPAS parameters which are constant for the entire run. 
These will be entered as command line arguments, not in the grid file, and
will show up correctly in the Run_Details file. This will also reduce the
diskspace of the grid files. 
"""
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
        '--hdf5-buffer-size' : 1,
        #'--metallicity' : constants.METALLICITY_SOL,
        #'--eccentricity' : 0,
        '--evolve-unbound-systems' : 'TRUE',
        #'--kick-magnitude-1' :  # (default = 0.000000 km s^-1 ),
        #'--kick-magnitude-2' :  # (default = 0.000000 km s^-1 ),
        #'--kick-magnitude-random-1' :  # (default = uniform random number [0.0, 1.0)),
        #'--kick-magnitude-random-2' :  # (default = uniform random number [0.0, 1.0)),
        ### COMPAS Fiducial Model, from pythonSubmit.py,
        '--use-mass-loss' : 'TRUE',
        '--mass-transfer' : 'TRUE',
        '--pair-instability-supernovae' : 'TRUE',
        '--pulsational-pair-instability' : 'TRUE',
        '--common-envelope-allow-main-sequence-survive' : 'TRUE',
        '--allow-rlof-at-birth' : 'TRUE',
        #'--metallicity' : 0.0142,
        '--common-envelope-alpha' : 1.0,
        '--common-envelope-lambda' : 0.1,
        '--common-envelope-slope-kruckow' : -0.8333333333333334,
        '--common-envelope-alpha-thermal' : 1.0,
        '--common-envelope-lambda-multiplier' : 1.0,
        '--luminous-blue-variable-multiplier' : 1.5,
        '--overall-wind-mass-loss-multiplier' : 1.0,
        '--wolf-rayet-multiplier' : 1.0,
        '--cool-wind-mass-loss-multiplier' : 1.0,
        '--mass-transfer-fa' : 0.5,
        '--mass-transfer-jloss' : 1.0,
        #'--initial-mass-min' : 5.0,
        #'--initial-mass-max' : 150.0,
        #'--initial-mass-power' : 0.0,
        #'--semi-major-axis-min' : 0.01,
        #'--semi-major-axis-max' : 1000.0,
        #'--mass-ratio-min' : 0.01,
        #'--mass-ratio-max' : 1.0,
        #'--kick-theta-1' : np.arccos(np.random.uniform(-1, 1)) - np.pi / 2   ,
        #'--kick-theta-2' : np.arccos(np.random.uniform(-1, 1)) - np.pi / 2   ,
        #'--kick-phi-1' : np.random.uniform(0, 2 * np.pi),
        #'--kick-phi-2' : np.random.uniform(0, 2 * np.pi),
        #'--kick-mean-anomaly-1' : np.random.uniform(0, 2 * np.pi),
        #'--kick-mean-anomaly-2' : np.random.uniform(0, 2 * np.pi),
        '--minimum-secondary-mass' : 0.1,
        #'--eccentricity-min' : 0.0,
        #'--eccentricity-max' : 1.0,
        #'--metallicity-min' : 0.0001,
        #'--metallicity-max' : 0.03,
        '--pulsar-birth-magnetic-field-distribution-min' : 11.0,
        '--pulsar-birth-magnetic-field-distribution-max' : 13.0,
        '--pulsar-birth-spin-period-distribution-min' : 10.0,
        '--pulsar-birth-spin-period-distribution-max' : 100.0,
        '--pulsar-magnetic-field-decay-timescale' : 1000.0,
        '--pulsar-magnetic-field-decay-massscale' : 0.025,
        '--pulsar-minimum-magnetic-field' : 8.0,
        #'--orbital-period-min' : 1.1,
        #'--orbital-period-max' : 1000,
        '--kick-magnitude-sigma-CCSN-NS' : 265.0,
        '--kick-magnitude-sigma-CCSN-BH' : 265.0,
        '--fix-dimensionless-kick-magnitude' : -1,
        '--kick-direction-power' : 0.0,
        #'--random-seed' : 0,
        '--mass-transfer-thermal-limit-C' : 10.0,
        '--eddington-accretion-factor' : 1,
        '--pisn-lower-limit' : 60.0,
        '--pisn-upper-limit' : 135.0,
        '--ppi-lower-limit' : 35.0,
        '--ppi-upper-limit' : 60.0,
        '--maximum-neutron-star-mass' : 2.5,
        '--kick-magnitude-sigma-ECSN' : 30.0,
        '--kick-magnitude-sigma-USSN' : 30.0,
        '--kick-scaling-factor' : 1.0,
        '--maximum-mass-donor-nandez-ivanova' : 2.0,
        '--common-envelope-recombination-energy-density' : 15000000000000.0,
        '--common-envelope-mass-accretion-max' : 0.1,
        '--common-envelope-mass-accretion-min' : 0.04,
        '--zeta-main-sequence' : 2.0,
        '--zeta-radiative-envelope-giant' : 6.5,
        '--kick-magnitude-max' : -1.0,
        '--muller-mandel-kick-multiplier-BH' : 200.0,
        '--muller-mandel-kick-multiplier-NS' : 400.0,
        '--neutrino-mass-loss-BH-formation-value' : 0.1,
        '--case-BB-stability-prescription' : 'ALWAYS_STABLE',
        '--chemically-homogeneous-evolution' : 'PESSIMISTIC',
        '--luminous-blue-variable-prescription' : 'BELCZYNSKI',
        '--mass-loss-prescription' : 'VINK',
        '--mass-transfer-angular-momentum-loss-prescription' : 'ISOTROPIC',
        '--mass-transfer-accretion-efficiency-prescription' : 'THERMAL',
        '--mass-transfer-rejuvenation-prescription' : 'STARTRACK',
        #'--initial-mass-function' : 'KROUPA',
        #'--semi-major-axis-distribution' : 'FLATINLOG',
        #'--orbital-period-distribution' : 'FLATINLOG',
        #'--mass-ratio-distribution' : 'FLAT',
        #'--eccentricity-distribution' : 'ZERO',
        #'--metallicity-distribution' : 'ZSOLAR',
        '--rotational-velocity-distribution' : 'ZERO',
        '--remnant-mass-prescription' : 'FRYER2012',
        '--fryer-supernova-engine' : 'DELAYED',
        '--black-hole-kicks' : 'FALLBACK',
        '--kick-magnitude-distribution' : 'MAXWELLIAN',
        '--kick-direction' : 'ISOTROPIC',
        #'--output-path' : /home/rwillcox/astro/compas/COMPAS/defaults,
        '--common-envelope-lambda-prescription' : 'LAMBDA_NANJING',
        '--stellar-zeta-prescription' : 'SOBERMAN',
        '--mass-transfer-thermal-limit-accretor' : 'CFACTOR',
        '--pulsational-pair-instability-prescription' : 'MARCHANT',
        '--neutron-star-equation-of-state' : 'SSE',
        '--pulsar-birth-magnetic-field-distribution' : 'ZERO',
        '--pulsar-birth-spin-period-distribution' : 'ZERO',
        '--common-envelope-mass-accretion-prescription' : 'ZERO',
        '--envelope-state-prescription' : 'LEGACY',
        '--neutrino-mass-loss-BH-formation' : 'FIXED_MASS',
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
    This function creates Stroopwafel Locations, which are all the parameters which are randomly 
    sampled and which should be included in the gridfile, but do not need to be "stroopwafelized". 
    This includes dependent variables, such as the mass of the secondary which do not contribute 
    to the AIS resampling. (when we would like to stroopwafelize the primary mass and mass ratio). 

    This does not include any parameters which will have a fixed value, including constants or 
    distribution choices. Those should be specified above, in command_line_args. 

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

        ### Grab kicks from Deller+
        
        # Kick 1
        useSpeed = [0, 0]
        for primaryOrSecondary in range(2):
            findNewPulsar = True
            while findNewPulsar:
                whichPulsar = np.random.randint(0, 81)
                fullData = dellerData[whichPulsar]
                if np.max(fullData) > 50:
                    findNewPulsar = False # good pulsar, use it

            while useSpeed[primaryOrSecondary] < 50:
                # randomly choose which pulsar
                useSpeed[primaryOrSecondary] = np.random.choice(fullData)

            # Deproject to 3D
            deproj_factor = .000001  # want it very small, but not 0
            while useSpeed[primaryOrSecondary]/deproj_factor > 2000:
                deproj_factor = np.sin(np.arccos(2*np.random.rand()-1))
            useSpeed[primaryOrSecondary] /= deproj_factor

        
        # Set the injected kick values to the drawn deller speeds
        location.properties['--kick-magnitude-1'], location.properties['--kick-magnitude-2'] = useSpeed




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

