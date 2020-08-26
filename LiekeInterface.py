#!/usr/bin/env python

import os
import pandas as pd
import shutil
import time
import numpy as np
from stroopwafel import *
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--num_systems', help = 'Total number of systems', type = int, default = 100)#100000)
parser.add_argument('--num_cores', help = 'Number of cores to run in parallel', type = int, default = 10)#50)
parser.add_argument('--num_per_core', help = 'Number of systems to generate in one core', type = int, default = 10)#1000)
parser.add_argument('--debug', help = 'If debug of COMPAS is to be printed', type = bool, default = True)
parser.add_argument('--mc_only', help = 'If run in MC simulation mode only', type = bool, default = False)
parser.add_argument('--run_on_helios', help = 'If we are running on helios (or other slurm) nodes', type = bool, default = False)
parser.add_argument('--output_filename', help = 'Output filename', default = 'samples.csv')
parser.add_argument('--output_folder', help = 'output folder', default = './output')
parser.add_argument('--compas_arg_flags', '--cf', nargs='*', help='flag names of aditional compas arguments', required=False, default = None)
parser.add_argument('--compas_arg_vals', '--cv', nargs='*', help='aditional compas arguments', required=False, type=float, default = None)
namespace, extra_params = parser.parse_known_args()
if not np.logical_or(np.logical_and(not namespace.compas_arg_flags, not namespace.compas_arg_vals), np.logical_and(namespace.compas_arg_flags != None, namespace.compas_arg_vals != None) ):
    parser.error("Both --compas_arg_flags and --compas_arg_vals have to be defined simultaniously!")

# STEP 2 : Define the functions
def create_dimensions():
    """
    This Function that will create all the dimensions for stroopwafel, a dimension is basically one of the variables you want to sample
    Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in classes.py for more.
    It takes the name of the dimension, its max and min value. 
    The Sampler class will tell how to sample this dimension. Similarly, prior tells it how it calculates the prior. You can find more of these in their respective modules
    OUT:
        As Output, this should return a list containing all the instances of Dimension class.
    """
    m1 = classes.Dimension('Mass_1', 8, 150, sampler.kroupa, prior.kroupa)
    q = classes.Dimension('q', 0.01, 1, sampler.uniform, prior.uniform, should_print = False)
    metallicity_1 = classes.Dimension('Metallicity_1', 1e-4, 0.023, sampler.flat_in_log, prior.flat_in_log)
    a = classes.Dimension('Separation', 0.01, 1000, sampler.flat_in_log, prior.flat_in_log)
    #kick_velocity_random_1 = classes.Dimension('Kick_Velocity_Random_1', 0, 1, sampler.uniform, prior.uniform)
    #kick_theta_1 = classes.Dimension('Kick_Theta_1', -np.pi / 2, np.pi / 2, sampler.uniform_in_sine, prior.uniform_in_sine)
    #kick_phi_1 = classes.Dimension('Kick_Phi_1', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    #kick_velocity_random_2 = classes.Dimension('Kick_Velocity_Random_2', 0, 1, sampler.uniform, prior.uniform)
    #kick_theta_2 = classes.Dimension('Kick_Theta_2', -np.pi / 2, np.pi / 2, sampler.uniform_in_sine, prior.uniform_in_sine)
    #kick_phi_2 = classes.Dimension('Kick_Phi_2', 0, 2 * np.pi, sampler.uniform, prior.uniform)
    #return [m1, q, a, kick_velocity_random_1, kick_theta_1, kick_phi_1, kick_velocity_random_2, kick_theta_2, kick_phi_2]
    return [m1, q, metallicity_1, a]


def update_properties(locations, dimensions):
    """
    This function is not mandatory, it is required only if you have some dependent variable. 
    For example, if you want to sample Mass_1 and q, then Mass_2 is a dependent variable which is product of the two.
    Similarly, you can assume that Metallicity_2 will always be equal to Metallicity_1
    IN:
        locations (list(Location)) : A list containing objects of Location class in classes.py. 
        You can play with them and update whatever fields you like or add more in the property (which is a dictionary)
    OUT: Not Required
    """
    m1 = dimensions[0]
    q = dimensions[1]
    metallicity_1 = dimensions[2]
    for location in locations:
        location.properties['Mass_2'] = location.dimensions[m1] * location.dimensions[q]
        location.properties['Metallicity_2'] = location.dimensions[metallicity_1]
        # location.properties['Metallicity_2'] = location.properties['Metallicity_1'] = constants.METALLICITY_SOL
        location.properties['Eccentricity'] = 0
        #location.properties['Kick_Mean_Anomaly_1'] = np.random.uniform(0, 2 * np.pi, 1)[0]
        #location.properties['Kick_Mean_Anomaly_2'] = np.random.uniform(0, 2 * np.pi, 1)[0]

def get_default_COMPAS_commands():
    """
    This function loads the default values from $COMPAS_ROOT_DIR/defaults/pythonSubmitDefault.py
    IN:
        
    OUT: default_commandlineOptions (list(String)) : Will be added to compas_args. 
    """
    #Add the COMPAS root dir to the import envir
    import sys
    compas_root = os.environ["COMPAS_ROOT_DIR"]
    sys.path.append(compas_root+'/defaults/')
    #Import the pythonSubmitDefault file (can be adjusted to any pythonSubmit.py file)
    import pythonSubmitDefault as psd

    #-- Get the program options
    programOptions =psd.pythonProgramOptions()
    booleanChoices = programOptions.booleanChoices()
    booleanCommands = programOptions.booleanCommands()

    numericalChoices = programOptions.numericalChoices()
    numericalCommands = programOptions.numericalCommands()

    stringChoices = programOptions.stringChoices()
    stringCommands = programOptions.stringCommands()

    listChoices = programOptions.listChoices()
    listCommands = programOptions.listCommands()

    execute = ''

    # create a list of default command line options
    default_commandlineOptions = psd.generateCommandLineOptions(execute,booleanChoices,booleanCommands,numericalChoices,numericalCommands,stringChoices,stringCommands,listChoices,listCommands)
    default_commandlineOptions = default_commandlineOptions.split()

    return default_commandlineOptions

def get_extra_arg():
    """
    Reshape the compas_arg_flags and compas_arg_vals from the ArgumentParser to a list that can be added to the final compas_args
    IN: 
    OUT: extra_arg (list(String)) : Will be added to compas_args.  
    """
    extra_arg = []
    if np.logical_and(namespace.compas_arg_flags != None, namespace.compas_arg_vals != None):
        try: 
            additional_args = namespace.compas_arg_flags[0].split()
        except:
            additional_args = namespace.compas_arg_flags.split()        
        additional_values = namespace.compas_arg_vals
        if len(additional_args) != len(additional_values):
            print("currently len(compas_arg_flags)=%s and len(compas_arg_vals)=%s" % (len(additional_args), len(additional_values)))
            raise ValueError("you have to provide an equal number of compas_arg_flags and compas_arg_vals!")
        extra_arg = list(np.array([(additional_args[i], additional_values[i]) for i in range(len(additional_args)) ]).flat)
    return extra_arg

def check_duplicates(default_args, input_arg):
    """
    Create an adjusted default list that does not contain the flags in compas_arg_flags (to avoid duplicity)
    IN:  default_args = default_commandlineOptions from COMPAS, input_arg  = sw + ArgumentParser input flags
    OUT:  adjusted_default (list(String))
    """
    adjusted_default = default_args.copy()
    remove_from_default = ["--number-of-binaries"] 
    for arg in input_arg + remove_from_default:
        #print('arg', arg)
        try:
            if np.logical_and(arg.startswith('--'), arg in adjusted_default):
                problem_i = np.where(np.array(adjusted_default) == arg)[0][0]
                if not adjusted_default[problem_i+1].startswith('--'):
                    #print("Remove both the flag and the option", "[problem_i]", [problem_i], "[[problem_i+1]", [problem_i+1], adjusted_default[problem_i], adjusted_default[problem_i+1])
                    #Remove both the flag and the option
                    del adjusted_default[problem_i+1]#Always delete last one first!
                    del adjusted_default[problem_i]
                else:
                    #only remove the flag (e.g. for boolean)
                    #print("only remove the flag (e.g. for boolean)", "[problem_i]", [problem_i], "[[problem_i+1]", [problem_i+1], adjusted_default[problem_i])
                    del adjusted_default[problem_i]
        except:
            #print('this is not a flag: %s, continue' % arg)
            continue
    return adjusted_default




def configure_code_run(batch):
    """
    This function tells stroopwafel what program to run, along with its arguments.
    IN:
        batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
            It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
            for each batch run in this dictionary. For example, here I have stored the 'output_container' and 'grid_filename' so that I can read them during discovery of interesting systems below
    OUT:
        compas_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
        Additionally one must also store the grid_filename in the batch so that the grid file is created
    """
    batch_num = batch['number']
    randomSeedFileName = 'RandomSeed.txt'
    if os.path.isfile(output_folder + randomSeedFileName):
        #print('found randomseed file', output_folder + randomSeedFileName )
        seeds = np.loadtxt(output_folder + randomSeedFileName)
        current_seed = int(seeds[batch_num])
    else:
        current_seed = np.random.randint(2, 2**63 - 1)
    #print('current_seed', current_seed)

    grid_filename = output_folder + '/grid_' + str(batch_num) + '.csv'
    output_container = 'batch_' + str(batch_num)
    #get and combine commandline arguments
    extra_arg           = get_extra_arg()
    sw_args             = [compas_executable, "--grid", grid_filename, '--outputPath', output_folder, '--output-container', output_container, '--random-seed', current_seed]
    input_arg           = sw_args + extra_arg

    default_args        = get_default_COMPAS_commands()
    #Check for duplicate arguments between default and user input
    adjusted_default    = check_duplicates(default_args, input_arg)

    compas_args         = input_arg + adjusted_default
    for params in extra_params:
        compas_args.extend(params.split("="))
    #print('Run_details', compas_args)    
    batch['grid_filename'] = grid_filename
    batch['output_container'] = output_container
    return compas_args


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
        #What will be written as output to the .csv file wiill be determined here
        folder = os.path.join(output_folder, batch['output_container'])
        shutil.move(batch['grid_filename'], folder + '/grid_' + str(batch['number']) + '.csv')
        system_parameters = pd.read_csv(folder + '/BSE_System_Parameters.csv', skiprows = 2)
        system_parameters.rename(columns = lambda x: x.strip(), inplace = True)
        seeds = system_parameters['SEED']
        for index, sample in enumerate(batch['samples']):
            seed = seeds[index]
            sample.properties['SEED'] = seed
            sample.properties['is_hit'] = 0
            sample.properties['batch'] = batch['number']
        double_compact_objects = pd.read_csv(folder + '/BSE_Double_Compact_Objects.csv', skiprows = 2)
        double_compact_objects.rename(columns = lambda x: x.strip(), inplace = True)
        #Generally, this is the line you would want to change.
        dbh = double_compact_objects[np.logical_and(double_compact_objects['Merges_Hubble_Time'] == 1, \
            np.logical_and(double_compact_objects['Stellar_Type_1'] == 14, double_compact_objects['Stellar_Type_2'] == 14))]
        interesting_systems_seeds = set(dbh['SEED'])
        for sample in batch['samples']:
            if sample.properties['SEED'] in interesting_systems_seeds:
                sample.properties['is_hit'] = 1
        return len(dbh)
    except IOError as error:
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
            dco_file = pd.read_csv(folder + '/BSE_Double_Compact_Objects.csv', skiprows = 2)
            dco_file.rename(columns = lambda x: x.strip(), inplace = True)
            row = dco_file.loc[dco_file['SEED'] == distribution.mean.properties['SEED']]
            rows.append([row.iloc[0]['Mass_1'], row.iloc[0]['Mass_2']])
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
    metallicity_1 = dimensions[2]
    a = dimensions[3]
    [location.properties.update({'Mass_2': location.dimensions[m1] * location.dimensions[q]}) for location in locations]
    mass_1 = [location.dimensions[m1] for location in locations]
    mass_2 = [location.properties['Mass_2'] for location in locations]
    Z_1 = [location.dimensions[metallicity_1] for location in locations]
    Z_2 = Z_1
    # metallicity_1 = [location.properties['Metallicity_1'] for location in locations]
    # metallicity_2 = [location.properties['Metallicity_2'] for location in locations]
    eccentricity = [location.properties['Eccentricity'] for location in locations]
    num_rejected = 0
    for index, location in enumerate(locations):
        radius_1 = utils.get_zams_radius(mass_1[index], Z_1[index])
        radius_2 = utils.get_zams_radius(mass_2[index], Z_2[index])
        roche_lobe_tracker_1 = radius_1 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_1[index], mass_2[index]))
        roche_lobe_tracker_2 = radius_2 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_2[index], mass_1[index]))
        location.properties['is_rejected'] = 0
        if (mass_2[index] < constants.MINIMUM_SECONDARY_MASS) or (location.dimensions[a] <= (radius_1 + radius_2)) \
        or roche_lobe_tracker_1 > 1 or roche_lobe_tracker_2 > 1:
            location.properties['is_rejected'] = 1
            num_rejected += 1
    return num_rejected

if __name__ == '__main__':
    start_time = time.time()
    #Define the parameters to the constructor of stroopwafel
    TOTAL_NUM_SYSTEMS = namespace.num_systems #total number of systems you want in the end
    NUM_CPU_CORES = namespace.num_cores #Number of cpu cores you want to run in parellel
    NUM_SYSTEMS_PER_RUN = namespace.num_per_core #Number of systems generated by each of run on each cpu core
    debug = namespace.debug #If True, will print the logs given by the external program (like COMPAS)
    run_on_helios = namespace.run_on_helios #If True, it will run on a clustered system helios, rather than your pc
    mc_only = namespace.mc_only # If you dont want to do the refinement phase and just do random mc exploration
    output_filename = namespace.output_filename #The name of the output file
    compas_executable = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/COMPAS') # Location of the executable
    output_folder =  namespace.output_folder  #os.path.join(os.getcwd(), 'output') # Folder you want to receieve outputs, here the current working directory, but you can specify anywhere

    # Make the output directory
    # You already made the folder if you are running a grid of values
    Gridcall = True 
    if not os.path.exists(output_folder):
    	os.makedirs(output_folder)
    else:
        if not Gridcall:
            shutil.rmtree(output_folder)


    print('TOTAL_NUM_SYSTEMS', TOTAL_NUM_SYSTEMS, 'NUM_CPU_CORES', NUM_CPU_CORES, 'NUM_SYSTEMS_PER_RUN', NUM_SYSTEMS_PER_RUN, 'run_on_helios', run_on_helios)
    # STEP 1 : Create an instance of the Stroopwafel class
    sw = stroopwafel.Stroopwafel(TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_RUN, output_folder, output_filename, debug = debug, run_on_helios = run_on_helios, mc_only = mc_only)

    #STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
    dimensions = create_dimensions()
    sw.initialize(dimensions, interesting_systems, configure_code_run, rejected_systems, update_properties_method = update_properties)

    intial_pdf = distributions.InitialDistribution(dimensions)
    #STEP 4: Run the 4 phases of stroopwafel
    sw.explore(intial_pdf) #Pass in the initial distribution for exploration phase
    now = time.time()

    sw.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
    ## Do selection effects
    #selection_effects(sw)
    sw.refine() #Stroopwafel will draw samples from the adapted distributions
    sw.postprocess(distributions.Gaussian, only_hits = False) #Run it to create weights, if you want only hits in the output, then make only_hits = True

    end_time = time.time()
    print ("Total running time = %d seconds" %(end_time - start_time))
