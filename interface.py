#!/usr/bin/env python
# coding: utf-8

from stroopwafel import *
import os

#Define the parameters to the constructor of stroopwafel
NUM_DIMENSIONS = 4 #Number of dimensions you want to samples
NUM_BINARIES = 300 #total number of systems
NUM_BATCHES = 2 #Number of batches you want to run in parellel
NUM_SAMPLES_PER_BATCH = 20 #Number of samples generated by each of the batch
debug = False #If True, will generate the logs given by the external program (like COMPAS)
compas_executable = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/COMPAS') # Location of the executable
output_folder = os.getcwd() # Where you want to receieve outputs

# STEP 1 : Create an instance of the Stroopwafel class
sw = Stroopwafel(NUM_DIMENSIONS, NUM_BINARIES, NUM_BATCHES, NUM_SAMPLES_PER_BATCH, debug = debug)


# STEP 2 : Define the functions
def create_dimensions():
    """
    This Function that will create all the dimensions for stroopwafel, a dimension is basically one of the variables you want to sample
    Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in stroopwafel.py for more.
    It takes the name of the dimension, its max and min value. 
    The Sampler class (also in stroopwafel) will tell how to sample this dimension. Similarly, Prior tells it how it calculates the Prior
    OUT:
        As Output, this should return a list containing all the instances of Dimension class.
    """
    m1 = Dimension('Mass_1', 5, 150, Sampler.kroupa, Prior.kroupa)
    q = Dimension('q', 0, 1, Sampler.uniform, Prior.uniform, should_print = False)
    metallicity_1 = Dimension('Metallicity_1', 0, 1, Sampler.uniform, Prior.uniform)
    a = Dimension('Separation', -1, 3, Sampler.flat_in_log, Prior.uniform)
    return [m1, q, metallicity_1, a]

def update_properties(locations):
    """
    This function is not mandatory, it is required only if you have some dependent variable. 
    For example, if you want to sample Mass_1 and q, then Mass_2 is a dependent variable which is product of the two.
    Similarly, you can assume that Metallicity_2 will always be equal to Metallicity_1
    IN:
        locations (list(Location)) : A list containing objects of Location class in stroopwafel.py. 
        You can play with them and update whatever fields you like or add more in the property (which is a dictionary) of Location
    OUT: Not Required
    """
    m1 = dimensions[0]
    q = dimensions[1]
    metallicity_1 = dimensions[2]
    for location in locations:
        location.properties['Mass_2'] = location.dimensions[m1] * location.dimensions[q]
        location.properties['Metallicity_2'] = location.dimensions[metallicity_1]
        location.properties['Eccentricity'] = 0

def configure_code_run(batch):
    """
    This function tells stroopwafel what program to run, along with its arguments.
    IN:
        batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
            It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
            for each batch run in this dictionary. For example, here I have stored the 'system_params_filename' so that I can read them during discovery of interesting systems below
    OUT:
        compas_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
        Additionally one must also store the grid_filename in the batch so that the grid file is created
    """
    batch_num = batch['number']
    grid_filename = 'grid_' + str(batch_num) + '.txt'
    system_params_filename = 'system_params_' + str(batch_num)
    dco_filename = 'dco_' + str(batch_num)
    compas_args = [compas_executable, "--grid", grid_filename, "--logfile-BSE-system-parameters", system_params_filename, '--logfile-BSE-double-compact-objects', dco_filename, '--output', output_folder, '--logfile-delimiter', 'COMMA']
    batch['system_params_filename'] = system_params_filename
    batch['dco_filename'] = dco_filename
    batch['grid_filename'] = grid_filename
    return compas_args

def interesting_systems(batch):
    """
    This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
    IN:
        batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
    OUT:
        list(Location): A list of Location objects which a user defines as interesting.
        In the below example, I define all the DCOs as interesting, so I read the files, get the parameters from the system_params file and create 
        Location object for each of them with the dimensions and the properties.
    """
    try:
        double_compact_objects = np.genfromtxt(batch['dco_filename'] + '.csv', delimiter = ',', names=True, skip_header = 2)
        system_parameters = np.genfromtxt(batch['system_params_filename'] + '.csv', delimiter = ',', names = True, skip_header = 2)
        system_parameters.dtype.names = [sub.replace('ZAMS', '') if system_parameters.dtype.names.count(sub.replace('ZAMS', '')) == 0 else sub for sub in system_parameters.dtype.names]
        systems = system_parameters[np.isin(system_parameters['ID'], double_compact_objects['ID'])]
        locations = []
        for system in systems:
            location = dict()
            for dimension in dimensions:
                if dimension.name == 'q':
                    location[dimension] = system['Mass_2'] / system['Mass_1']
                else:
                    location[dimension] = system[dimension.name]
            properties = dict()
            for prop in ('ID', 'Metallicity_2', 'Mass_2', 'Eccentricity'):
                properties[prop] = system[prop]
            locations.append(Location(location, properties))
        return locations
    except IOError as error:
        return []


#STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
sw.initialize(interesting_systems, configure_code_run, update_properties_method = update_properties)
dimensions = create_dimensions()
intial_pdf = InitialDistribution(dimensions)

#STEP 4: Run the 4 phases of stroopwafel
sw.explore(dimensions, intial_pdf) #Pass in the dimensions list created, and the initial distribution for exploration phase
sw.adapt(n_dimensional_distribution_type = Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
sw.refine() #Stroopwafel will draw samples from the adapted distributions
sw.postprocess("hits.csv") #Run it to create weights of the hits found. Pass in a filename to store all the hits
