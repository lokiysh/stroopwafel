#!/usr/bin/env python
# coding: utf-8

from stroopwafel import *
import subprocess
import os

NUM_BATCHES = 2
NUM_SAMPLES_PER_BATCH = 10
NUM_BINARIES = 100
NUM_DIMENSIONS = 4

git_directory = os.environ.get('COMPAS_ROOT_DIR')
compas_executable = os.path.join(git_directory, 'src/COMPAS')
output_folder = os.getcwd()
finished = 0
printProgressBar(finished, NUM_BINARIES, prefix = 'progress', suffix = 'complete', length = 20)

def create_dimensions():
    """
    Function that will create all the dimensions for stroopwafel
    """
    m1 = Dimension('Mass_1', 5, 150, Samplers.kroupa, SigmaCalculationMethod.kroupa, Priors.kroupa, should_refine = True, should_print = True)
    q = Dimension('q', 0, 1, Samplers.uniform, SigmaCalculationMethod.usual, Priors.usual, should_refine = True, should_print = False)
    metallicity_1 = Dimension('Metallicity_1', 0, 1, Samplers.uniform, SigmaCalculationMethod.usual, Priors.usual, should_refine = True, should_print = True)
    a = Dimension('Separation', 1, 20, Samplers.power_law, SigmaCalculationMethod.usual, Priors.usual, should_refine = True, should_print = True)
    e = Dimension('Eccentricity', sampler = Samplers.zero, should_refine = False, should_print = True)
    m2 = Dimension('Mass_2', 5, 150, should_refine = False, should_print = True)
    metallicity_2 = Dimension('Metallicity_2', 0, 1, should_refine = False, should_print = True)
    return [m1, q, metallicity_1, a, e, m2, metallicity_2]

def update_derived_dimensions(locations, dimensions):
    """
    Function that will update the derived variables if required
    """
    m2 = dimensions[5]
    m1 = dimensions[0]
    q = dimensions[1]
    metallicity_1 = dimensions[2]
    metallicity_2 = dimensions[6]
    for location in locations:
        location.value[m2] = location.value[m1] * location.value[q]
        location.value[metallicity_2] = location.value[metallicity_1]

def populate_interesting_systems(batch):
    """
    This function should tell stroopwafel what an interesting system is.
    This has to be defined and changed by the user according to requirements
    """
    try:
        double_compact_objects = np.genfromtxt(batch['dco_filename'] + '.csv', delimiter = ',', names=True, skip_header = 2)
        system_parameters = np.genfromtxt(batch['system_params_filename'] + '.csv', delimiter = ',', names = True, skip_header = 2)
        system_parameters.dtype.names = [sub.replace('ZAMS', '') if system_parameters.dtype.names.count(sub.replace('ZAMS', '')) == 0 else sub for sub in system_parameters.dtype.names]
        interesting_systems = system_parameters[np.isin(system_parameters['ID'], double_compact_objects['ID'])]
        locations = []
        for system in interesting_systems:
            location = dict()
            for dimension in dimensions:
                if dimension.name == 'q':
                    location[dimension] = system['Mass_2'] / system['Mass_1']
                else:
                    location[dimension] = system[dimension.name]
            locations.append(Location(location, compas_id = system['ID']))
        return locations
    except IOError as error:
        return []

def run_compas(locations, batch_num):
    grid_filename = 'grid_' + str(batch_num) + '.txt'
    system_params_filename = 'system_params_' + str(batch_num)
    dco_filename = 'dco_' + str(batch_num)
    generate_grid(locations, grid_filename)
    compas_args = [compas_executable, "--grid", grid_filename, "--logfile-BSE-system-parameters", system_params_filename, '--logfile-BSE-double-compact-objects', dco_filename, '--output', output_folder, '--logfile-delimiter', 'COMMA']
    process = subprocess.Popen(" ".join(compas_args), stdout = subprocess.PIPE, shell = True, stderr = subprocess.PIPE)
    return {'process' : process, 'system_params_filename' : system_params_filename, 'dco_filename' : dco_filename}

def wait_for_completion(batches, hits):
    for batch in batches:
        batch['process'].wait()
        hits.extend(populate_interesting_systems(batch))
        global finished
        finished += NUM_SAMPLES_PER_BATCH
        printProgressBar(finished, NUM_BINARIES, prefix = 'progress', suffix = 'complete', length = 20)


dimensions = create_dimensions()
intial_pdf = InitialDistribution(dimensions)

hits = []
batch_num = 0
fraction_exploration_phase = 1.0
should_continue_exploration = True
num_explored = 0

#explore
while (should_continue_exploring(num_explored, NUM_BINARIES, fraction_exploration_phase) == True):
    batches = []
    for batch in range(NUM_BATCHES):
        (locations, mask) = intial_pdf.run_sampler(NUM_SAMPLES_PER_BATCH)
        update_derived_dimensions(locations, dimensions)
        batches.append(run_compas(locations, batch_num))
        batch_num = batch_num + 1
    wait_for_completion(batches, hits)
    num_explored += NUM_BATCHES * NUM_SAMPLES_PER_BATCH
    fraction_exploration_phase = update_fraction_explored(fraction_exploration_phase, NUM_BINARIES, num_explored, len(hits))


print ("Exploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(len(hits), num_explored, len(hits) / num_explored, fraction_exploration_phase))
print ()
#refine
num_to_be_refined = NUM_BINARIES - num_explored
gaussians = []
if num_to_be_refined > 0:
    average_density_one_dim = 1.0 / np.power(num_explored, 1.0 / NUM_DIMENSIONS)
    gaussians = draw_gaussians(hits, average_density_one_dim)
    while num_to_be_refined > 0:
        batches = []
        for batch in range(NUM_BATCHES):
            locations_ref = []
            for gaussian in gaussians:
                (locations, mask) = gaussian.run_sampler(NUM_SAMPLES_PER_BATCH, True)
                locations_ref.extend(np.asarray(locations)[mask])
            np.random.shuffle(locations_ref)
            locations_ref = locations_ref[0:NUM_SAMPLES_PER_BATCH]
            update_derived_dimensions(locations_ref, dimensions)
            batches.append(run_compas(locations_ref, batch_num))
            batch_num = batch_num + 1
        wait_for_completion(batches, hits)
        num_to_be_refined -= NUM_BATCHES * NUM_SAMPLES_PER_BATCH
    print ("Refinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(len(hits) - len(gaussians), (NUM_BINARIES - num_explored), (len(hits) - len(gaussians)) / (NUM_BINARIES - num_explored)))
    print ()
    #postprocess
    calculate_weights_of_gaussians(hits[len(gaussians):], gaussians, NUM_DIMENSIONS, fraction_exploration_phase)
    print(determine_rate(hits, NUM_BINARIES))
    print_hits(hits)
    print ("Weights file has been created!!")