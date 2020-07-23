#!/usr/bin/env python

import os
import pandas as pd
import shutil
import time
import numpy as np
from scipy.stats import multivariate_normal
from modules import *

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
    m1 = classes.Dimension('Mass_1', 5, 150, sampler.kroupa, prior.kroupa)
    q = classes.Dimension('q', 0, 1, sampler.uniform, prior.uniform, should_print = False)
    metallicity_1 = classes.Dimension('Metallicity_1', 0.0001, 0.023, sampler.flat_in_log, prior.flat_in_log)
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
    grid_filename = output_folder + '/grid_' + str(batch_num) + '.csv'
    output_container = 'batch_' + str(batch_num)
    compas_args = [compas_executable, "--grid", grid_filename, '--outputPath', output_folder, '--logfile-delimiter', 'COMMA', '--output-container', output_container, '--pair-instability-supernovae', True, '--alwaysStableCaseBBBCFlag', True, '--common-envelope-slope-Kruckow', -0.833333, '--forceCaseBBBCStabilityFlag', True, '--kick-velocity-sigma-CCSN-BH', 265, '--mass-transfer-fa', 0.5, '--mass-transfer-rejuvenation-prescription', 'STARTRACK', '--pulsar-birth-spin-period-distribution-min', 10, '--pulsational-pair-instability', True, '--pulsational-pair-instability-prescription', 'MARCHANT', '--use-mass-loss', True, '--remnant-mass-prescription', 'FRYER2012', '--random-seed', np.random.randint(2,2**63-1)+(batch_num*10000)]
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
        folder = os.path.join(output_folder, batch['output_container'])
        shutil.move(batch['grid_filename'], folder + '/grid_' + str(batch['number']) + '.csv')
        system_parameters = pd.read_csv(folder + '/BSE_System_Parameters.csv', skiprows = 2)
        system_parameters.rename(columns = lambda x: x.strip(), inplace = True)
        seeds = system_parameters['SEED']
        ST1 = system_parameters['Stellar_Type_1']
        ST2 = system_parameters['Stellar_Type_2']
        for index, sample in enumerate(batch['samples']):
            seed = seeds[index]
            sample.properties['SEED'] = seed
            sample.properties['is_hit'] = 0
            sample.properties['coalescence_time'] = 0
            sample.properties['time'] = 0
            sample.properties['Mass_1@DCO'] = 0
            sample.properties['Mass_2@DCO'] = 0
            sample.properties['Merges_Hubble_Time'] = 0
            sample.properties['batch'] = batch['number']
            sample.properties['Stellar_Type_1'] = ST1[index]
            sample.properties['Stellar_Type_2'] = ST2[index]

             #sample.properties['SE_bias'] =
        double_compact_objects = pd.read_csv(folder + '/BSE_Double_Compact_Objects.csv', skiprows = 2)
        double_compact_objects.rename(columns = lambda x: x.strip(), inplace = True)
        #Generally, this is the line you would want to change.
        dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 14, np.logical_and(double_compact_objects['Stellar_Type_2'] == 14, double_compact_objects['Merges_Hubble_Time'] == True))]
        #dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 14, double_compact_objects['Stellar_Type_2'] == 14)]
        #dns = double_compact_objects[np.logical_and(double_compact_objects['Stellar_Type_1'] == 13, double_compact_objects['Stellar_Type_2'] == 13)]
        DCO_seeds = set(double_compact_objects['SEED'])
        interesting_systems_seeds = set(dns['SEED'])
        for sample in batch['samples']:
            if sample.properties['SEED'] in interesting_systems_seeds:
                sample.properties['is_hit'] = 1
                sample.properties['Merges_Hubble_Time'] = 1
            if sample.properties['SEED'] in DCO_seeds:
                sample.properties['coalescence_time'] = double_compact_objects.loc[double_compact_objects['SEED'] == sample.properties['SEED'], 'Coalescence_Time'].iloc[0]
                sample.properties['time'] = double_compact_objects.loc[double_compact_objects['SEED'] == sample.properties['SEED'], 'Time'].iloc[0]
                sample.properties['Mass_1@DCO'] = double_compact_objects.loc[double_compact_objects['SEED'] == sample.properties['SEED'], 'Mass_1'].iloc[0]
                sample.properties['Mass_2@DCO'] = double_compact_objects.loc[double_compact_objects['SEED'] == sample.properties['SEED'], 'Mass_2'].iloc[0]
        return len(dns)
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

    #if hasattr(sw, 'adapted_distributions'):
    #    ligo_dist = []
    #    rows = []
    #    mean = []
    #    cov = []
    #    #mean = sw.adapted_distributions.mean.items().to_array()
    #    #variance = np.diagflat(sw.adapted_distributions.dimensions.cov)
    #    #Rrej = sw.adapted_distributions.rejection_rate
    #    for distribution in sw.adapted_distributions:
    #        folder = os.path.join(output_folder, 'batch_' + str(int(distribution.mean.properties['batch'])))
    #        dco_file = pd.read_csv(folder + '/BSE_Double_Compact_Objects.csv', skiprows = 2)
    #        dco_file.rename(columns = lambda x: x.strip(), inplace = True)
    #        row = dco_file.loc[dco_file['SEED'] == distribution.mean.properties['SEED']]
    #        rows.append([row.iloc[0]['Mass_1'], row.iloc[0]['Mass_2']])
    #        # We use model B from Abbott et al. 2019
    #        m1 = max(rows[-1])
    #        m2 = min(rows[-1])
    #        q = m2/m1
    #        print(m2)
    #        # the model gives the expected intrinsic merger rate. Use selection effects from Fishbach et al. 2017 to recover observed distribution
    #        # No need to normalize the distribution since we're only interested in the relative values
    #        bias = 0
    #        if m2 < 7.9 or m1 > 42.0:
    #            bias = (42.0**(-1.6 + 2.2) / 10) * q**(0.3)
    #        else:
    #            bias = m1**(-1.6 + 2.2) * q**(0.3)
    #        ligo_dist.append(bias)
    #        mean_per_dist = []
    #        cov_per_dist = []
    #        for dimension in distribution.mean.dimensions:
    #            #print(str(dimension) + '!')
    #            if str(dimension) == 'Mass_1' or str(dimension) == 'q':
    #                #print(distribution.mean.dimensions[dimension])
    #                mean_per_dist.append(distribution.mean.dimensions[dimension])
    #        for dimension in distribution.sigma.dimensions:
    #            if str(dimension) == 'Mass_1' or str(dimension) == 'q':
    #                cov_per_dist.append((distribution.sigma.dimensions[dimension])**2)
    #        mean.append(mean_per_dist)
    #        cov.append(cov_per_dist)

    #    mean = np.asarray(mean)
    #    variance = np.asarray(cov)
    #    #print(mean)
    #    #print(variance)
    #    pdf = np.zeros((len(mean),len(mean)))
    #    for distribution in range(len(mean)):
    #        pdf[distribution, :] = multivariate_normal.pdf(mean, mean[distribution], np.diagflat(variance[distribution]), allow_singular = True)
    #    pdf = np.sum(pdf, axis=0)
    #    sampling_bias = ligo_dist / pdf
    #    # update the weights
    #    print(pdf, sampling_bias)
    #    mean_weights = np.mean(sampling_bias)
    #    print(mean_weights)
    #    for index, distribution in enumerate(sw.adapted_distributions):
    #        distribution.biased_weight = sampling_bias[index] / mean_weights
    #        print('bias: ' + str(sampling_bias[index]))
    #        print('weight: ' + str(distribution.biased_weight))


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
    eccentricity = [location.properties['Eccentricity'] for location in locations]
    #radius_1 = utils.get_zams_radius(mass_1, constants.METALLICITY_SOL)
    #radius_2 = utils.get_zams_radius(mass_2, constants.METALLICITY_SOL)
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
    TOTAL_NUM_SYSTEMS = 10000000 #total number of systems you want in the end
    NUM_CPU_CORES = 25 #Number of cpu cores you want to run in parellel
    NUM_SYSTEMS_PER_RUN = 10000 #Number of systems generated by each of run on each cpu core
    debug = False #If True, will print the logs given by the external program (like COMPAS)
    run_on_helios = True #If True, it will run on a clustered system helios, rather than your pc
    mc_only = False # If you dont want to do the refinement phase and just do random mc exploration
    output_filename = 'samples.csv' #The name of the output file

    compas_executable = os.path.join(os.environ.get('COMPAS_ROOT_DIR'), 'src/COMPAS') # Location of the executable
    output_folder =  '/n/de_mink_lab/Users/lvanson/CompasOutput/StroopwafelTest/FlorisTest/'#'/home/10760369/stroopwafel/bbh_10M' # Folder you want to receieve outputs, here the current working directory, but you can specify anywhere

    if os.path.exists(output_folder):
        command = input ("The output folder already exists. If you continue, I will remove all its content. Press (Y/N)\n")
        if (command == 'Y'):
            shutil.rmtree(output_folder)
        else:
            exit()
    os.makedirs(output_folder)

    # STEP 1 : Create an instance of the Stroopwafel class
    sw = stroopwafel.Stroopwafel(TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_RUN, output_folder, output_filename, debug = debug, run_on_helios = run_on_helios, mc_only = mc_only)


    #STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
    dimensions = create_dimensions()
    sw.initialize(dimensions, interesting_systems, configure_code_run, rejected_systems, update_properties_method = update_properties)

    intial_pdf = distributions.InitialDistribution(dimensions)
    #STEP 4: Run the 4 phases of stroopwafel
    sw.explore(intial_pdf) #Pass in the initial distribution for exploration phase
    sw.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
    ## Do selection effects
    #selection_effects(sw)
    sw.refine() #Stroopwafel will draw samples from the adapted distributions
    sw.postprocess(distributions.Gaussian, only_hits = False) #Run it to create weights, if you want only hits in the output, then make only_hits = True

    end_time = time.time()
    print ("Total running time = %d seconds" %(end_time - start_time))

