import os
from .utils import *
import shutil

# the code is based on the adaptive importance sampling (AIS) algorithm from
# Broekgaarden et al. 2019 (https://arxiv.org/pdf/1905.00910.pdf).

class Stroopwafel:

    # general arguments concerning the simulation that the user has to define in the interface
    def __init__(self, total_num_systems, num_batches_in_parallel, num_samples_per_batch, output_folder, output_filename, debug = False, run_on_helios = True, mc_only = False):
        self.total_num_systems = total_num_systems
        self.num_batches_in_parallel = num_batches_in_parallel
        self.num_samples_per_batch = num_samples_per_batch
        self.output_folder = output_folder
        self.output_filename = os.path.join(self.output_folder, output_filename)
        self.debug = debug # if true, print logs given by external program
        # CHANGE NAME TO "run_on_cluster"?
        self.run_on_helios = run_on_helios # if true, run the simulation on a remote computer cluster
        self.mc_only = mc_only # if true, run a full monte-carlo simulation instead of using adaptive importance sampling

    def update_fraction_explored(self):
        """
        Function which updates the fraction of region which is already explored.
        See eq. 18 from Broekgaarden et al. 2019
        """
        # We try to find the fraction of samples in the exploration phase that corresponds
        # to the smallest uncertainty on the target population rate
        unidentified_region_weight = 1.0 / (self.fraction_explored * self.total_num_systems)  # estimate for target population region that is yet undiscovered
        target_rate = float(self.num_hits) / self.num_explored # estimate for fraction of samples that produce a target system
        numerator = target_rate * (np.sqrt(1. - target_rate) - np.sqrt(unidentified_region_weight))
        denominator = np.sqrt(1. - target_rate) * (np.sqrt(unidentified_region_weight * (1. - target_rate)) + target_rate)
        self.fraction_explored = 1 - numerator / denominator

    def should_continue_exploring(self):
        """
        Function that estimates if we should continue exploring or are ready to switch to the next phase
        OUT:
            bool : boolean value telling If we should continue exploring or not
        """
        if self.mc_only:
            return self.num_explored < self.total_num_systems # we want to always keep exploring when running a monte-carlo only simulation
        return self.num_explored / self.total_num_systems < self.fraction_explored # otherwise, keep track whether we have explored enough to continue to the next phase

    def determine_rate(self, locations):
        """
        Function that determines the rate of producing the hits from the algorithm.
        See eq. 14 from Broekgaarden et al. 2019
        IN:
            hit_locations(list(Location)): All the locations
        OUT:
            (float, float): A pair of values which has the rate of stroopwafel rate and the uncertainty in the rate
        """
        phi = np.zeros(len(locations))
        for index, location in enumerate(locations):
            if location.properties['is_hit'] == 1:
                phi[index] = location.properties['mixture_weight'] # collect the mixture weights of all target systems
        stroopwafel_rate = np.sum(phi) / len(locations) # the target rate is the number of target systems divided by the total number of systems
        uncertainty = np.std(phi, ddof = 1) / np.sqrt(len(locations)) # the uncertainty is estimated by the random Poisson noise, which is proportional to the square-root of the number of samples
        return (np.round(stroopwafel_rate, 6), np.round(uncertainty, 6))

    def calculate_mixture_weights(self, n_dimensional_distribution_type, locations):
        """
        Function that will calculate the mixture weights of all the locations provided. The mixture weights are defined as the ratio between the
        probability of having drawn a sample from the prior distribution and the mixture distribution.
        See eqs. 8,9,12,13 from Broekgaarden et al. 2019
        IN:
            locations (list(Location)) : All the locations for which weight needs to be computed
        """
        # If running only monte-carlo, set all mixture weights to 1, since all samples were drawn from the prior distribution
        [location.properties.update({'mixture_weight': 1}) for location in locations]
        if self.num_explored == self.total_num_systems:
            return
        n_dimensional_distribution_type.calculate_probability_of_locations_from_distribution(locations, self.adapted_distributions) # probability of having drawn each sample from the adapted sampling distribution
        pi_norm = 1.0 / (1 - self.prior_fraction_rejected)
        q_norm = 1.0 / (1 - self.distribution_rejection_rate) # by rejecting certain samples each unrejected system has a slightly higher probabilty of being drawn. we need to correct for this
        fraction_explored = self.num_explored / self.total_num_systems
        for location in locations:
            prior_pdf = location.calculate_prior_probability() * pi_norm # normalised probability of each sample drawn from the prior distribution
            q_pdf = location.properties.pop('q_pdf') / len(self.adapted_distributions) # normalised probability of each sample from the adapted sampling distribution
            Q = (fraction_explored * prior_pdf) + ((1 - fraction_explored) * q_pdf * q_norm) # complete normalised probability of each sample drawn from the combination of prior and adapted sampling distributions (mixture distribution)
            location.properties['mixture_weight'] = prior_pdf / Q

    def process_batches(self, batches, is_exploration_phase):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
            is_exploration_phase (Boolean) : Whether the given batches come from exploration phase or not
        """
        # Process each batch and keep track of the progress of the simulation
        for batch in batches:
            if batch['process']:
                returncode = batch['process'].wait()
            folder = os.path.join(self.output_folder, batch['output_container'])
            shutil.move(batch['grid_filename'], os.path.join(folder, 'grid_' + str(batch['number']) + '.csv')) # move the grid files into their corresponding batch directories
            [location.properties.update({'is_hit': 0}) for location in batch['samples']]
            hits = 0
            if returncode >= 0 and self.interesting_systems_method is not None:
                hits = self.interesting_systems_method(batch) # determine which systems are target systems
            # If too many batches were created, the excess ones have to be removed
            if (is_exploration_phase and not self.should_continue_exploring()) or self.finished >= self.total_num_systems or returncode < 0:
                #This batch is not needed anymore, delete the folder
                shutil.rmtree(os.path.join(self.output_folder, 'batch_' + str(batch['number'])))
                self.batch_num = self.batch_num - 1
                continue
            self.num_hits += hits # update the number of target systems found for each batch
            print_samples(batch['samples'], self.output_filename, 'a')
            self.finished += self.num_samples_per_batch
            printProgressBar(self.finished, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)
            # During the exploration phase, keep track of the number and fraction of explored systems
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
            # During the refinement phase, keep track of how many systems are left to be sampled
            else:
                self.num_to_be_refined -= self.num_samples_per_batch

    def initialize(self, dimensions, interesting_systems_method, configure_code_run, rejected_systems_method, update_properties_method = None):
        """
        This function is the one which is run only once in the stroopwafel class. It initializes the associated variables and the function calls that user will specify
        IN:
            interesting_system_method: The method provided by the user which will determine what an interesting system is for stroopwafel
            update_properties_method: The method provided by the user which will run to update the properties of each of the location once it is sampled
            configure_code_run: The method provided by the user which will be running for each of the batches to determine the command line args for that batch
            rejected_systems_method: The method provided by the user which will calculate how many systems can be rejected given their initial properties
        """
        self.dimensions = dimensions
        self.interesting_systems_method = interesting_systems_method
        self.update_properties_method = update_properties_method
        self.configure_code_run = configure_code_run
        self.rejected_systems_method = rejected_systems_method
        self.batch_num = 0
        self.num_explored = 0
        self.finished = 0
        self.num_hits = 0
        self.fraction_explored = 1
        printProgressBar(0, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def explore(self, intial_pdf):
        """
        This function covers the exploration phase of the stroopwafel algorithm. In this phase, we use monte-carlo sampling from the initial property distributions.
        It consistently calculates whether we should stay exploring or switch to the adaptation phase
        IN:
            initial_pdf (NDimensionalDistribution) : An instance of NDimensionalDistribution showing how to sample from in the exploration phase
        """
        # If we are not running a Monte-Carlo only simulation, we need to keep track of the number
        # of rejected systems, in order to recover the correct mixture weights in the end
        if not self.mc_only:
            self.prior_fraction_rejected = intial_pdf.calculate_rejection_rate(self.update_properties_method, self.rejected_systems_method, self.dimensions)
            print_logs(self.output_folder, "prior_fraction_rejected", self.prior_fraction_rejected)
        else:
            self.prior_fraction_rejected = 0
        while self.should_continue_exploring():
            batches = []
            # For each batch, we sample a number of systems that is written
            # to a grid file, which is read by the external code
            for batch in range(self.num_batches_in_parallel):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                num_samples = int(2 * np.ceil(self.num_samples_per_batch / (1 - self.prior_fraction_rejected))) # draw extra samples to make sure we won't end up with too little systems due to rejection
                (locations, mask) = intial_pdf.run_sampler(num_samples)
                [location.revert_variables_to_original_scales() for location in locations] # to facilitate the calculations, all the property values have to be user-defined in linear scale. here we revert back to these original scales
                if self.update_properties_method != None:
                    self.update_properties_method(locations, self.dimensions) # update the properties of sampled systems
                if self.rejected_systems_method != None:
                    self.rejected_systems_method(locations, self.dimensions) # calculate whether sampled systems should be rejected
                locations[:] = [location for location in locations if location.properties.get('is_rejected', 0) == 0]
                # Since we drew too many samples, shuffle them to randomize and save the number of systems we actually want
                np.random.shuffle(locations)
                locations = locations[:self.num_samples_per_batch]
                [location.properties.pop('is_rejected', None) for location in locations]
                current_batch['samples'] = locations
                command = self.configure_code_run(current_batch) # pass additional command line arguments
                generate_grid(locations, current_batch['grid_filename']) # create a grid file with the initial sample properties
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios) # pass the grid to the external code to evolve the systems
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            # After completing a set of batches, calculate the progress of the phase/simulation
            self.process_batches(batches, True)
        if not self.mc_only:
            print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(self.num_hits, self.num_explored, self.num_hits / self.num_explored, self.fraction_explored))
            print_logs(self.output_folder, "num_explored", self.num_explored)
            print_logs(self.output_folder, "fraction_explored", self.fraction_explored)

    def adapt(self, n_dimensional_distribution_type):
        """
        Adaptive phase of stroopwafel. Here, the locations of the target systems are used to create a new n-dimensional sampling distribution
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        # If we spend all samples on exploring, there is no room left for refining
        if self.num_explored != self.total_num_systems:
            # The adapted distributions are created from the target systems, so we need at least one hit
            if self.num_hits > 0:
                hits = read_samples(self.output_filename, self.dimensions, only_hits = True)
                [location.transform_variables_to_new_scales() for location in hits]
                average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / len(self.dimensions)) # calculates the average distance between each sample in 1-D parameter space
                self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(hits, average_density_one_dim) # create the adapted distributions
                print_distributions(os.path.join(self.output_folder, 'distributions.csv'), self.adapted_distributions)
                self.distribution_rejection_rate = n_dimensional_distribution_type.calculate_rejection_rate(self.adapted_distributions, self.update_properties_method, self.rejected_systems_method, self.dimensions) # recalculate how many samples will be rejected from the new sampling distributions
                print_logs(self.output_folder, "distribution_rejection_rate", self.distribution_rejection_rate)
            print ("Adaptation phase finished!")

    def refine(self):
        """
        Refinement phase of stroopwafel. During this phase the remaining samples are drawn from the updated sampling distributions
        """
        self.num_to_be_refined = self.total_num_systems - self.num_explored
        while self.num_to_be_refined > 0:
            batches = []
            # In each batch we will draw samples, which are passed to the external code,
            # similar to the exploration phase. The difference is that these n_systems
            # are drawn equally from the set of adapted distributions.
            for batch in range(min(self.num_batches_in_parallel, int(self.num_to_be_refined / self.num_samples_per_batch))): # if the refinement phase needs to sample less batches than we run in parallel, make sure we don't run too many
                current_batch = dict()
                current_batch['number'] = self.batch_num
                locations_ref = []
                num_samples = np.ceil(self.num_samples_per_batch / len(self.adapted_distributions)) # divide the samples equally between each adapted distribution
                # Repeat the process of sampling and evolving as seen in the exploration phase
                for distribution in self.adapted_distributions:
                    (locations, mask) = distribution.run_sampler(num_samples , self.dimensions, True)
                    locations_ref.extend(np.asarray(locations)[mask])
                [location.revert_variables_to_original_scales() for location in locations_ref]
                if self.update_properties_method != None:
                    self.update_properties_method(locations_ref, self.dimensions)
                if self.rejected_systems_method != None:
                    self.rejected_systems_method(locations_ref, self.dimensions)
                locations_ref[:] = [location for location in locations_ref if location.properties.get('is_rejected', 0) == 0]
                np.random.shuffle(locations_ref)
                locations_ref = locations_ref[:self.num_samples_per_batch]
                [location.properties.pop('is_rejected', None) for location in locations_ref]
                current_batch['samples'] = locations_ref
                command = self.configure_code_run(current_batch)
                generate_grid(locations_ref, current_batch['grid_filename'])
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.process_batches(batches, False)
        if self.num_explored != self.total_num_systems:
            num_refined = self.total_num_systems - self.num_explored
            print_logs(self.output_folder, "total_num_systems", self.num_explored + num_refined)
            print ("\nRefinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(self.num_hits - len(self.adapted_distributions), num_refined, (self.num_hits - len(self.adapted_distributions)) / num_refined))

    def postprocess(self, n_dimensional_distribution_type, only_hits = False):
        """
        Postprocessing phase of stroopwafel. Here, the mixture weights of all samples are calculated
        IN:
            only_hits (Boolean) : If you want to print only the hits
        """
        locations = read_samples(self.output_filename, self.dimensions, only_hits) # read in all stored samples
        [location.transform_variables_to_new_scales() for location in locations]
        self.calculate_mixture_weights(n_dimensional_distribution_type, locations) # calculate the mixture weights and add them to the locations
        [location.revert_variables_to_original_scales() for location in locations]
        print_samples(locations, self.output_filename, 'w') # print the updated locations to the output file
        (stroopwafel_rate, uncertainty) = self.determine_rate(locations)
        print ("Rate of hits = %f with uncertainty = %f" %(stroopwafel_rate, uncertainty))
        print_logs(self.output_folder, "rate_of_hits", stroopwafel_rate)
        print_logs(self.output_folder, "uncertainty", uncertainty)
