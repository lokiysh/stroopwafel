import os
from utils import *
import shutil
from scipy.stats import multivariate_normal, entropy

class Pmc:

    def __init__(self, total_num_systems, num_batches_in_parallel, num_samples_per_batch, output_folder, output_filename, debug = False, run_on_helios = True, mc_only = False):
        self.total_num_systems = total_num_systems
        self.num_batches_in_parallel = num_batches_in_parallel
        self.num_samples_per_batch = num_samples_per_batch
        self.output_folder = output_folder
        self.output_filename = os.path.join(self.output_folder, output_filename)
        self.exploratory_filename = os.path.join(self.output_folder, 'exploratory_samples.csv')
        self.debug = debug
        self.run_on_helios = run_on_helios
        self.mc_only = mc_only

    def update_fraction_explored(self):
        """
        Function which updates the fraction of region which is already explored
        """
        unidentified_region_weight = 1.0 / (self.fraction_explored * self.total_num_systems)
        target_rate = float(self.num_hits) / self.num_explored
        numerator = target_rate * (np.sqrt(1. - target_rate) - np.sqrt(unidentified_region_weight))
        denominator = np.sqrt(1. - target_rate) * (np.sqrt(unidentified_region_weight * (1. - target_rate)) + target_rate)
        self.fraction_explored = 1 - numerator / denominator

    def should_continue_exploring(self):
        """
        Function that estimates if we should continue exploring or are we ready
        OUT:
            bool : boolean value telling If we should continue exploring or not
        """
        if self.mc_only:
            return self.num_explored < self.total_num_systems
        return self.num_explored / self.total_num_systems < self.fraction_explored

    def initialize(self, dimensions, interesting_systems_method, configure_code_run, rejected_systems_method, update_properties_method = None):
        """
        This function is the one which is run only once in the stroopwafel class. It initializes the associated variables and the function calls that user will specify
        IN:
            interesting_system_method: The method provided by the user which will determine what an interesting system is for stroopwafel
            update_properties_method: The method provided by the user which will run to update the properties of each of the location once it is sampled
            configure_code_run: The method provided by the user which will be running for each of the batches to determine the command line args for that batch
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
        This function is the exploration phase of the stroopwafel
        IN:
            initial_pdf (NDimensionalDistribution) : An instance of NDimensionalDistribution showing how to sample from in the exploration phase
        """
        self.prior_fraction_rejected = intial_pdf.calculate_rejection_rate(self.num_batches_in_parallel, self.output_folder, self.debug, self.run_on_helios)
        print_logs(self.output_folder, "prior_fraction_rejected", self.prior_fraction_rejected)
        while self.should_continue_exploring():
            batches = []
            for batch in range(self.num_batches_in_parallel):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                num_samples = int(2 * np.ceil(self.num_samples_per_batch / (1 - self.prior_fraction_rejected)))
                (locations, mask) = intial_pdf.run_sampler(num_samples)
                [location.revert_variables_to_original_scales() for location in locations]
                if self.update_properties_method != None:
                    self.update_properties_method(locations, self.dimensions)
                self.rejected_systems_method(locations, self.dimensions)
                locations[:] = [location for location in locations if location.properties['is_rejected'] == 0]
                np.random.shuffle(locations)
                locations = locations[:self.num_samples_per_batch]
                [location.properties.pop('is_rejected') for location in locations]
                current_batch['samples'] = locations
                command = self.configure_code_run(current_batch)
                generate_grid(locations, current_batch['grid_filename'])
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.process_batches(batches, True)
        print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(self.num_hits, self.num_explored, self.num_hits / self.num_explored, self.fraction_explored))
        print_logs(self.output_folder, "num_explored", self.num_explored)
        if self.mc_only:
            exit()

    def adapt(self, n_dimensional_distribution_type):
        """
        Adaptive phase of stroopwafel
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        if self.num_hits == 0:
            print ("No hits in the exploration phase\n")
            exit()
        hits = read_samples(self.exploratory_filename, self.dimensions, only_hits = True)
        [location.transform_variables_to_new_scales() for location in hits]
        average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / len(self.dimensions))
        self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(hits, average_density_one_dim, kappa = 5)
        print_distributions(self.output_folder + '/distributions.csv', self.adapted_distributions)
        self.alpha = np.ones(len(self.adapted_distributions)) / len(self.adapted_distributions)
        print ("Adaptation phase finished!")

    def refine(self, n_dimensional_distribution_type):
        """
        Refinement phase of stroopwafel
        """
        self.num_hits = 0
        self.finished = 0
        for generation in range(NUM_GENERATIONS):
            samples = []
            self.entropies = []
            self.distribution_rejection_rate = self.calculate_rejection_rate()
            self.num_samples_per_generation = int(self.total_num_systems / NUM_GENERATIONS)
            while self.num_samples_per_generation > 0:
                batches = []
                for batch in range(min(self.num_batches_in_parallel, int(np.ceil(self.num_samples_per_generation / self.num_samples_per_batch)))):
                    current_batch = dict()
                    current_batch['number'] = self.batch_num
                    locations_ref = []
                    num_samples = np.ceil(self.num_samples_per_batch / len(self.adapted_distributions))
                    for distribution in self.adapted_distributions:
                        (locations, mask) = distribution.run_sampler(num_samples , self.dimensions, True)
                        locations_ref.extend(np.asarray(locations)[mask])
                    [location.revert_variables_to_original_scales() for location in locations_ref]
                    if self.update_properties_method != None:
                        self.update_properties_method(locations_ref, self.dimensions)
                    self.rejected_systems_method(locations_ref, self.dimensions)
                    locations_ref[:] = [location for location in locations_ref if location.properties['is_rejected'] == 0]
                    np.random.shuffle(locations_ref)
                    locations_ref = locations_ref[:self.num_samples_per_batch]
                    [location.properties.pop('is_rejected') for location in locations_ref]
                    current_batch['samples'] = locations_ref
                    samples.extend(locations_ref)
                    command = self.configure_code_run(current_batch)
                    generate_grid(locations_ref, current_batch['grid_filename'])
                    current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                    batches.append(current_batch)
                    self.batch_num = self.batch_num + 1
                self.process_batches(batches, False)
            self.calculate_weights_and_readjust_gaussians(samples)
            if self.finished >= self.total_num_systems:
                break
        print ("\nRefinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(self.num_hits, self.total_num_systems, self.num_hits / self.total_num_systems))

    def process_batches(self, batches, is_exploration_phase):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
            is_exploration_phase (Boolean) : Whether the given batches come from exploration phase or not
        """
        for batch in batches:
            if batch['process']:
                returncode = batch['process'].wait()
            hits = 0
            if returncode >= 0:
                hits = self.interesting_systems_method(batch)
            if (is_exploration_phase and not self.should_continue_exploring()) or self.finished >= self.total_num_systems or returncode < 0:
                #This batch is not needed anymore, delete the folder
                shutil.rmtree(self.output_folder + '/batch_' + str(batch['number']))
                self.batch_num = self.batch_num - 1
                continue
            self.num_hits += hits
            self.finished += self.num_samples_per_batch
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
                print_samples(batch['samples'], self.exploratory_filename, 'a')
            else:
                self.num_samples_per_generation -= self.num_samples_per_batch
                print_samples(batch['samples'], self.output_filename, 'a')
            printProgressBar(self.finished, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def calculate_weights_and_readjust_gaussians(self, locations):
        [location.transform_variables_to_new_scales() for location in locations]
        pi_norm = 1.0 / (1 - self.prior_fraction_rejected)
        q_norm = 1.0 / (1 - self.distribution_rejection_rate)
        samples = []
        mu = []
        sigma = []
        pi = []
        mask_hits = []
        [samples.append(location.to_array()) for location in locations]
        [pi.append(location.calculate_prior_probability() * pi_norm) for location in locations]
        [mu.append(distribution.mean.to_array()) for distribution in self.adapted_distributions]
        [sigma.append(distribution.cov) for distribution in self.adapted_distributions]
        [mask_hits.append(location.properties['is_hit']) for location in locations]
        samples = np.asarray(samples)
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        pi = np.asarray(pi)
        mask_hits = np.asarray(mask_hits)
        num_distributions = len(mu)
        num_samples = len(samples)
        xPDF = np.zeros((num_distributions, num_samples))
        for i in range(num_distributions):
            xPDF[i, :] = multivariate_normal.pdf(samples, mu[i], sigma[i], allow_singular = True)
        xPDF = xPDF.T
        qPDF = xPDF * self.alpha * q_norm
        weights = pi  / (np.sum(qPDF, axis = 1))
        with open(self.output_folder + '/weights.txt', 'a') as file:
            np.savetxt(file, weights)
        #Updating the gaussians from here
        if len(self.entropies) >= 2 and (self.entropies[-1] - self.entropies[-2]) < MAX_ENTROPY_CHANGE:
            return
        rho = qPDF / np.sum(qPDF, axis = 1)[:, None]
        gaussian_weights = np.asarray((pi * mask_hits) / np.sum(qPDF, axis = 1))
        weights_normalized = (gaussian_weights / np.sum(gaussian_weights))[:, None]
        self.alpha = np.sum(weights_normalized * rho, axis = 0)
        insignificant_components = np.argwhere(self.alpha < 1e-10)
        self.alpha = np.delete(self.alpha, insignificant_components)
        for index in range(len(self.dimensions)):
            mu[:, index] = np.sum(weights_normalized * samples[:, index][:, None] * rho, axis = 0)
        mu = np.delete(mu, insignificant_components, axis = 0)
        mu = mu / self.alpha[:, None]
        sigma = np.delete(sigma, insignificant_components, axis = 0)
        for i in range(len(mu)):
            distance = np.asarray(mu[i] - samples)[:, :, None]
            matrix = np.einsum('nij,nji->nij', distance, distance)
            factor = weights_normalized[:, 0] * rho [:, i]
            sigma[i] = np.sum(factor[:, None, None] * matrix, axis = 0) / self.alpha[i]
        self.adapted_distributions = self.adapted_distributions[:len(self.alpha)]
        for index, distribution in enumerate(self.adapted_distributions):
            for i, dimension in enumerate(sorted(distribution.mean.dimensions.keys(), key = lambda d: d.name)):
                distribution.mean.dimensions[dimension] = mu[index][i]
            distribution.cov = sigma[index]
        self.entropies.append(np.exp(entropy(weights_normalized)) / num_samples)

    def calculate_rejection_rate(self):
        num_rejected = 0
        N_GAUSS = 10000
        for distribution in self.adapted_distributions:
            mask = np.ones(N_GAUSS)
            samples = multivariate_normal.rvs(mean = distribution.mean.to_array(), cov = distribution.cov, size = N_GAUSS)
            samples = samples.T
            for index, dimension in enumerate(self.dimensions):
                mask = (mask == 1) & (samples[index] >= dimension.min_value) & (samples[index] <= dimension.max_value)
            num_rejected += N_GAUSS - np.sum(mask)
        return num_rejected / (N_GAUSS * len(self.adapted_distributions))
