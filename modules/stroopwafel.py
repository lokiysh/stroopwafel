import os
from utils import *
from scipy.stats import multivariate_normal

class Stroopwafel:

    def __init__(self, num_systems = 100, num_batches = 1, num_samples_per_batch = 100, output_folder = os.getcwd(), debug = False, run_on_helios = True, mc_only = False):
        self.num_systems = num_systems
        self.num_batches = num_batches
        self.num_samples_per_batch = num_samples_per_batch
        self.output_folder = output_folder
        self.debug = debug
        self.run_on_helios = run_on_helios
        self.mc_only = mc_only

    def update_fraction_explored(self):
        """
        Function which updates the fraction of region which is already explored
        """
        num_hits = len(self.hits)
        unidentified_region_weight = 1.0 / (self.fraction_explored * self.num_systems)
        target_rate = num_hits / self.num_explored
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
            return self.num_explored < self.num_systems
        return self.num_explored / self.num_systems < self.fraction_explored

    def determine_rate(self, hit_locations):
        """
        Function that determines the rate of producing the hits from the algorithm
        IN:
            hit_locations(list(Location)): All the hits that need to be printed
        OUT:
            (float, float): A pair of values which has the rate of stroopwafel rate and the uncertainity in the rate
        """
        if len(hit_locations) == 0:
            return (0, 0)
        phi = np.ones(len(hit_locations))
        for index, location in enumerate(hit_locations):
            phi[index] = location.properties['mixture_weight']
        stroopwafel_rate = np.round(np.sum(phi) / self.num_systems, 4)
        uncertainity = np.round(np.std(phi, ddof = 1) / np.sqrt(self.num_systems), 6)
        return (stroopwafel_rate, uncertainity)

    def process_batches(self, batches, is_exploration_phase):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
        """
        for batch in batches:
            if batch['process']:
                batch['process'].wait()
            if self.interesting_systems_method != None:
                locations = self.interesting_systems_method(batch)
                for location in locations:
                    location.transform_variables_to_new_scales()
                    location.calculate_prior_probability()
                if is_exploration_phase:
                    # TODO: Get q_pdf from the original N dimensional distribution, for now setting as prior
                    for location in locations:
                        location.properties['q_pdf'] = location.properties['p']
                        location.properties['mixture_weight'] = 1
                        location.properties['exact_weight'] = 1
                else:
                    for distribution in self.adapted_distributions:
                        distribution.calculate_probability_of_locations_from_distribution(locations)
                    for location in locations:
                        location.properties['q_pdf'] /= len(self.adapted_distributions) #normalized
                        Q = (self.fraction_explored * location.properties['p']) + ((1 - self.fraction_explored) * location.properties['q_pdf'])
                        location.properties['mixture_weight'] = location.properties['p'] / Q
                        location.properties['exact_weight'] = location.properties['p'] / location.properties['q_pdf']
                self.hits.extend(locations)
                print_hits(locations, filename = self.output_folder + "/hits.csv")
            self.finished += self.num_samples_per_batch
            printProgressBar(self.finished, self.num_systems, prefix = 'progress', suffix = 'complete', length = 20)
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
                if not self.should_continue_exploring():
                    break
            if self.finished >= self.num_systems:
                break

    def initialize(self, interesting_systems_method, configure_code_run, num_dimensions, update_properties_method = None):
        """
        This function is the one which is run only once in the stroopwafel class. It initializes the associated variables and the function calls that user will specify
        IN:
            interesting_system_method: The method provided by the user which will determine what an interesting system is for stroopwafel
            update_properties_method: The method provided by the user which will run to update the properties of each of the location once it is sampled
            configure_code_run: The method provided by the user which will be running for each of the batches to determine the command line args for that batch
        """
        self.interesting_systems_method = interesting_systems_method
        self.update_properties_method = update_properties_method
        self.configure_code_run = configure_code_run
        self.num_dimensions = num_dimensions
        self.batch_num = 0
        self.num_explored = 0
        self.finished = 0
        self.hits = []
        self.fraction_explored = 1
        printProgressBar(0, self.num_systems, prefix = 'progress', suffix = 'complete', length = 20)

    def explore(self, dimensions, intial_pdf):
        """
        This function is the exploration phase of the stroopwafel
        IN:
            dimensions (list(Dimension)) : A list of user created dimensions, each an instance of Dimension class
            initial_pdf (NDimensionalDistribution) : An instance of NDimensionalDistribution showing how to sample from in the exploration phase
        """
        while self.should_continue_exploring():
            batches = []
            for batch in range(self.num_batches):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                (locations, mask) = intial_pdf.run_sampler(self.num_samples_per_batch)
                [location.revert_variables_to_original_scales() for location in locations]
                if self.update_properties_method != None:
                    self.update_properties_method(locations)
                command = self.configure_code_run(current_batch)
                generate_grid(locations, current_batch['grid_filename'])
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.process_batches(batches, True)
        print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(len(self.hits), self.num_explored, len(self.hits) / self.num_explored, self.fraction_explored))

    def adapt(self, n_dimensional_distribution_type):
        """
        Adaptive phase of stroopwafel
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        self.num_to_be_refined = self.num_systems - self.num_explored
        if self.num_to_be_refined > 0:
            average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / self.num_dimensions)
            self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(self.hits, average_density_one_dim)
            n_dimensional_distribution_type.calculate_rejection_rate(self.adapted_distributions, self.num_batches, self.output_folder, self.debug, self.run_on_helios)
            print ("Adaptation phase finished!")
                
    def refine(self):
        """
        Refinement phase of stroopwafel
        """
        refined = False
        while self.num_to_be_refined > 0:
            batches = []
            for batch in range(self.num_batches):
                current_batch = dict()
                current_batch['number'] = self.batch_num
                locations_ref = []
                for distribution in self.adapted_distributions:
                    (locations, mask) = distribution.run_sampler(self.num_samples_per_batch, True)
                    locations_ref.extend(np.asarray(locations)[mask])
                np.random.shuffle(locations_ref)
                locations_ref = locations_ref[0 : self.num_samples_per_batch]
                [location.revert_variables_to_original_scales() for location in locations_ref]
                if self.update_properties_method != None:
                    self.update_properties_method(locations_ref)
                command = self.configure_code_run(current_batch)
                generate_grid(locations_ref, current_batch['grid_filename'])
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
            self.process_batches(batches, False)
            self.num_to_be_refined -= self.num_batches * self.num_samples_per_batch
            refined = True
        if refined:
            print ("\nRefinement phase finished, found %d hits out of %d tried. Rate = %.6f" %(len(self.hits) - len(self.adapted_distributions), (self.num_systems - self.num_explored), (len(self.hits) - len(self.adapted_distributions)) / (self.num_systems - self.num_explored)))

    def postprocess(self, filename):
        """
        Postprocessing phase of stroopwafel
        IN:
            filename(String): It tells what is the filename to store all the hits and its properties
        """
        (stroopwafel_rate, uncertainity) = self.determine_rate(self.hits)
        print ("Rate of hits = %f with uncertainity = %f" %(stroopwafel_rate, uncertainity))
