import os
from .utils import *
import shutil
import pandas as pd
import h5py as h5


class Stroopwafel:

    def __init__(self, total_num_systems, num_batches_in_parallel, num_samples_per_batch, output_folder, output_filename, debug = False, run_on_helios = True, mc_only = False):
        self.total_num_systems = total_num_systems
        self.num_batches_in_parallel = num_batches_in_parallel
        self.num_samples_per_batch = num_samples_per_batch
        self.output_folder = output_folder
        self.output_filename = os.path.join(self.output_folder, output_filename)
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

    def determine_rate(self, locations):
        """
        Function that determines the rate of producing the hits from the algorithm
        IN:
            hit_locations(list(Location)): All the locations
        OUT:
            (float, float): A pair of values which has the rate of stroopwafel rate and the uncertainity in the rate
        """
        phi = np.zeros(len(locations))
        for index, location in enumerate(locations):
            if location.properties['is_hit'] == 1:
                phi[index] = location.properties['mixture_weight']
        stroopwafel_rate = np.sum(phi) / len(locations)
        uncertainity = np.std(phi, ddof = 1) / np.sqrt(len(locations))
        return (np.round(stroopwafel_rate, 6), np.round(uncertainity, 6))

    def calculate_mixture_weights(self, n_dimensional_distribution_type, locations):
        """
        Function that will calculate the mixture weights of all the locations provided
        IN:
            locations (list(Location)) : All the locations for which weight needs to be computed
        """
        [location.properties.update({'mixture_weight': 1}) for location in locations]
        if self.num_explored == self.total_num_systems:
            return
        n_dimensional_distribution_type.calculate_probability_of_locations_from_distribution(locations, self.adapted_distributions)
        pi_norm = 1.0 / (1 - self.prior_fraction_rejected)
        q_norm = 1.0 / (1 - self.distribution_rejection_rate)
        fraction_explored = self.num_explored / self.total_num_systems
        for location in locations:
            prior_pdf = location.calculate_prior_probability() * pi_norm
            q_pdf = location.properties.pop('q_pdf') / len(self.adapted_distributions)
            Q = (fraction_explored * prior_pdf) + ((1 - fraction_explored) * q_pdf * q_norm)
            location.properties['mixture_weight'] = prior_pdf / Q

    def process_batches(self, batches, is_exploration_phase):
        """
        Function that waits for the completion of the commands which were running in batches
        IN:
            batches (list(dict)) : list of instance of batches, each having a batch number and a subprocess instance to wait for
            is_exploration_phase (Boolean) : Whether the given batches come from exploration phase or not
        """
        for batch in batches:
            # assign the exit status of the subprocess to "returncode"
            if batch['process']:
                returncode = batch['process'].wait()
                
            folder = os.path.join(self.output_folder, batch['output_container'])
            
            os.makedirs(folder, exist_ok=True)# Check if batch folder exists before you move the grid 
            shutil.move(batch['grid_filename'], os.path.join(folder, 'grid_' + str(batch['number']) + '.csv'))
            
            # Initialize the 'is_hit' to 0 for each sample in the batch 
            [location.properties.update({'is_hit': 0}) for location in batch['samples']]
            hits = 0
            
            # if you are still running, and the interesting_systems_method is defined, count hits using interesting_systems()
            if returncode >= 0 and self.interesting_systems_method is not None:
                # interesting_systems_method is the function interesting_systems(batch) from stroopwafel_interface.py
                hits = self.interesting_systems_method(batch)
            
            if (is_exploration_phase and not self.should_continue_exploring()) or self.finished >= self.total_num_systems or returncode < 0:                    
                # Delete the folder associated with the current batch
                print(' This batch is not needed anymore, delete the folder')
                shutil.rmtree(os.path.join(self.output_folder, 'batch_' + str(batch['number'])))
                
                self.batch_num = self.batch_num - 1
                # Continue to next batch
                continue
            
            self.num_hits += hits

            self.finished += self.num_samples_per_batch
            printProgressBar(self.finished, self.total_num_systems, prefix = 'progress', suffix = 'complete', length = 20)
            
            if is_exploration_phase:
                self.num_explored += self.num_samples_per_batch
                self.update_fraction_explored()
            else:
                self.num_to_be_refined -= self.num_samples_per_batch

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
        if not self.mc_only:
            # Calculate the rejection rate of the initial probability density function (pdf)
            self.prior_fraction_rejected = intial_pdf.calculate_rejection_rate(self.update_properties_method, self.rejected_systems_method, self.dimensions)
            # Log the rejection rate
            print_logs(self.output_folder, "prior_fraction_rejected", self.prior_fraction_rejected)
        else:
            # If running in Monte Carlo only mode, set the rejection rate to 0
            self.prior_fraction_rejected = 0

        # Continue exploring while the condition is met
        while self.should_continue_exploring():
            batches = []
            
            # Loop over the number of batches to run in parallel
            for batch in range(self.num_batches_in_parallel):
                # Initialize a dictionary for the current batch
                current_batch = dict()
                current_batch['number'] = self.batch_num
                
                # Calculate the number of samples, accounting for the rejection rate
                num_samples = int(2 * np.ceil(self.num_samples_per_batch / (1 - self.prior_fraction_rejected)))
                
                # Draw samples from the initial pdf (distributions.InitialDistribution(dimensions)) that is defined in the stroopwafel_interface
                # Note Location objects represent a point in N-Dimensional space (see Location class in classes.py)
                (locations, mask) = intial_pdf.run_sampler(num_samples)
                
                # ??
                [location.revert_variables_to_original_scales() for location in locations]
                
                # If an update properties method is provided, apply it to the locations
                if self.update_properties_method != None:
                    self.update_properties_method(locations, self.dimensions)
                    
                # If a rejected systems method is provided, apply it to the locations
                if self.rejected_systems_method != None:
                    self.rejected_systems_method(locations, self.dimensions)
                    
                # Filter out the locations that are rejected
                locations[:] = [location for location in locations if location.properties.get('is_rejected', 0) == 0]
                # no idea why we need to randomize here? -Lieke-
                np.random.shuffle(locations)
                
                # Trim the locations list to the desired number of samples per batch
                locations = locations[:self.num_samples_per_batch]
                # Remove the 'is_rejected' from each location
                [location.properties.pop('is_rejected', None) for location in locations]
                # Add the points in N-dimensional space to the current batch
                current_batch['samples'] = locations
                
                # Configure the code run for the current batch
                command = self.configure_code_run(current_batch)
                
                # Generate a COMPAS batch grid for the locations ()
                generate_grid(locations, current_batch['grid_filename'])
                
                # Run the grid and store the process in the current batch
                current_batch['process'] = run_code(command, current_batch['number'], self.output_folder, self.debug, self.run_on_helios)
                
                # Add the current batch to the batches list and increment the batch number
                batches.append(current_batch)
                self.batch_num = self.batch_num + 1
 
            # Process the batches in the exploration phase
            print('Lieke: Processing the batches in the exploration phase')
            self.process_batches(batches, True)
            
        if not self.mc_only:
            print ("\nExploratory phase finished, found %d hits out of %d explored. Rate = %.6f (fexpl = %.4f)" %(self.num_hits, self.num_explored, self.num_hits / self.num_explored, self.fraction_explored))
            print_logs(self.output_folder, "num_explored", self.num_explored)
            print_logs(self.output_folder, "fraction_explored", self.fraction_explored)

    def adapt(self, n_dimensional_distribution_type):
        """
        Adaptive phase of stroopwafel
        IN:
            n_dimensional_distribution_type(NDimensionalDistribution) : This tells stroopwafel what kind of distribution is to be adapted for refinment phase
        """
        print('IN ADAPT')
        if self.num_explored != self.total_num_systems:
            if self.num_hits > 0:
                hits = read_samples(self.output_filename, self.dimensions, only_hits = True)
                [location.transform_variables_to_new_scales() for location in hits]
                average_density_one_dim = 1.0 / np.power(self.num_explored, 1.0 / len(self.dimensions))
                self.adapted_distributions = n_dimensional_distribution_type.draw_distributions(hits, average_density_one_dim)
                print_distributions(os.path.join(self.output_folder, 'distributions.csv'), self.adapted_distributions)
                self.distribution_rejection_rate = n_dimensional_distribution_type.calculate_rejection_rate(self.adapted_distributions, self.update_properties_method, self.rejected_systems_method, self.dimensions)
                print_logs(self.output_folder, "distribution_rejection_rate", self.distribution_rejection_rate)
            print ("Adaptation phase finished!")
                
    def refine(self):
        """
        Refinement phase of stroopwafel
        """
        self.num_to_be_refined = self.total_num_systems - self.num_explored
        while self.num_to_be_refined > 0:
            batches = []
            for batch in range(min(self.num_batches_in_parallel, int(self.num_to_be_refined / self.num_samples_per_batch))):
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
        Postprocessing phase of stroopwafel
        IN:
            only_hits (Boolean) : If you want to print only the hits
        """
        locations = read_samples(self.output_filename, self.dimensions, only_hits)
        [location.transform_variables_to_new_scales() for location in locations]
        self.calculate_mixture_weights(n_dimensional_distribution_type, locations)
        [location.revert_variables_to_original_scales() for location in locations]
        print_samples(locations, self.output_filename, 'w')
        (stroopwafel_rate, uncertainity) = self.determine_rate(locations)
        print ("Rate of hits = %f with uncertainity = %f" %(stroopwafel_rate, uncertainity))
        print_logs(self.output_folder, "rate_of_hits", stroopwafel_rate)
        print_logs(self.output_folder, "uncertainity", uncertainity)
        
