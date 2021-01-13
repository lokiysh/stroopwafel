import os
import time
import shutil
from . import sw, distributions, constants


##############################################################
#### 
#### Functions to handle all the overall calls for stroopwafel
#### 
##############################################################

def run_stroopwafel(output_folder, output_filename, random_seed_base, 
        executable, extra_params,
        TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_BATCH, 
        time_request, debug, run_on_helios, mc_only,
        create_dimensions, update_properties, interesting_systems,
        selection_effects, rejected_systems):

    start_time = time.time()

    constants.RANDOM_SEED = random_seed_base # initialize random seed 

    def configure_code_run(batch):
        """
        This function tells stroopwafel what program to run, along with its arguments.
        IN:
            batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
                It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
                for each batch run in this dictionary. For example, here I have stored the 'output_container' and 'grid_filename' so that I can read them during discovery of interesting systems below
        OUT:
            exe_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
            Additionally one must also store the grid_filename in the batch so that the grid file is created
        """
        batch_num = batch['number']
        grid_filename = os.path.join(output_folder, 'grid_' + str(batch_num) + '.csv')
        output_container = 'batch_' + str(batch_num)
        exe_args = [executable, '--grid', grid_filename, '--output-container', output_container, '--output-path', output_folder]
        for params in extra_params:
            exe_args.extend(params.split("="))
        batch['grid_filename'] = grid_filename
        batch['output_container'] = output_container
        return exe_args

    print("Output folder is: ", output_folder)
    if os.path.exists(output_folder):
        command = input ("The output folder already exists. If you continue, I will remove all its content. Press (Y/N)\n")
        if (command == 'Y'):
            shutil.rmtree(output_folder)
        else:
            exit()
    os.makedirs(output_folder)


    # STEP 2 : Create an instance of the Stroopwafel class
    sw_object = sw.Stroopwafel(TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_BATCH, output_folder, output_filename, time_request, debug = debug, run_on_helios = run_on_helios, mc_only = mc_only)


    # STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
    dimensions = create_dimensions()
    sw_object.initialize(dimensions, interesting_systems, configure_code_run, rejected_systems, update_properties_method = update_properties)


    intial_pdf = distributions.InitialDistribution(dimensions)

    # STEP 4: Run the 4 phases of stroopwafel
    sw_object.explore(intial_pdf) #Pass in the initial distribution for exploration phase
    if not mc_only:
        sw_object.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
        # Do selection effects
        selection_effects(sw)
        sw_object.refine() #Stroopwafel will draw samples from the adapted distributions
        sw_object.postprocess(distributions.Gaussian, only_hits = False) #Run it to create weights, if you want only hits in the output, then make only_hits = True

    end_time = time.time()
    print ("Total running time = %d seconds" %(end_time - start_time))





