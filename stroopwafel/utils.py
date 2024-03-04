import numpy as np
import os
import subprocess
import csv
from .classes import Location
from .constants import *
import math

def generate_grid(locations, filename):
    """
    Function which generated a txt file with the locations specified
    IN:
        locations (list[Location]) : list of locations to be printed in the file
        filename (string) : filename to save
    OUT:
        generates file with name filename with the given locations and saves to the disk
    """    
    # Initialize an empty list for the grid
    grid = []

    # Loop over each location in the locations list
    for location in locations:
        current_location = [] # Initialize an empty list for the current location

        # Loop over each dimension; things you are optimizing for (e.g. --initial-mass-1) 
        for key, value in location.dimensions.items():
            # If the dimension should be printed, add it to the current location list
            if key.should_print:
                current_location.append(key.name + ' ' + str(value))

        # Loop over each property; other necessary params, which you are not optimizing for (e.g. --metallicity) 
        for key, value in location.properties.items():
            # If the property is 'generation' or 'gaussian', skip it
            if key in ['generation', 'gaussian']:
                continue
            # Otherwise, add it to the current location list
            else:
                current_location.append(key + ' ' + str(value))

        # Add the current location list to the grid list
        grid.append(current_location)
        
    DELIMITER = ' '
    np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, comments = '')
    

#copied from stack overflow
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|', autosize = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        autosize    - Optional  : automatically resize the length of the progress bar to the terminal window (Bool)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    styling = '%s |%s| %s%% %s' % (prefix, fill, percent, suffix)
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s' % styling.replace(fill, bar), end = '\r')
    # Print New Line on Complete
    if iteration >= total:
        print()

def print_samples(samples, filename, mode):
    """
    Function that saves all the hits to a file
    IN:
        samples(list(Location)): All the samples that need to be printed
        filename (String) : The filename that will be saved
        mode (Character) : Append mode / write mode / read mode, which mode to open the file with
    """
    with open(filename, mode) as file:
        for sample in samples:
            current_dict = {}
            for dimension in sorted(sample.dimensions.keys(), key = lambda d: d.name):
                current_dict[dimension.name] = sample.dimensions[dimension]
            for prop in sorted(sample.properties.keys()):
                current_dict[prop] = sample.properties[prop]
            writer = csv.DictWriter(file, current_dict.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(current_dict)

def read_samples(filename, dimensions, only_hits = False):
    """
    Function that reads samples from a given file and convert it to Location objects
    IN:
        filename (String) : The filename that will be read
        dimensions (List(Dimension)) : List of dimensions to make variables
        only_hits (Boolean) : If true it will only return the Locations which are target hits
    OUT:
        locations (List(Location)) : List of locations generated
    """
    with open(filename, newline = '') as file:
        samples = csv.DictReader(file)
        dimensions_hash = dict()
        for dimension in dimensions:
            dimensions_hash[dimension.name] = dimension
        locations = []
        for sample in samples:
            if only_hits and int(sample['is_hit']) == 0:
                continue
            sample.update((k, float(v)) for k, v in sample.items())
            locations.append(Location.create_location(dimensions_hash, sample))
        return locations

def generate_slurm_file(command, batch_num, output_folder):
    """
    Function that generates a slurm file to run the given command on helios batch
    IN:
        command (String) : The command to be run
        batch_bum (int) : Batch number, would be used to generate file name
        output_folder (Path) : Where to generate the output
    OUT:
        slurm_file (String) : Path of the file generated
    """
    slurm_folder = get_or_create_folder(output_folder, 'slurms')
    log_folder = get_or_create_folder(output_folder, 'logs')
    slurm_file = os.path.join(slurm_folder, "slurm_" + str(batch_num) + ".sh")
    log_file = os.path.join(log_folder, "log_" + str(batch_num) + ".txt")
    writer = open(slurm_file, 'w')
    writer.write("#!/bin/bash\n")
    writer.write("#SBATCH --mem=4G\n")
    # writer.write(f"#SBATCH --output={slurm_folder}/batch_{batch_num}.out\n")
    # writer.write(f"#SBATCH --error={slurm_folder}/batch_{batch_num}.err\n")
    writer.write("#SBATCH -t 0-04:00:00\n") # max 4 hours (which is a very high upper boud)
    # Lieke: Customizing the slurm file (your modules and partitions might be different)!!!
    # writer.write("#SBATCH -p genx \n") # genx is for  small serial jobs 
    writer.write("module load gsl boost hdf5 gcc python \n")
    ##
    writer.write(f"{command} >  {log_file}  2> {slurm_folder}/batch_{batch_num}.err  \n")
    writer.close()
    return slurm_file


def run_code(command, batch_num, output_folder, debug = True, run_on_helios = True):
    """
    Function that runs the command specified on the command shell.
    IN:
        command list(String): A list of commands to be triggered along with the options
        batch_num (int) : The current batch number to generate the filename
        output_folder (Path) : Where to generate the output
        debug (Boolean) : If true will print stuff to console
        run_on_helios (Boolean) : If true will generate slurm file to run on helios
    OUT:
        subprocess : An instance of subprocess created after running the command
    """
    if command != None:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
        
        command_to_run = " ".join(str(v) for v in command)
        
        if run_on_helios:
            slurm_file = generate_slurm_file(" ".join(str(v) for v in command), batch_num, output_folder)
            
            # command_to_run = "sbatch -W -Q " + slurm_file
            # srun in stead of sbatch will use already allocated resources to run each subcommand
            command_to_run = "srun -n 1 bash " + slurm_file # suggested by NJC 
        else:
            log_folder = get_or_create_folder(output_folder, 'logs')
            log_file = os.path.join(log_folder, "log_" + str(batch_num) + ".txt")
            command_to_run = command_to_run + " > " + log_file
        process = subprocess.Popen(command_to_run, shell = True, stdout = stdout, stderr = stderr)
        
        return process


def get_slurm_output(output_folder, batch_num):
    """
    Reads the output generated by the slurm run
    IN:
        output_folder (Path) : Where the output resides
        batch_num (int) : Will be used to generate file name
    OUT:
        (String) : The output returned by slurm run
    """
    try:
        log_folder = os.path.join(output_folder, 'logs')
        log_file = os.path.join(log_folder, "log_" + str(batch_num) + ".txt")
        with open(log_file) as f:
            return_list = [float(line.rstrip()) for line in f]
            return return_list
    except:
        pass

def get_or_create_folder(path, name):
    """
    Utility function to create a folder at a given path and name or return it if it already exists
    path (String) : Path where it is created
    name (String) : Name of the folder
    """
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def print_distributions(filename, distributions):
    with open(filename, 'w') as file:
        for distribution in distributions:
            current_dict = dict()
            for dimension, value in distribution.mean.dimensions.items():
                current_dict[dimension.name + '_mean'] = value
            for dimension, value in distribution.sigma.dimensions.items():
                current_dict[dimension.name + '_sigma'] = value
            writer = csv.DictWriter(file, current_dict.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(current_dict)

def print_logs(output_folder, log_string, log_value):
    with open(os.path.join(output_folder, 'log.txt'), 'a') as file:
        file.write("%s = %f\n"%(log_string, log_value))

def get_zams_radius(mass, metallicity):
    metallicity_xi = math.log10(metallicity / ZSOL)
    radius_coefficients = []
    for coeff in R_COEFF:
        value = 1
        total = 0
        for series in coeff:
            total += series * value
            value *= metallicity_xi
        radius_coefficients.append(total)
    top = radius_coefficients[0] * pow(mass, 2.5) + radius_coefficients[1] * pow(mass, 6.5) \
        + radius_coefficients[2] * pow(mass, 11) + radius_coefficients[3] * pow(mass, 19) \
        + radius_coefficients[4] * pow(mass, 19.5)
    bottom = radius_coefficients[5] + radius_coefficients[6] * pow(mass, 2) \
        + radius_coefficients[7] * pow(mass, 8.5) + pow(mass, 18.5) \
        + radius_coefficients[8] * pow(mass, 19.5)
    radius = top / bottom
    return radius * R_SOL_TO_AU

def calculate_roche_lobe_radius(mass1, mass2):
    q = mass1 / mass2
    return 0.49 / (0.6 + pow(q, -2.0 / 3.0) * math.log(1.0 + pow(q, 1.0 / 3.0)))

def inverse_back(dimension, inverse):
    norm_factor = (ALPHA_IMF + 1) / (pow(dimension.max_value, ALPHA_IMF + 1) - pow(dimension.min_value, ALPHA_IMF + 1))
    return norm_factor / pow(inverse * (pow(norm_factor / dimension.max_value, 1 / -ALPHA_IMF) \
        - pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF)) \
        + pow(norm_factor / dimension.min_value, 1 / -ALPHA_IMF), -ALPHA_IMF)
