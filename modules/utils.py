import numpy as np
import os
import subprocess

def generate_grid(locations, filename = 'grid.txt'):
    """
    Function which generated a txt file with the locations specified
    IN:
        locations (list[Location]) : list of locations to be printed in the file
        filename (string) : filename to save
    OUT:
        generates file with name filename with the given locations and saves to the disk
    """
    header = []
    grid = []
    for location in locations:
        current_location = []
        for key, value in location.dimensions.items():
            if key.should_print:
                if len(grid) == 0:
                    header.append(key.name)
                current_location.append(value)
        for key, value in location.properties.items():
            if len(grid) == 0:
                header.append(key)
            current_location.append(value)
        grid.append(current_location)
    DELIMITER = ', '
    np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments = '')

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

def print_hits(hit_locations, filename):
    """
    Function that prints all the hits to a file
    IN:
        hit_locations(list(Location)): All the hits that need to be printed
        filename (String) : The filename that will be saved
    """
    header = []
    grid = []
    for hit in hit_locations:
        current_hit = []
        hit.revert_variables_to_original_scales()
        for key, value in hit.properties.items():
            if len(grid) == 0:
                header.append(key)
            current_hit.append(value)
        for key, value in hit.dimensions.items():
            if len(grid) == 0:
                header.append(key.name)
            current_hit.append(value)
        grid.append(current_hit)
    DELIMITER = ', '
    np.savetxt(filename, grid, fmt = "%s", delimiter = DELIMITER, header = DELIMITER.join(header), comments = '')

def generate_slurm_file(command, batch_num, output_folder):
    slurm_folder = get_or_create_folder(output_folder, 'slurms')
    log_folder = get_or_create_folder(output_folder, 'logs')
    slurm_file = slurm_folder + "/slurm_" + str(batch_num) + ".sh"
    log_file = log_folder + "/log_" + str(batch_num) + ".txt"
    writer = open(slurm_file, 'w')
    writer.write("#!/bin/bash\n")
    writer.write("#SBATCH --output=output.out\n")
    writer.write(command + " > " + log_file + " \n")
    writer.close()
    return slurm_file

def run_code(command, batch_num, output_folder, debug = False, run_on_helios = True):
    """
    Function that runs the command specified on the command shell.
    IN:
        command list(String): A list of commands to be triggered along with the options
        batch_num (int) : The current batch number
    OUT:
        subprocess : An instance of subprocess created after running the command
    """
    if command != None:
        if not debug:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        else:
            stdout = stderr = None
        command_to_run = " ".join(str(v) for v in command)
        if run_on_helios:
            slurm_file = generate_slurm_file(" ".join(str(v) for v in command), batch_num, output_folder)
            command_to_run = "sbatch -W -Q " + slurm_file
        else:
            log_folder = get_or_create_folder(output_folder, 'logs')
            log_file = log_folder + "/log_" + str(batch_num) + ".txt"
            command_to_run = command_to_run + " > " + log_file
        process = subprocess.Popen(command_to_run, shell = True, stdout = stdout, stderr = stderr)
        return process

def get_slurm_output(output_folder, batch_num):
    try:
        log_folder = os.path.join(output_folder, 'logs')
        log_file = log_folder + "/log_" + str(batch_num) + ".txt"
        with open(log_file) as f:
            return f.readline()
    except:
        pass

def get_or_create_folder(path, name):
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder