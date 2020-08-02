##############################################
#
# GridCallStroopwafel.py
#
# Python script meant to call STROOPWAFEL-interface.py for a variety of grid values. 
# This file calls interface.py, which calls COMPAS in a bunch of slurm scripts ##
# created 30-06-2020
# !! Many of these functions are adapted from compas_hpc_functions.py
#
##############################################
import numpy as np
import os
import time
from subprocess import Popen, PIPE
import sys
import pickle
import math
import shutil

SlurmJobStringTemplateWithEmail="""#!/bin/bash
#SBATCH --job-name=%s 			#job name
#SBATCH --nodes=%s 			#
#SBATCH --ntasks=%s 			# Number of cores
#SBATCH --partition=serial_requeue 	# Partition to submit to
#SBATCH --output=%s 			# output storage file
#SBATCH --error=%s 			# error storage file
#SBATCH --time=%s 			# Runtime in minutes
#SBATCH --mem=%s 			# Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --mail-user=%s 			# Send email to user
#SBATCH --mail-type=END			#
#
#Print some stuff on screen
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export COMPAS_ROOT_DIR=%s
export SW_ROOT_DIR=%s
export BASE_DIR=%s
export OUT_DIR=%s
#
#copy relevant files
cp $SW_ROOT_DIR/GridCallStroopwafel_input.py $BASE_DIR/
cp $SW_ROOT_DIR/COMPAS_Output_Definitions.txt $BASE_DIR/
cp $SW_ROOT_DIR/randomSeed.txt $BASE_DIR/
#
#CD to folder
cd $SW_ROOT_DIR
#
# Run interface.py with command-line Flags
python $SW_ROOT_DIR/LiekeInterface.py --output_folder=$OUT_DIR --run_on_helios=True --num_systems=%s --num_per_core=%s --num_cores=%s --logfile-definition=$BASE_DIR/COMPAS_Output_Definitions.txt --compas_arg_flags="%s" --compas_arg_vals %s %s > $OUT_DIR/LiekeInterface.log 

cp $COMPAS_ROOT_DIR/CompasHPC/postProcessing.py  $OUT_DIR/
cp $COMPAS_ROOT_DIR/postProcessing/Folders/CosmicIntegration/PythonScripts_lieke/ComputeCIweights.py  $OUT_DIR/
"""
#--dependency=afterok:' + str(dependencyID)
# --compas_arg_flags="%s" --compas_arg_vals %s %s

PP_SlurmJobStringTemplate="""#!/bin/bash
#SBATCH --job-name=COMPAS_PP 		#job name
#SBATCH --nodes=1 			#
#SBATCH --ntasks=1 			# Number of cores
#SBATCH --partition=serial_requeue 	# Partition to submit to
#SBATCH --output=%s 			# output storage file
#SBATCH --error=%s 			# error storage file
#SBATCH --time=10:00:00 		# Runtime in minutes
#SBATCH --mem=0 			# Memory per cpu in MB (0 means unlimited)
#
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export OUT_DIR=%s
cd $OUT_DIR
#Run postProcessing to combine output
python $OUT_DIR/postProcessing.py --masterFolderDir $OUT_DIR/masterFolder > $OUT_DIR/COMPAS_PP.log
"""
#

CI_SlurmJobStringTemplate="""#!/bin/bash
#SBATCH --job-name=COMPAS_CI 		#job name
#SBATCH --nodes=1 			#
#SBATCH --ntasks=1 			# Number of cores
#SBATCH --partition=serial_requeue 	# Partition to submit to
#SBATCH --output=%s 			# output storage file
#SBATCH --error=%s 			# error storage file
#SBATCH --time=10:00:00 		# Runtime in minutes
#SBATCH --mem=0 			# Memory per cpu in MB (0 means unlimited)
#
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export OUT_DIR=%s
cd $OUT_DIR
#Run Coens cosmic integration to calculate CI weights
python $OUT_DIR/ComputeCIweights.py > $OUT_DIR/COMPAS_CI.log
"""
#

def runBashCommand(bashCommand, verbose=True):
	"""
	Run bash command
	Parameters
	-----------
	bashCommand : str
		Bash command to be run
	verbose : boolean
		Whether to echo command being run
	Returns
	--------
	"""
	if(verbose):
		print(bashCommand)
	os.system(bashCommand)
	return

###############################################
###		 MAIN      ###
###############################################
if __name__ == '__main__':
	# def main():
	###############################################
	# set variables
	###############################################
	job_name 		= "StroopGrid"
	number_of_nodes = 1
	number_of_cores = 50 
	walltime 		= "5:30:00"
	memory 			= 0#12000
	send_email		= True
	user_email 		= "aac.van.son@gmail.com"
	COMPAS_ROOT_DIR = "/n/home04/lvanson/Programs/COMPAS/" #"/Users/lieke/Documents/COMPAS/"
	SW_ROOT_DIR 	= "/n/home04/lvanson/Programs/stroopwafel/"#"/Users/lieke/surfdrive/Documents/CompareCOMPAS/Code/"

	#Parameters for interface.py
	I_num_systems	= int(1e5) # Number of systems per grid point
     # !! WARNING each batch has a max run time of 1 Hr and mem 1Gb
     # This corresponds to approx 3000 sys per core
	I_num_per_core	= 1000  # Number of systems run per core
	I_num_of_core	= 10	# number of cores/batches to run per grid point

	###############################################
	# Base outbut dir
	###############################################
	base_OUT_DIR			= "/n/de_mink_lab/Users/lvanson/CompasOutput/CItest_Default_N1e5"
	#"/Users/lieke/surfdrive/Documents/CompareCOMPAS/CompasOutput/SW/GridTest3x3/"#
	pklName='pickledGrid.pkl'
	verbose = True

	if os.path.exists(base_OUT_DIR):
		print('!!! REMOVING EXISTING DIR %s !!!' % base_OUT_DIR )
		shutil.rmtree(base_OUT_DIR)

	###############################################
	# Create Grid and Grid directories
	###############################################
	#  To see available options to change do 
	#  $COMPAS_ROOT_DIR/COMPAS/COMPAS --help
	gridDictionary = {}
	gridDictionary['--common-envelope-alpha'] = [1.0]#np.linspace(1.,10.,1) 
	gridDictionary['--wolf-rayet-multiplier'] = [1.0]#np.linspace(0.2,2,1) 
	dirNames = ['alpha', 'fWR']
	keys = list(gridDictionary.keys())
	I_flags			= "--common-envelope-alpha --wolf-rayet-multiplier"

	###############################################
	# !!! CURRENT STRUCTURE WORKS ONLY FOR 2D GRID !!!
	nGridPoints = 1
	for key in gridDictionary.keys():
		print(key, gridDictionary[key])
		nGridPoints *= len(gridDictionary[key])

	for j in range(nGridPoints):
		l = math.trunc(j/len(gridDictionary[keys[0]]) )
		k = j - (len(gridDictionary[keys[0]]) * l) 
		# print('k',k, 'l', l)
		alpha = gridDictionary[keys[0]][k] 
		fWR = gridDictionary[keys[1]][l]
		gridPointDir = dirNames[0]+str(alpha)+'_'+dirNames[1]+str(fWR)
		print('Making ', gridPointDir)

		OUT_DIR			= base_OUT_DIR + gridPointDir +'/' #"/n/de_mink_lab/Users/lvanson/CompasOutput/StroopwafelTest/GridTest/"
		outfile 		= OUT_DIR + "StroopGrid.out"
		errfile 		= OUT_DIR + "StroopGrid.err"


		###############################################
		# Make the output directory
		###############################################
		if not os.path.exists(OUT_DIR):
			os.makedirs(OUT_DIR)

		###############################################
		# Generate Slurm Job String for interface.py
		###############################################
		interface_job_string = SlurmJobStringTemplateWithEmail % (job_name, number_of_nodes, number_of_cores, outfile, errfile, walltime, memory, user_email,COMPAS_ROOT_DIR, SW_ROOT_DIR, base_OUT_DIR, OUT_DIR, I_num_systems, I_num_per_core, I_num_of_core, I_flags, alpha, fWR)
		#
		# print(interface_job_string)
		# Save to a file
		sbatchFile = open(OUT_DIR+'/SW_Slurm.sbatch','w')
		sbatchFile.write(interface_job_string)
		sbatchFile.close()

		###############################################
		# Send SW command to bash
		###############################################	
		sbatchArrayCommand = 'sbatch ' + os.path.join(OUT_DIR+'/SW_Slurm.sbatch') 
		# Open a pipe to the sbatch command.
		proc = Popen(sbatchArrayCommand, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
		## Send job_string to sbatch
		if (sys.version_info > (3, 0)):
			proc.stdin.write(sbatchArrayCommand.encode('utf-8'))
		else:
			proc.stdin.write(sbatchArrayCommand)

		print('sbatchArrayCommand', sbatchArrayCommand)
		out, err = proc.communicate()
		print("out = ", out)
		main_job_id = out.split()[-1]
		print("main_job_id", main_job_id)

		###############################################
		# Generate Slurm Job String for postprocessing
		###############################################
		PPoutfile 		= OUT_DIR + "COMPAS_PP.out"
		PPerrfile 		= OUT_DIR + "COMPAS_PP.err"
		PP_job_string = PP_SlurmJobStringTemplate % (PPoutfile, PPerrfile, OUT_DIR)
		# print(PP_job_string)
		# Save to a file
		PPsbatchFile = open(OUT_DIR+'/postProcessing.sbatch','w')
		PPsbatchFile.write(PP_job_string)
		PPsbatchFile.close()

		###############################################
		# Send PP command to bash with afterok main COMPAS
		###############################################	        
		sbatchPPCommand = 'sbatch --dependency=afterok:' + str(int(main_job_id)) + ' ' + os.path.join(OUT_DIR+'/postProcessing.sbatch') 
		# Open a pipe to the sbatch command.
		proc = Popen(sbatchPPCommand, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
		# Send job_string to sbatch
		if (sys.version_info > (3, 0)):
		        proc.stdin.write(sbatchPPCommand.encode('utf-8'))
		else:
		        proc.stdin.write(sbatchPPCommand)

		print('sbatchPPCommand', sbatchPPCommand)
		ppOut, pperr = proc.communicate()
		print("ppOut = ", ppOut)
		print("err = ", pperr)
		ppJobID = ppOut.split()[-1]
		print("Post-processing job ID = ", ppJobID)


		###################################################
		# Generate Slurm Job String for Cosmic integration
		###################################################
		CIoutfile 		= OUT_DIR + "COMPAS_CI.out"
		CIerrfile 		= OUT_DIR + "COMPAS_CI.err"
		CI_job_string   = CI_SlurmJobStringTemplate % (CIoutfile, CIerrfile, OUT_DIR)
		# print(PP_job_string)
		# Save to a file
		CIsbatchFile = open(OUT_DIR+'/CosmicIntegration.sbatch','w')
		CIsbatchFile.write(CI_job_string)
		CIsbatchFile.close()

		##########################################################
		# Send Cosmic integration command to bash with afterok PP
		##########################################################        
		sbatchCICommand = 'sbatch --dependency=afterok:' + str(int(ppJobID)) + ' ' + os.path.join(OUT_DIR+'/CosmicIntegration.sbatch') 
		# Open a pipe to the sbatch command.
		proc = Popen(sbatchCICommand, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
		# Send job_string to sbatch
		if (sys.version_info > (3, 0)):
		        proc.stdin.write(sbatchPPCommand.encode('utf-8'))
		else:
		        proc.stdin.write(sbatchPPCommand)

		print('sbatchPPCommand', sbatchPPCommand)
		CIOut, CIerr = proc.communicate()
		print("ppOut = ", CIOut)
		print("err = ", CIerr)
		CIJobID = CIOut.split()[-1]
		print("Cosmic integration job ID = ", CIJobID)


		###############################################
		# Run bash command (OLD)
		###############################################
		# runBashCommand(interface_job_string, verbose=False)


