#!/bin/bash
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1:00:00
#SBATCH --output=output.out
/n/home04/lvanson/Programs/COMPAS/src/COMPAS --grid /n/home04/lvanson/Programs/stroopwafel/output/grid_3.csv --outputPath /n/home04/lvanson/Programs/stroopwafel/output --logfile-delimiter COMMA --output-container batch_3 --random-seed 9175444627444954001 > /n/home04/lvanson/Programs/stroopwafel/output/logs/log_3.txt 
