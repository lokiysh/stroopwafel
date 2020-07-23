#!/bin/bash
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1:00:00
#SBATCH --output=output.out
/n/home04/lvanson/Programs/COMPAS/src/COMPAS --grid /n/home04/lvanson/Programs/stroopwafel/output/grid_5.csv --outputPath /n/home04/lvanson/Programs/stroopwafel/output --logfile-delimiter COMMA --output-container batch_5 --random-seed 3911400170357152419 > /n/home04/lvanson/Programs/stroopwafel/output/logs/log_5.txt 
