#!/bin/bash
#SBATCH --mem-per-cpu=6144
#SBATCH --time=5:00:00
#SBATCH --output=output.out
python /n/home04/lvanson/Programs/stroopwafel/modules/find_rejection_rate.py '{"exploration": 1}' > ./output/logs/log_initial_8.txt 
