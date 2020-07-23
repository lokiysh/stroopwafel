#!/bin/bash
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1:00:00
#SBATCH --output=output.out
python /n/home04/lvanson/Programs/stroopwafel/modules/find_rejection_rate.py '{"exploration": 1}' > /n/home04/lvanson/Programs/stroopwafel/output/logs/log_initial_2.txt 
