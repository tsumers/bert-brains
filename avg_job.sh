#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00
#SBATCH --output slurm_outputs/mean_0.out
#SBATCH --job-name img_avg_0
python -u save_mean_volumes.py encoding_onerep