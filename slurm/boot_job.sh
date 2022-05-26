#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00
#SBATCH --output slurm_outputs/p_value_0.out
#SBATCH --job-name bootstrap_0
python -u calculate_bootstrap_pvalue.py encoding_onerep