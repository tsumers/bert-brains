#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 180G  
#SBATCH --time 06:00:00
#SBATCH --output slurm_outputs/srm.out
#SBATCH --job-name srm_narratives
srun -n $SLURM_NTASKS --mpi=pmi2 python -u functional_alignment.py

