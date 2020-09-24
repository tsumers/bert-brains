#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 30G  
#SBATCH --time 06:00:00
#SBATCH --output slurm_outputs/functional_space.out
#SBATCH --job-name functional_space
srun -n $SLURM_NTASKS --mpi=pmi2 python -u make_spaces.py

