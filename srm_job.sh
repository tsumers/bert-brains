#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 200G  
#SBATCH --time 04:00:00
#SBATCH --output slurm_outputs/srm_490_.out
#SBATCH --job-name srm_490
srun -n $SLURM_NTASKS --mpi=pmi2 python -u functional_alignment.py 490
