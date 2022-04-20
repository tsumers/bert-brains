#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 5G  
#SBATCH --time 16:00:00
#SBATCH --output slurm_outputs/master_job.out
#SBATCH --job-name master
srun -n $SLURM_NTASKS --mpi=pmi2 python -u run_func_nn_jobs.py

