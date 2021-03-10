#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 36G  
#SBATCH --time 07:00:00
#SBATCH --output slurm_outputs/anat_sl_sub-271_21st_year_gpt2_semantic_composition.out
#SBATCH --job-name anat_sl_sub-271_21st_year_gpt2_semantic_composition
srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py sub-271 /jukebox/griffiths/bert-brains/code/bert-brains/data/21st_year/gpt2/syntactic_analyses/21st_year_gpt2_semantic_composition_max_l2.npy /jukebox/griffiths/bert-brains/results/21st_year/rsa-21st_year_gpt2_semantic_composition/ /jukebox/griffiths/bert-brains/21styear_data/ 15 2240 rsa
