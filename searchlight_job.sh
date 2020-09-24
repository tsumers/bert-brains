#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 18G  
#SBATCH --time 00:30:00
#SBATCH --output slurm_outputs/anat_sl_sub-271_layer_11_attention.out
#SBATCH --job-name anat_sl_sub-271_layer_11_attention
srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py sub-271 /jukebox/griffiths/bert-brains/code/bert-brains/data/21st_year/bert-base-uncased/all_attentions/layer_11_rsm.npy /jukebox/griffiths/bert-brains/results/21st_year/rsa-attentions_layer_11 /jukebox/griffiths/bert-brains/21styear_data/ 1 971
