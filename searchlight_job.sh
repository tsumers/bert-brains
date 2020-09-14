#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 30G 
#SBATCH --time 03:00:00 

#SBATCH --output slurm_outputs/anat_sl_sub-075_layer_0.out
#SBATCH --job-name anat_sl_sub-075_layer_0
srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py sub-075 /jukebox/griffiths/bert-brains/model_rsms/21styear/bert-base-uncased/layer_0_rsm /jukebox/griffiths/bert-brains/results/21styear/rsa-bert-base-uncased-layer_0/ /jukebox/griffiths/bert-brains/21styear_data/
