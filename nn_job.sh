#!/usr/bin/env bash 
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --mem-per-cpu 18G
#SBATCH --time 6:00:00 

#SBATCH --job-name nn_sl_sub-271_layer_12_activations_44000_52422
#SBATCH --output /scratch/sreejank/slurm/nn_sl_sub-271_layer_12_activations_44000_52422.out
python encoding_model.py sub-271 /jukebox/griffiths/bert-brains/code/bert-brains/data/21st_year/bert-base-uncased/raw_embeddings/layer_12_activations.npy layer_12_activations /jukebox/griffiths/bert-brains/21styear_data/ 15 2240 44000 52422
