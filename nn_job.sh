#!/usr/bin/env bash 
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --mem-per-cpu 18G
#SBATCH --time 1:00:00 

#SBATCH --job-name nn_sl_sub-271_layer_11_attention_50000_52422
#SBATCH --output /scratch/sreejank/slurmnn_sl_sub-271_layer_11_attention_50000_52422.out
python func_nn_job.py sub-271 /jukebox/griffiths/bert-brains/code/bert-brains/data/21st_year/bert-base-uncased/all_attentions/layer_11_rsm.npy layer_11_attention /jukebox/griffiths/bert-brains/21styear_data/ 1 971 50000 52422
