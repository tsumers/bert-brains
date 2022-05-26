#!/bin/bash
#SBATCH --output slurm_outputs/dsq-joblist.out
#SBATCH --array 0-62
#SBATCH --job-name encoding
#SBATCH --time "05:59:00"

# DO NOT EDIT LINE BELOW
/mnt/cup/labs/griffiths/bert-brains/code/bert-brains/dsq/dSQBatch.py --job-file /mnt/cup/labs/griffiths/bert-brains/code/bert-brains/joblist.txt --status-dir /mnt/cup/labs/griffiths/bert-brains/code/bert-brains/slurm_outputs


