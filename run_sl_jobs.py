import os

d='/jukebox/griffiths/bert-brains/'
subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']

layer=d+'model_rsms/21styear/bert-base-uncased/layer_0_rsm'
results_dir=d+'results/21styear/rsa-bert-base-uncased-layer_0/'
data_dir=d+'21styear_data/'

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 30G 
#SBATCH --time 03:00:00 
"""

for sub in subs[:1]:
	out_name="#SBATCH --output "+"slurm_outputs/anat_sl_"+str(sub)+"_layer_0.out"
	job_name="#SBATCH --job-name "+"anat_sl_"+str(sub)+"_layer_0"
	with open("searchlight_job.sh","w") as out:
		out.write(begin+"\n")
		out.write(out_name+"\n")
		out.write(job_name+"\n")
		out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py "+sub+" "+layer+" "+results_dir+" "+data_dir+"\n")
		out.close()
	os.system("sbatch searchlight_job.sh")


