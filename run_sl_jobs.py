import os

d='/jukebox/griffiths/bert-brains/'

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271'] 

layers=['layer_'+str(i) for i in range(13)]
rsm_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/' 
layer_dirs=[rsm_prefix+layer+"_rsm.npy" for layer in layers]+[rsm_prefix+"all_attentions/"+layer+"_rsm.npy" for layer in layers[:-1]] 
results_prefix=d+'results/21st_year/'
if not(os.path.isdir(results_prefix)):
	os.mkdir(results_prefix)
results_dirs=[results_prefix+"rsa-activations_"+layer for layer in layers]+[results_prefix+"rsa-attentions_"+layer for layer in layers[:-1]]
for direc in results_dirs:
	if not(os.path.isdir(direc)):
		os.mkdir(direc)
assert len(layer_dirs)==len(results_dirs)

data_dir=d+'21styear_data/'


begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 18G  
#SBATCH --time 00:30:00"""



for sub in subs:
	for i in range(len(layer_dirs)):
		layer=layer_dirs[i]
		results_dir=results_dirs[i]
		if 'attention' in layer: 
			begin_trim="1"
			end_trim="971"
			lname=layers[i%len(layers)]+'_attention'
		else:
			begin_trim="0"
			end_trim="970"
			lname=layers[i%len(layers)]+'_activation'

		out_name="#SBATCH --output "+"slurm_outputs/anat_sl_"+str(sub)+"_"+lname+".out"
		job_name="#SBATCH --job-name "+"anat_sl_"+str(sub)+"_"+lname
		with open("searchlight_job.sh","w") as out:
			out.write(begin+"\n")
			out.write(out_name+"\n")
			out.write(job_name+"\n")
			out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py "+sub+" "+layer+" "+results_dir+" "+data_dir+" "+begin_trim+" "+end_trim+"\n") 
			out.close()
		os.system("sbatch searchlight_job.sh")  


