import os 


d="/jukebox/griffiths/bert-brains/"


#Black 
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)] 
layer_prefix=d+'code/bert-brains/data/black/bert-base-uncased/raw_embeddings/'
save_prefix=d+"results/black/"
layer_dirs=[layer_prefix+"black_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names] 
save_dirs=[]
layer_prefix=d+'code/bert-brains/data/black/bert-base-uncased/syntactic_analyses/'
for fname in os.listdir(layer_prefix):
	layer_names.append(fname[:-4]) 
	layer_dirs.append(layer_prefix+fname)
	save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/")


""" Slumlordreach
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)] 
save_prefix=d+"results/slumlordreach/"
save_dirs=[save_prefix+"encoding-gpt_"+layer+"/" for layer in layer_names]

layer_prefix=d+'code/bert-brains/data/slumlordreach/gpt2/syntactic_analyses/'
for fname in os.listdir(layer_prefix):
	save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/")
"""


begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00"""

for i,direc in enumerate(save_dirs):
	out_name="#SBATCH --output "+"slurm_outputs/p_value_"+str(i)+".out"
	job_name="#SBATCH --job-name "+"bootstrap_"+str(i)
	with open("boot_job.sh","w") as out:
		out.write(begin+"\n")
		out.write(out_name+"\n")
		out.write(job_name+"\n")
		out.write("python -u calculate_bootstrap_pvalue.py "+direc)
		out.close()
	os.system('sbatch boot_job.sh')

