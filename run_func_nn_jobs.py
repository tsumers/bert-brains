import os
import time
d='/jukebox/griffiths/bert-brains/'

subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
			'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']

layer_names=['layer_'+str(i)+"_activations" for i in range(13)]
layer_dirs=[d+"code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_"+l+".npy" for l in layer_names]

data_dir=d+'slumlordreach_data/'   

size=52422  
step=11000
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
	numbers.append(size) 

begin="""#!/usr/bin/env bash 
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --mem-per-cpu 18G
#SBATCH --time 1:00:00 
"""
with open("joblist.txt","w") as f:
	for sub in subs:
		for i in range(len(layer_names)):
			layer_name=layer_names[i]
			layer_dir=layer_dirs[i]
			j=0 
			while j<len(numbers)-1: 
				r1=numbers[j]
				r2=numbers[j+1]
				job_name="nn_sl_"+str(sub)+"_"+str(layer_name)+"_"+str(r1)+"_"+str(r2)
				output_name="/scratch/sreejank/slurm"+job_name+".out"
				#f.write(begin+"\n") 
				#f.write("#SBATCH --job-name "+job_name+"\n")
				#f.write("#SBATCH --output "+output_name+"\n")
				f.write("python func_nn_job.py "+sub+" "+layer_dir+" "+layer_name+" "+data_dir+" "+" "+str(r1)+" "+str(r2)+"\n")
				j+=1
				#os.system('sbatch nn_job.sh')
				#time.sleep(2)
	f.close() 



