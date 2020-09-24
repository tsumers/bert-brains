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
layer_names=[layer+"_activations" for layer in layers]+[layer+"_attention" for layer in layers[:-1]]
data_dir=d+"21styear_data/"

size=52422
step=5000
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
	numbers.append(size)

begin="""#!/usr/bin/env bash 
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --mem-per-cpu 18G
#SBATCH --time 1:00:00 
"""

for sub in subs:
	for i in range(len(layer_names)):
		layer_name=layer_names[i]
		layer_dir=layer_dirs[i]
		i=0
		if 'attention' in layer_name:
			begin_trim="1"
			end_trim="971"
		else:
			begin_trim="0"
			end_trim="970" 
		while i<len(numbers)-1:
			r1=numbers[i]
			r2=numbers[i+1]
			job_name="nn_sl_"+str(sub)+"_"+str(layer_name)+"_"+str(r1)+"_"+str(r2)
			output_name="/scratch/sreejank/slurm"+job_name+".out"
			with open("nn_job.sh","w") as f:
				f.write(begin+"\n")
				f.write("#SBATCH --job-name "+job_name+"\n")
				f.write("#SBATCH --output "+output_name+"\n")
				f.write("python func_nn_job.py "+sub+" "+layer_dir+" "+layer_name+" "+data_dir+" "+begin_trim+" "+end_trim+" "+str(r1)+" "+str(r2)+"\n")
				f.close()
			i+=1
			os.system('sbatch nn_job.sh')




