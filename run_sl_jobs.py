import os

d='/jukebox/griffiths/bert-brains/'

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264', 
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271'] 

result_prefix=d+"results/21st_year/"

data_dir=d+'21styear_data/'

layer_names=[]
layer_dirs=[]
result_dirs=[]

layer_prefix=d+'code/bert-brains/data/21st_year/gpt2/syntactic_analyses/'

layer_names.append('21st_year_gpt2_semantic_composition')
layer_dirs.append(layer_prefix+"21st_year_gpt2_semantic_composition_max_l2.npy")
result_dirs.append(result_prefix+'rsa-21st_year_gpt2_semantic_composition/')
"""
layer_names.append('21st_year_gpt2_syntactic_complexity')
layer_dirs.append(layer_prefix+"21st_year_gpt2_syntactic_complexity_L-inf.npy")
result_dirs.append(result_prefix+'rsa-21st_year_gpt2_syntactic_complexity/')

layer_names.append('21st_year_gpt2_syntactic_distance') 
layer_dirs.append(layer_prefix+"21st_year_gpt2_syntactic_distance.npy")
result_dirs.append(result_prefix+'rsa-21st_year_gpt2_syntactic_distance/') 

layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/syntactic_analyses/'

layer_names.append('semantic_composition')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_semantic_composition_max_l2.npy")
result_dirs.append(result_prefix+'rsa-semantic_composition/')

layer_names.append('syntactic_complexity')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_complexity_L-inf.npy")
result_dirs.append(result_prefix+'rsa-syntactic_complexity/')

layer_names.append('syntactic_distance') 
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_distance.npy")
result_dirs.append(result_prefix+'rsa-syntactic_distance/') 
"""
for direc in result_dirs:
	if not(os.path.isdir(direc)):
		os.mkdir(direc)




begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 36G  
#SBATCH --time 07:00:00"""



for sub in subs: 
	for i in range(len(layer_dirs)):
		layer=layer_dirs[i]
		results_dir=result_dirs[i]

		#if 'syntactic_complexity' in layer:
		
		if 'gpt' in layer:
			if 'semantic' not in layer:
				begin_trim="18"
			else:
				begin_trim="15" 
		else:
			if 'semantic' in layer:
				begin_trim="16"
			elif 'syntactic' in layer:
				begin_trim="17"
			else:
				begin_trim="15"
		end_trim="2240"
		lname=layer_names[i]
		out_name="#SBATCH --output "+"slurm_outputs/anat_sl_"+str(sub)+"_"+lname+".out"
		job_name="#SBATCH --job-name "+"anat_sl_"+str(sub)+"_"+lname
		with open("searchlight_job.sh","w") as out:
			out.write(begin+"\n")
			out.write(out_name+"\n")
			out.write(job_name+"\n")
			out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py "+sub+" "+layer+" "+results_dir+" "+data_dir+" "+begin_trim+" "+end_trim+" rsa\n")  
			out.close()
		os.system("sbatch searchlight_job.sh")  


