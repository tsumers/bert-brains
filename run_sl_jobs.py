import os

d='/jukebox/griffiths/bert-brains/'

subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
			'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']

result_prefix="/scratch/sreejank/sl_results/"

data_dir=d+'slumlordreach_data/'

layer_names=['layer_'+str(i)+"_z_representations" for i in range(13)]
layer_dirs=[d+"code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_"+l+".npy" for l in layer_names]
result_dirs=[result_prefix+'slumlordreach/'+l+"/" for l in layer_names] 




for direc in result_dirs:
	if not(os.path.isdir(direc)):
		os.mkdir(direc)




begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 4
#SBATCH --mem-per-cpu 9G  
#SBATCH --time 07:00:00"""



for sub in subs: 
	for i in range(len(layer_dirs)):
		layer=layer_dirs[i]
		results_dir=result_dirs[i]

		#if 'syntactic_complexity' in layer:
		
		
		lname=layer_names[i]
		out_name="#SBATCH --output "+"slurm_outputs/anat_sl_"+str(sub)+"_"+lname+".out"
		job_name="#SBATCH --job-name "+"anat_sl_"+str(sub)+"_"+lname
		with open("searchlight_job.sh","w") as out:
			out.write(begin+"\n")
			out.write(out_name+"\n")
			out.write(job_name+"\n")
			out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u anatomical_searchlight.py "+sub+" "+layer+" "+results_dir+" "+data_dir+" \n")  
			out.close()
		os.system("sbatch searchlight_job.sh")   


