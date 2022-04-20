import os 


d="/jukebox/griffiths/bert-brains/"

model='bert-base-uncased'

#ds=['layer_'+str(i)+"_activations" for i in range(12)]
#ds+=['layer_'+str(i)+"_z_representations" for i in range(12)]
#ds+=['bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat']
#ds=['ling_features']
#ds=['bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat','bert-base-uncased_syntactic_complexity_L-inf_T-128_D-fwd','bert-base-uncased_syntactic_complexity_L-inf_T-128_D-bck','bert-base-uncased_syntactic_complexity_L-inf_T-20_D-concat']
#ds=['encoding_full_z']
ds=['encoding_onerep']
"""
for fname in os.listdir(d+'code/bert-brains/data/black/bert-base-uncased/syntactic_analyses/'):
	ds.append(fname[6:-4])
"""
prefix="/jukebox/griffiths/bert-brains/results/"
for dataset in ['black','slumlordreach']:
	for d in ds:
		if 'layer' not in d and 'ling' not in d: 
			d1=prefix+dataset+"/encoding-"+dataset+"_"+d+"/" 
		else:
			d1=prefix+dataset+"/encoding-"+d+"/"
		print(os.path.isdir(d1),d1)   
		
print(ds)
begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00"""


for i,rep in enumerate(ds):
	out_name="#SBATCH --output "+"slurm_outputs/p_value_"+str(i)+".out"
	job_name="#SBATCH --job-name "+"bootstrap_"+str(i)
	with open("boot_job.sh","w") as out:
		out.write(begin+"\n")
		out.write(out_name+"\n")
		out.write(job_name+"\n")
		out.write("python -u calculate_bootstrap_pvalue.py "+rep) 
		out.close()
	os.system('sbatch boot_job.sh')


