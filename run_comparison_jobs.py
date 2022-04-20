import os 

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 06:00:00"""

model='bert-base-uncased'


d1s=['syntactic_complexity_L-inf_T-128_D-concat' for _ in range(4)]
d2s=['syntactic_complexity_L-inf_T-20_D-concat','syntactic_complexity_L-inf_T-128_D-fwd','syntactic_complexity_L-inf_T-128_D-bck','ling_features']
names=['windowsize','fwdcompare','bckcompare','attention_linguistics']

assert len(d1s)==len(d2s) 
assert len(names)==len(d1s)
"""
prefix="/jukebox/griffiths/bert-brains/results/"
for dataset in ['black','slumlordreach']:
	for d in d1s:
		if 'activations' not in d: 
			fname=prefix+dataset+"/encoding-"+dataset+"_"+model+"_"+d+"/"
		else:
			fname=prefix+dataset+"/encoding-"+d+"/"
		print(fname,os.path.isdir(fname))
	
	for d in d2s:
		if 'activations' not in d: 
			fname=prefix+dataset+"/encoding-"+dataset+"_"+model+"_"+d+"/"
		else:
			fname=prefix+dataset+"/encoding-"+d+"/"
		print(fname,os.path.isdir(fname))
"""
print("Begin")

for i in range(len(d1s)):
	d1=d1s[i]
	d2=d2s[i]
	name=names[i] 
	out_name="#SBATCH --output "+"slurm_outputs/diff_"+str(i)+".out"
	job_name="#SBATCH --job-name "+"permutation_"+str(i)

	with open("perm_job.sh","w") as out:
		out.write(begin+"\n")
		out.write(out_name+"\n")
		out.write(job_name+"\n")
		out.write("python -u comparison_maps.py "+d1+" "+d2+" "+name+ " "+model+"\n")
		out.close()
	os.system('sbatch perm_job.sh')


