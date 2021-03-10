import os 

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 06:00:00"""

model='bert-base-uncased'



if model=='bert-base-uncased':
	d1s=['layer_'+str(i)+"_activations" for i in range(12)]
	d2s=['layer_'+str(i+1)+"_activations" for i in range(12)]
else:
	d1s=['gpt_layer_'+str(i)+"_activations" for i in range(12)]
	d2s=['gpt_layer_'+str(i+1)+"_activations" for i in range(12)]


d1s+=['semantic_composition_0-TRs','semantic_composition_1-TRs','semantic_composition_2-TRs','semantic_composition_3-TRs','semantic_composition_4-TRs','semantic_composition_5-TRs','semantic_composition_10-TRs']
d2s+=['semantic_composition_1-TRs','semantic_composition_2-TRs','semantic_composition_3-TRs','semantic_composition_4-TRs','semantic_composition_5-TRs','semantic_composition_10-TRs','semantic_composition_20-TRs']
d1s+=['syntactic_complexity_L-inf_T-128_D-concat','syntactic_distance_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-concat','syntactic_distance_T-128_D-concat','syntactic_distance_T-128_D-concat']
d2s+=['syntactic_complexity_L-inf_T-10_D-concat','syntactic_distance_T-20_D-concat','syntactic_distance_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-concat','syntactic_complexity_L-inf_T-128_D-fwd','syntactic_complexity_L-inf_T-128_D-bck','syntactic_distance_T-128_D-bck','syntactic_distance_T-128_D-fwd']

if model=='bert-base-uncased':
	names=['BERT_Semantic_1_'+str(i) for i in range(12)]
	names+=['BERT_Semantic_2_'+str(i) for i in range(7)]
	names+=['BERT_'+x for x in ['Distance_1','Distance_2','Distance_3','Distance_4','Direction_1','Direction_2','Direction_3','Direction_4']]
else:
	names=['GPT2_Semantic_1_'+str(i) for i in range(12)]
	names+=['GPT2_Semantic_2_'+str(i) for i in range(7)]
	names=['GPT2_'+x for x in ['Semantic_2','Distance_1','Distance_2','Distance_3','Distance_4','Direction_1','Direction_2','Direction_3','Direction_4']]


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


