import os 

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00"""

for dataset in ['black','slumlordreach']:
	for model in ['bert-base-uncased','gpt2']:
		d="/jukebox/griffiths/bert-brains/results/"+dataset+"/"
		first=['syntactic_complexity_L-1_T-120','syntactic_distance_T-20','syntactic_complexity_L-1_T-20','semantic_composition_5-TRs']
		second=['syntactic_distance_T-120','syntactic_distance_T-120','syntactic_complexity_L-1_T-120','semantic_composition_20-TRs']

		d1s=[d+'encoding-'+dataset+"_"+model+"_"+x+"/" for x in first]
		d2s=[d+'encoding-'+dataset+"_"+model+"_"+x+"/" for x in second]

		if model=='bert-base-uncased':
			names=[dataset+"_"+x for x in ['bert_complexity_v_distance','bert_T20distance_v_T120distance','bert_l1_T20complexity_v_T120complexity','bert_5TRcomposition_v_20TRcomposition']]
		else: 
			names=[dataset+"_"+x for x in ['gpt_complexity_v_distance','gpt_T20distance_v_T120distance','gpt_l1_T20complexity_v_T120complexity','gpt_5TRcomposition_v_20TRcomposition']]
		for i in range(len(d1s)):
			d1=d1s[i]
			d2=d2s[i]
			name=names[i] 
			out_name="#SBATCH --output "+"slurm_outputs/diff_"+str(i)+"_"+dataset+"_"+model+".out"
			job_name="#SBATCH --job-name "+"permutation_"+str(i)

			with open("perm_job.sh","w") as out:
				out.write(begin+"\n")
				out.write(out_name+"\n")
				out.write(job_name+"\n")
				out.write("python -u comparison_maps.py "+d1+" "+d2+" "+dataset+" "+name+ "\n")
				out.close()
			os.system('sbatch perm_job.sh')


