import os 


begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 05:00:00"""

idxs=list(range(10))

for gpt_large in [0]: 
	if gpt_large==0: 
		for bert_or_gpt in [1]:
			for i in range(len(idxs)):
				out_name="#SBATCH --output "+"slurm_outputs/plot_"+"_"+str(bert_or_gpt)+".out"
				job_name="#SBATCH --job-name "+"plot_"+"_"+str(bert_or_gpt)
				with open("plot_job.sh","w") as out:
					out.write(begin+"\n")
					out.write(out_name+"\n")
					out.write(job_name+"\n")
					out.write("python -u save_full_layer_plot.py "+" "+str(bert_or_gpt)+" "+str(gpt_large)+" "+str(i)+"\n") 
					out.close()
				os.system('sbatch plot_job.sh')  
	else:
		layer_or_z=0
		bert_or_gpt=0
		out_name="#SBATCH --output "+"slurm_outputs/plot_"+str(layer_or_z)+"_"+str(bert_or_gpt)+".out"
		job_name="#SBATCH --job-name "+"plot_"+str(layer_or_z)+"_"+str(bert_or_gpt)
		with open("plot_job.sh","w") as out:
			out.write(begin+"\n")
			out.write(out_name+"\n")
			out.write(job_name+"\n")
			out.write("python -u save_full_layer_plot.py "+str(layer_or_z)+" "+str(bert_or_gpt)+" "+str(gpt_large)+"\n") 
			out.close()
		os.system('sbatch plot_job.sh')


