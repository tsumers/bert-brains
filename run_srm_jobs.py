import os

d='/jukebox/griffiths/bert-brains/'

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 200G  
#SBATCH --time 04:00:00"""



for features in list(range(290,500,10)): 
    out_name="#SBATCH --output "+"slurm_outputs/srm_"+str(features)+"_"+".out"
    job_name="#SBATCH --job-name "+"srm_"+str(features)
    with open("srm_job.sh","w") as out:
        out.write(begin+"\n")
        out.write(out_name+"\n")
        out.write(job_name+"\n")
        out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u functional_alignment.py "+str(features)+"\n") 
        out.close()
    os.system("sbatch srm_job.sh")   


