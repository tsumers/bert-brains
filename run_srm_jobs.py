import os

d='/jukebox/griffiths/bert-brains/'

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 9G  
#SBATCH --time 01:00:00"""

subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']


for sub in subs: 
    out_name="#SBATCH --output "+"slurm_outputs/latent_"+str(sub)+"_"+".out"
    job_name="#SBATCH --job-name "+"latent_"+str(sub)
    with open("srm_job.sh","w") as out:
        out.write(begin+"\n")
        out.write(out_name+"\n")
        out.write(job_name+"\n")
        out.write("srun -n $SLURM_NTASKS --mpi=pmi2 python -u functional_alignment.py "+str(sub)+"\n") 
        out.close()
    os.system("sbatch srm_job.sh")   


