import os 

begin="""#!/usr/bin/env bash
# Input python command to be submitted as a job 

#SBATCH -p all
#SBATCH --time 01:00:00"""


for dataset in ['slumlordreach','black']:
    if dataset=='slumlordreach':
        subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
        'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
    else:
        subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
                'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
                'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
                'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
                    'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
                    'sub-315', 'sub-314']


    

    for sub in subs:
        out_name="#SBATCH --output "+"slurm_outputs/save_"+str(sub)+".out"
        job_name="#SBATCH --job-name "+"save_parcels_"+str(sub)
        with open("parc_job.sh","w") as out:
            out.write(begin+"\n")
            out.write(out_name+"\n")
            out.write(job_name+"\n")
            out.write("python -u save_parcel_data.py "+dataset+" "+sub) 
            out.close() 
        os.system('sbatch parc_job.sh')

