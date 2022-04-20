import os
import time
d='/jukebox/griffiths/bert-brains/'


with open("joblist.txt","w") as f:
	for dataset in ['slumlordreach']:

		#Slumlordreach 
		if dataset=='slumlordreach': 

			subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
			'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
			data_dir=d+"slumlordreach_data/"
			save_dirs=[d+"results/slumlordreach/encoding_partition_z/",d+"results/slumlordreach/encoding_partition/"]



		#Black 
		else:
			subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
			'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
			'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
			'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
			 'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
			 'sub-315', 'sub-314']

			data_dir=d+"black_data/"
			save_dirs=[d+"results/black/encoding_partition_z/",d+"results/black/encoding_partition/"] 

 
		for save_dir in save_dirs:
			if not os.path.isdir(save_dir):
				os.mkdir(save_dir)


		begin="""#!/usr/bin/env bash 
		# Input python command to be submitted as a job

		#SBATCH -p all
		#SBATCH --mem-per-cpu 9G
		#SBATCH --time 11:00:00 
		"""


		for sub in subs: 
			for layer_or_z in [0,1]:
				fname=save_dirs[layer_or_z]+sub+"_parcelwise_results.npy"
				#if not(os.path.exists(fname)):
				f.write("python encoding_variance_partitioning.py "+sub+" "+str(layer_or_z)+" "+data_dir+" "+save_dirs[layer_or_z]+"\n")
	f.close()     

