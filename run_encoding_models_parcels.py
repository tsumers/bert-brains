import os
import time
d='/jukebox/griffiths/bert-brains/'


with open("joblist.txt","w") as f:
	for dataset in ['slumlordreach','black']:

		#Slumlordreach 
		if dataset=='slumlordreach': 

			subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
			'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']

			
			#Z reps
			layer_names=['layer_'+str(i)+"_z_representations" for i in range(0,12)]
			layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/'
			save_prefix=d+"results/slumlordreach/"
			layer_dirs=[layer_prefix+"slumlordreach_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
			save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]
			data_dir=d+"slumlordreach_data/"   


			layer_names2=['layer_'+str(i)+"_activations" for i in range(0,13)]   
			layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/'
			save_prefix=d+"results/slumlordreach/"
			layer_dirs+=[layer_prefix+"slumlordreach_bert-base-uncased_"+layer+".npy" for layer in layer_names2] 
			save_dirs+=[save_prefix+"encoding-"+layer+"/" for layer in layer_names2]
			layer_names+=layer_names2 
			data_dir=d+"slumlordreach_data/"   

			data_dir=d+"slumlordreach_data/"   
			#layer_names=[]
			#layer_dirs=[]
			#save_dirs=[] 
			save_prefix=d+"results/slumlordreach/"
			layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/syntactic_analyses/'
			for fname in os.listdir(layer_prefix):
				if 'syntactic_complexity_L-inf_T-128_D-concat' in fname:
					layer_names.append(fname[:-4]) 
					layer_dirs.append(layer_prefix+fname)
					save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/") 
			



		#Black 
		else:
			subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
			'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
			'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
			'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
			 'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
			 'sub-315', 'sub-314']

			#Z reps

			layer_names=['layer_'+str(i)+"_z_representations" for i in range(0,12)]
			layer_prefix=d+'code/bert-brains/data/black/bert-base-uncased/raw_embeddings/'
			save_prefix=d+"results/black/"
			layer_dirs=[layer_prefix+"black_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
			save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]

			data_dir=d+"black_data/"     
			#layer_names=[]
			#layer_dirs=[]
			#save_dirs=[]


			layer_names2=['layer_'+str(i)+"_activations" for i in range(0,13)] 
			layer_prefix=d+'code/bert-brains/data/black/bert-base-uncased/raw_embeddings/'
			save_prefix=d+"results/black/"
			layer_dirs+=[layer_prefix+"black_bert-base-uncased_"+layer+".npy" for layer in layer_names2] 
			save_dirs+=[save_prefix+"encoding-"+layer+"/" for layer in layer_names2]

			layer_names+=layer_names2 

			#layer_names=[]
			#save_dirs=[]
			#layer_dirs=[]
			save_prefix=d+"results/black/"
			data_dir=d+"black_data/"    
			layer_prefix=d+'code/bert-brains/data/black/bert-base-uncased/syntactic_analyses/'
			for fname in os.listdir(layer_prefix):
				if 'syntactic_complexity_L-inf_T-128_D-concat' in fname:
					layer_names.append(fname[:-4]) 
					layer_dirs.append(layer_prefix+fname)
					save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/") 

		print(save_dirs)  

		for save_dir in save_dirs:
			if not os.path.isdir(save_dir):
				os.mkdir(save_dir)

		begin="""#!/usr/bin/env bash 
		# Input python command to be submitted as a job

		#SBATCH -p all
		#SBATCH --mem-per-cpu 9G
		#SBATCH --time 6:00:00 
		"""


		for sub in subs: 
			for i in range(len(layer_names)):
			#for i in range(len(layer_names)):
				#layer_name=layer_names[i]
				#layer_dir=layer_dirs[i]
				#if 'attention' in layer_name:
				#	begin_trim="16"
				#	end_trim="2240"
				#else:
				#	begin_trim="15"
				#	end_trim="2240" 

				layer_name=layer_names[i]
				layer_dir=layer_dirs[i]
				save_dir=save_dirs[i]
				if 'semantic_composition' in layer_name:
					begin_trim="16"
				elif 'syntactic' in layer_name:
					begin_trim="17"
				else: 
					begin_trim="15" 
				end_trim="2240" 
				f.write("python encoding_model_parcels.py "+sub+" "+layer_dir+" "+layer_name+" "+data_dir+" "+begin_trim+" "+end_trim+" "+save_dir+"\n")
	f.close()     
