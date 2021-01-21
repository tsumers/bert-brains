import os
import time
d='/jukebox/griffiths/bert-brains/'

"""
subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']


#layer_names=['layer_'+str(i)+"_activations" for i in range(13)] 
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)]   
layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/raw_embeddings/'
save_prefix=d+"results/21st_year/"
layer_dirs=[layer_prefix+layer+".npy" for layer in layer_names] 
save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]
data_dir=d+"21styear_data/"    

layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/syntactic_analyses/'

layer_names.append('21st_year_bert-base-uncased_semantic_composition')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_semantic_composition_max_l2.npy")
save_dirs.append(save_prefix+"encoding-semantic_composition_max_l2/")

layer_names.append('21st_year_bert-base-uncased_syntactic_complexity')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_complexity_L-inf-T20.npy")
save_dirs.append(save_prefix+"encoding-syntactic_complexity_L-inf/")

layer_names.append('21st_year_bert-base-uncased_syntactic_distance') 
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_distance.npy")   
save_dirs.append(save_prefix+"encoding-syntactic_distance/")
"""

subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)]   
layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/'
save_prefix=d+"results/slumlordreach/"
layer_dirs=[layer_prefix+"slumlordreach_bert-base-uncased_"+layer+".npy" for layer in layer_names] 
save_dirs=[save_prefix+"encoding-"+layer+"/" for layer in layer_names]
data_dir=d+"slumlordreach_data/" 

layer_prefix=d+'code/bert-brains/data/slumlordreach/bert-base-uncased/syntactic_analyses/'
for fname in os.listdir(layer_prefix):
	layer_names.append(fname[:-4])
	layer_dirs.append(layer_prefix+fname)
	save_dirs.append(save_prefix+'encoding-'+fname[:-4]+"/")


for save_dir in save_dirs:
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

begin="""#!/usr/bin/env bash 
# Input python command to be submitted as a job

#SBATCH -p all
#SBATCH --mem-per-cpu 9G
#SBATCH --time 6:00:00 
"""

with open("joblist.txt","w") as f:



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
