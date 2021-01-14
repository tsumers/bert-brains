import os
import time
d='/jukebox/griffiths/bert-brains/'

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']


#layer_names=['layer_'+str(i)+"_activations" for i in range(13)] 
layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)]   
layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/raw_embeddings/'
layer_dirs=[layer_prefix+layer+".npy" for layer in layer_names] 
data_dir=d+"21styear_data/"   

layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/syntactic_analyses/'
layer_names.append('21st_year_bert-base-uncased_semantic_composition')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_semantic_composition_max_l2.npy")
layer_names.append('21st_year_bert-base-uncased_syntactic_complexity')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_complexity_L-inf-T20.npy")
layer_names.append('21st_year_bert-base-uncased_syntactic_distance') 
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_distance.npy")   



size=52422  
step=10000 
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
	numbers.append(size) 

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
			if 'semantic_composition' in layer_name:
				begin_trim="16"
			elif 'syntactic' in layer_name:
				begin_trim="17"
			else:
				begin_trim="15" 
			end_trim="2240" 
			j=0 
			while j<len(numbers)-1:  
				r1=numbers[j]
				r2=numbers[j+1]
				out_name="/scratch/sreejank/enc_"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy" 
				
				f.write("python encoding_model.py "+sub+" "+layer_dir+" "+layer_name+" "+data_dir+" "+begin_trim+" "+end_trim+" "+str(r1)+" "+str(r2)+"\n") 
				j+=1
				#time.sleep(2) 
	f.close()
