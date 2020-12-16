import numpy as np 
import nibabel as nib  
import os 

d='/jukebox/griffiths/bert-brains/'

mask=nib.load(d+"21styear_data/whole_brain_mask.nii.gz").get_fdata().astype('bool')
data_dir=d+"21styear_data/" 

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']




results_prefix=d+'results/21st_year/'
size=52422
step=10000
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
        numbers.append(size)
#layer_names=['layer_'+str(i)+"_activations" for i in range(13)]

layer_names=['layer_'+str(i)+"_activations" for i in range(0,13)] 
layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/raw_embeddings/'
layer_dirs=[layer_prefix+layer+".npy" for layer in layer_names]
data_dir=d+"21styear_data/"

layer_prefix=d+'code/bert-brains/data/21st_year/bert-base-uncased/syntactic_analyses/'
layer_names.append('21st_year_bert-base-uncased_semantic_composition')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_semantic_composition_max_l2.npy")
layer_names.append('21st_year_bert-base-uncased_syntactic_complexity')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_complexity_L-inf.npy")
layer_names.append('21st_year_bert-base-uncased_syntactic_distance')
layer_dirs.append(layer_prefix+"21st_year_bert-base-uncased_syntactic_distance.npy") 


for sub in subs:
	#for i in range(len(results_dirs)):

	#print(sub,i)
	#layer_name=layer_names[i]
	#result_dir=results_dirs[i]+"/"
	for layer_name in layer_names: 
		idx=0
		aggregate=[]
		print(sub)
		#layer_name="semantic_composition"
		result_dir=results_prefix+"encoding-"+layer_name+"/" 
		if not(os.path.isdir(result_dir)):
			os.mkdir(result_dir)
		while idx<len(numbers)-1:
			r1=numbers[idx]
			r2=numbers[idx+1]
			output_name="/scratch/sreejank/enc_"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy"
			aggregate.append(np.load(output_name))
			idx+=1
		sl_voxels=np.concatenate(aggregate)
		volume=np.zeros(mask.shape)
		volume[mask]=sl_voxels
		affine=nib.load(data_dir+sub+".nii.gz").affine 
		nii=nib.Nifti1Image(volume,affine)
		nib.save(nii,result_dir+sub+"_whole_brain_encoding.nii.gz")  
			
