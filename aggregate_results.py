import numpy as np 
import nibabel as nib  
import os 

d='/jukebox/griffiths/bert-brains/'

data_dir=d+'slumlordreach_data/'   
mask=nib.load(data_dir+"/whole_brain_mask.nii.gz").get_fdata().astype('bool')

subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
			'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']




results_prefix='/scratch/sreejank/sl_results/slumlordreach/'
size=52422
step=11000
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
        numbers.append(size)
#layer_names=['layer_'+str(i)+"_activations" for i in range(13)]

layer_names=['layer_'+str(i)+"_activations" for i in range(13)]


for sub in subs:
	#for i in range(len(results_dirs)):

	#print(sub,i)
	#layer_name=layer_names[i]
	#result_dir=results_dirs[i]+"/"
	for layer_name in layer_names: 
		idx=0
		aggregate=[]
		print(sub)
		result_dir=results_prefix+"encoding-"+layer_name+"/" 
		if not(os.path.isdir(result_dir)):
			os.mkdir(result_dir)
		while idx<len(numbers)-1:
			r1=numbers[idx]
			r2=numbers[idx+1]
			output_name="/scratch/sreejank/func_buffer/"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy"
			aggregate.append(np.load(output_name))
			idx+=1
		sl_voxels=np.concatenate(aggregate)
		volume=np.zeros(mask.shape)
		volume[mask]=sl_voxels
		affine=nib.load(data_dir+sub+".nii.gz").affine 
		nii=nib.Nifti1Image(volume,affine)
		nib.save(nii,result_dir+sub+"_whole_brain_functional_SL.nii.gz")   
			
