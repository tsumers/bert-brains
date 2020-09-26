import numpy as np 
import nibabel as nib 


d='/jukebox/griffiths/bert-brains/'

mask=nib.load(d+"21styear_data/whole_brain_mask.nii.gz").get_fdata().astype('bool')


subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']


layers=['layer_'+str(i) for i in range(13)]
layer_names=[layer+"_activations" for layer in layers]+[layer+"_attention" for layer in layers[:-1]]

results_prefix=d+'results/21st_year/'
results_dirs=[results_prefix+"rsa-activations_"+layer for layer in layers]+[results_prefix+"rsa-attentions_"+layer for layer in layers[:-1]]

assert len(results_dirs),len(layer_names)

size=52422
step=5000
numbers=list(range(0,size+1,step))
if numbers[-1]!=size:
        numbers.append(size)


for sub in subs:
	for i in range(len(results_dirs)):
		print(sub,i)
		idx=0
		aggregate=[]
		layer_name=layer_names[i]
		result_dir=results_dirs[i]+"/"
		while idx<len(numbers)-1:
			r1=numbers[idx]
			r2=numbers[idx+1]
			output_name="/scratch/sreejank/"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy"
			aggregate.append(np.load(output_name))
			idx+=1
		sl_voxels=np.concatenate(aggregate)
		volume=np.zeros(mask.shape)
		volume[mask]=sl_voxels
		affine=nib.load(result_dir+sub+"_whole_brain_anatomical_SL.nii.gz").affine 
		nii=nib.Nifti1Image(volume,affine)
		nib.save(nii,result_dir+sub+"_whole_brain_functional_SL.nii.gz")
		
