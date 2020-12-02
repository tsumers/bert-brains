import numpy as np 
import nibabel as nib 
import os 

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']

layer_names=['layer_'+str(i)+"_activations" for i in range(13)]
layer_names.append('semantic_composition')
layer_names.append('syntactic_complexity_L-inf')
layer_names.append('syntactic_distance')

d='/jukebox/griffiths/bert-brains/'
results_prefix=d+'results/21st_year/'
result_dirs=[results_prefix+"encoding-"+layer_name+"/" for layer_name in layer_names]

shape=nib.load("/jukebox/griffiths/bert-brains/21styear_data/whole_brain_mask.nii.gz").shape

for result_dir in result_dirs:
	print(result_dir)
	stacked_volume=np.zeros(shape+(len(subs),))
	for i,sub in enumerate(subs):
		nii=nib.load(result_dir+sub+"_whole_brain_encoding.nii.gz")
		stacked_volume[:,:,:,i]=nii.get_fdata()
	stacked_nii=nib.Nifti1Image(stacked_volume,nii.affine)
	nib.save(stacked_nii,result_dir+"stacked_whole_brain_encoding.nii.gz")
	os.system('randomise -i '+result_dir+"stacked_whole_brain_encoding.nii.gz"+" -o "+result_dir+"OneSampT -1 -T") 