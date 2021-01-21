import nibabel as nib 
import numpy as np 
import os
import sys

d=sys.argv[1]

#Slumlordreach subs
subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
#21st year subs
"""
subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']
"""


parcellation=nib.load("/jukebox/griffiths/bert-brains/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
affine=nib.load("/jukebox/griffiths/bert-brains/Schaefer1000_3mm.nii.gz").affine


def bootstrap_pvalue(data):
	shifted=data-np.mean(data)
	sampling=[]
	for boot in range(1000):
		sampling.append(np.mean(np.random.choice(shifted,replace=True,size=len(shifted))))
	sampling=np.asarray(sampling)
	p=np.sum(sampling>=np.mean(data))/len(sampling)
	print(p)
	return p

num_parcels=1000


print(d)
full_data=np.zeros((num_parcels,len(subs)))
for i,sub in enumerate(subs):
	print(i)
	data_sub=nib.load(d+sub+"_parcels_encoding.nii.gz").get_fdata()
	for p in range(num_parcels):
		full_data[p,i]=data_sub[np.where(parcellation==p+1)][0]

p_values=[]
for p in range(num_parcels):
	print(p)
	p_values.append(bootstrap_pvalue(full_data[p,:]))
p_value_volume=np.zeros(parcellation.shape)
for p in range(num_parcels):
	p_value_volume[np.where(parcellation==p+1)]=1.0-p_values[p]
p_value_nii=nib.Nifti1Image(p_value_volume,affine)
nib.save(p_value_nii,d+"bootstrap_pvalue_parcellation.nii.gz") 










