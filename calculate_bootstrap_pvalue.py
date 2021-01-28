import nibabel as nib 
import numpy as np 
import os
import sys

d=sys.argv[1]


#Black subs
subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
 'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
 'sub-315', 'sub-314']
 
#Slumlordreach subs
#subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
#'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
#21st year subs
"""
subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']
"""


parcellation=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
affine=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").affine


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


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig] 

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
p_values=p_adjust_bh(p_values)
p_value_volume=np.zeros(parcellation.shape)
for p in range(num_parcels):
	p_value_volume[np.where(parcellation==p+1)]=1.0-p_values[p] 
p_value_nii=nib.Nifti1Image(p_value_volume,affine)
nib.save(p_value_nii,d+"bootstrap_pvalue_parcellation.nii.gz") 










