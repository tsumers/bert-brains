import nibabel as nib 
import numpy as np 
import os
import sys

rep=sys.argv[1]

num_parcels=1000 
prefix="/jukebox/griffiths/bert-brains/results/"

full_data_datasets=[]
for dataset in ['black','slumlordreach']:
	if dataset=='black':
		#Black sub
		subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-275', 
		'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-310', 'sub-302', 'sub-312', 
		'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
		'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
		'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-127', 'sub-279', 
		'sub-315', 'sub-314']
		parcellation=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
		affine=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").affine
	else:
		#Slumlordreach subs
		subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
		'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
		parcellation=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
		affine=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").affine

	if 'activations' not in rep: 
		d=prefix+dataset+"/encoding-"+dataset+"_"+rep+"/" 
	else:
		d=prefix+dataset+"/encoding-"+rep+"/"

	full_data=np.zeros((num_parcels,len(subs)))
	for i,sub in enumerate(subs):
		print(i)
		data_sub=nib.load(d+sub+"_parcels_encoding.nii.gz").get_fdata()
		for p in range(num_parcels):
			full_data[p,i]=data_sub[np.where(parcellation==p+1)][0]
	
	for i in range(len(subs)):
		full_data_datasets.append(full_data[:,i].reshape((-1,1)))

full_data=np.concatenate(full_data_datasets,axis=1) 
cutpoint=42 
def bootstrap_pvalue(data):
	shifted=data-np.mean(data)
	sampling=[]
	for _ in range(1000):
		m1=np.mean(np.random.choice(shifted[:42],replace=True,size=42))
		m2=np.mean(np.random.choice(shifted[42:],replace=True,size=17))
		sampling.append(np.mean([m1,m2]))
	sampling=np.asarray(sampling)
	p=np.sum(sampling>=np.mean(data))/len(sampling)
	print(p)
	return p

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig] 


p_values=[]
for p in range(num_parcels):
	print(p)
	p_values.append(bootstrap_pvalue(full_data[p,:]))
p_values=p_adjust_bh(p_values)
p_value_volume=np.zeros(parcellation.shape)
for p in range(num_parcels):
	p_value_volume[np.where(parcellation==p+1)]=1.0-p_values[p] 
p_value_nii=nib.Nifti1Image(p_value_volume,affine)
nib.save(p_value_nii,d+"combined_bootstrap_pvalue_parcellation.nii.gz") 


######## Run individual pvalues

def bootstrap_pvalue_individual(data): 
	shifted=data-np.mean(data)
	sampling=[]
	for _ in range(1000):
		sampling.append(np.mean(np.random.choice(shifted,replace=True,size=len(shifted))))
	sampling=np.asarray(sampling)
	p=np.sum(sampling>=np.mean(data))/len(sampling)
	print(p)
	return p




for dataset in ['black','slumlordreach']:
	if dataset=='black':
		#Black subs
		subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-275', 
		'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-310', 'sub-302', 'sub-312', 
		'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
		'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
		'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-127', 'sub-279', 
		'sub-315', 'sub-314']
		parcellation=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
		affine=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").affine
	else:
		#Slumlordreach subs
		subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
		'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
		parcellation=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
		affine=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").affine

	if 'activations' not in rep: 
		d=prefix+dataset+"/encoding-"+dataset+"_"+rep+"/"
	else:
		d=prefix+dataset+"/encoding-"+rep+"/"
	
	full_data=np.zeros((num_parcels,len(subs)))
	for i,sub in enumerate(subs):
		print(i)
		data_sub=nib.load(d+sub+"_parcels_encoding.nii.gz").get_fdata()
		for p in range(num_parcels):
			full_data[p,i]=data_sub[np.where(parcellation==p+1)][0]
	
	p_values=[]
	for p in range(num_parcels):
		print(p)
		p_values.append(bootstrap_pvalue_individual(full_data[p,:]))
	p_values=p_adjust_bh(p_values)
	p_value_volume=np.zeros(parcellation.shape)
	for p in range(num_parcels):
		p_value_volume[np.where(parcellation==p+1)]=1.0-p_values[p] 
	p_value_nii=nib.Nifti1Image(p_value_volume,affine)
	nib.save(p_value_nii,d+"individual_bootstrap_pvalue_parcellation.nii.gz")  
	








