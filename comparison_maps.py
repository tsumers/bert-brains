import nibabel as nib 
import numpy as np 
import os 
import sys 

prefix="/jukebox/griffiths/bert-brains/results/"

rep1=sys.argv[1]
rep2=sys.argv[2]
#dataset=sys.argv[3]
name=sys.argv[3]
model=sys.argv[4]


full_data1_datasets=[]
full_data2_datasets=[]
for dataset in ['black','slumlordreach']:
    d=rep1
    if 'activations' not in d: 
        d1=prefix+dataset+"/encoding-"+dataset+"_"+model+"_"+d+"/"
    else:
        d1=prefix+dataset+"/encoding-"+d+"/"
    d=rep2 
    if 'activations' not in d: 
        d2=prefix+dataset+"/encoding-"+dataset+"_"+model+"_"+d+"/"
    else:
        d2=prefix+dataset+"/encoding-"+d+"/" 
    if dataset=='black':

        #Black subs
        subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
        'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
        'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
        'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
        'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
        'sub-315', 'sub-314']

        parcellation=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
        affine=nib.load("/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz").affine
    elif dataset=='slumlordreach':
        #Slumlordreach subs
        subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
        'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
        parcellation=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
        affine=nib.load("/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz").affine
    
    num_parcels=1000
    full_data1=np.zeros((num_parcels,len(subs)))
    full_data2=np.zeros((num_parcels,len(subs)))
    for i,sub in enumerate(subs):
        print(i)
        data_sub1=nib.load(d1+sub+"_parcels_encoding.nii.gz").get_fdata()
        data_sub2=nib.load(d2+sub+"_parcels_encoding.nii.gz").get_fdata()
        for p in range(num_parcels):
            full_data1[p,i]=data_sub1[np.where(parcellation==p+1)][0]
            full_data2[p,i]=data_sub2[np.where(parcellation==p+1)][0]
    for i in range(len(subs)):
        full_data1_datasets.append(full_data1[:,i].reshape((-1,1)))
        full_data2_datasets.append(full_data2[:,i].reshape((-1,1)))


full_data1=np.concatenate(full_data1_datasets,axis=1)
full_data2=np.concatenate(full_data2_datasets,axis=1)



def permutation_t_test(data1,data2):
    diff=data1-data2
    stderr=np.std(diff,ddof=1)/np.sqrt(len(diff))+1e-9
    original_t=np.mean(diff)/stderr
    permutations=[]
    flip_diff=diff.copy()
    for _ in range(10000):
        flip_idxs=np.random.choice([0,1],size=len(diff),replace=True).astype('bool')
        flip_diff[flip_idxs]=diff[flip_idxs]*-1
        flip_diff[~flip_idxs]=diff[~flip_idxs]  
        flip_stderr=np.std(flip_diff,ddof=1)/np.sqrt(len(flip_diff))+1e-9 
        permutations.append(np.mean(flip_diff)/flip_stderr)
    permutations=np.asarray(permutations)

    if original_t<0:
        p_value=float(np.sum(permutations<=original_t))/len(permutations)
    else:
        p_value=float(np.sum(permutations>=original_t))/len(permutations)
    return original_t,p_value

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig] 

t_parcels=np.zeros((num_parcels,))
p_parcels=np.zeros((num_parcels,))
for i in range(num_parcels):
    t,p=permutation_t_test(full_data1[i,:],full_data2[i,:])
    t_parcels[i]=t
    p_parcels[i]=p
    print(i,t,p)

p_parcels=p_adjust_bh(p_parcels)

t_volume=np.zeros(parcellation.shape)
p_volume=np.zeros(parcellation.shape)

for i in range(num_parcels):
    t_volume[np.where(parcellation==i+1)]=t_parcels[i]
    p_volume[np.where(parcellation==i+1)]=1.0-p_parcels[i]




prefix="/jukebox/griffiths/bert-brains/results/difference_maps/"
t_nii=nib.Nifti1Image(t_volume,affine)
nib.save(t_nii,prefix+name+"_tvalues.nii.gz")

p_nii=nib.Nifti1Image(p_volume,affine)
nib.save(p_nii,prefix+name+"_pvalues.nii.gz") 