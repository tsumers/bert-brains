import numpy as np 
import nibabel as nib 
import os
import sys 

rep=sys.argv[1]

d="/jukebox/griffiths/bert-brains/"
#mask=nib.load(d+"21styear_data/whole_brain_mask.nii.gz").get_fdata().astype('bool')
parce1="/jukebox/griffiths/bert-brains/black_data/Schaefer1000_3mm.nii.gz"
parce2="/jukebox/griffiths/bert-brains/slumlordreach_data/Schaefer1000_3mm.nii.gz"
num_parcels=1000
def get_result(rep,threshold=0.95):
    full_results=[]
    prefix="/jukebox/griffiths/bert-brains/results/"
    for dataset in ['black','slumlordreach']:
        print(dataset)
        if dataset=='black':
            #Black subs
            subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
            'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
            'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
            'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
            'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
            'sub-315', 'sub-314']
            parcellation=nib.load(parce1).get_fdata().astype('int')
            affine=nib.load(parce1).affine
        else:
            #Slumlordreach subs
            subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
            'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
            parcellation=nib.load(parce2).get_fdata().astype('int')
            affine=nib.load(parce2).affine
        
        if 'activations' not in rep: 
            d=prefix+dataset+"/encoding-"+dataset+"_"+rep+"/"
        else:
            d=prefix+dataset+"/encoding-"+rep+"/"

        full_data=np.zeros((num_parcels,len(subs)))
        for i,sub in enumerate(subs):
            print(sub)
            data_sub=nib.load(d+sub+"_parcels_encoding.nii.gz").get_fdata()
            for p in range(num_parcels):
                full_data[p,i]=data_sub[np.where(parcellation==p+1)][0]
        for i in range(len(subs)):
            full_results.append(full_data[:,i].reshape((-1,1)))
    
    result_array=np.mean(full_results,axis=1)
    result_dir=d 
    result_volume=np.zeros(parcellation.shape)
    for i,value in enumerate(result_array):
        result_volume[np.where(parcellation==i+1)]=value 
    nib.save(nib.Nifti1Image(result_volume,affine),result_dir+"two_story_combined_mean.nii.gz")

get_result(rep)