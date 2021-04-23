import numpy as np 
import nibabel as nib 
import sys 
from scipy.stats import zscore 

d='/jukebox/griffiths/bert-brains/'

dataset=sys.argv[1]
sub=sys.argv[2]

if dataset=='slumlordreach':
    subs=['sub-145', 'sub-143', 'sub-016', 'sub-142', 'sub-141', 'sub-133', 'sub-140', 'sub-136', 
    'sub-084', 'sub-135', 'sub-137', 'sub-138', 'sub-111', 'sub-106', 'sub-134', 'sub-132', 'sub-144']
else:
    subs=['sub-300', 'sub-304', 'sub-293', 'sub-273', 'sub-265', 'sub-307', 'sub-283', 'sub-275', 
            'sub-291', 'sub-297', 'sub-303', 'sub-294', 'sub-286', 'sub-282', 'sub-310', 'sub-302', 'sub-312', 
            'sub-301', 'sub-287', 'sub-298', 'sub-313', 'sub-285', 'sub-292', 'sub-311', 'sub-267', 'sub-295', 
            'sub-305', 'sub-274', 'sub-290', 'sub-288', 'sub-281', 'sub-276', 'sub-277', 'sub-299', 'sub-308',
                'sub-272', 'sub-284', 'sub-289', 'sub-280', 'sub-309', 'sub-306', 'sub-296', 'sub-127', 'sub-279', 
                'sub-315', 'sub-314']

data_dir=d+dataset+'_data/'  
parcellation_nii=nib.load(data_dir+"Schaefer1000_3mm.nii.gz")
affine=parcellation_nii.affine
parcellation=parcellation_nii.get_fdata().astype('int') 


layer_dir=d+'code/bert-brains/data/'+dataset+'/bert-base-uncased/raw_embeddings/'+dataset+'_bert-base-uncased_layer_5_activations.npy'
load_features=np.load(layer_dir,allow_pickle=True)
raw_features=[]
for i in range(load_features.shape[0]):
       if load_features[i] is not None and len(load_features[i])>0:
              raw_features.append(load_features[i])
raw_features=np.vstack(raw_features) 


parcel_data=[]

nii = nib.load(data_dir+sub+".nii.gz")
if 'slumlordreach' in data_dir:
    begin_delay=3+(1192-raw_features.shape[0])
    splice1=619-begin_delay
    splice2=644-begin_delay 
    load_data=nii.get_fdata()[:,:,:,begin_delay:1205]  
    load_data=zscore(load_data[:,:,:,:splice1],axis=3,ddof=1)
    load_data[np.isnan(load_data)]=0.0
    features=zscore(raw_features[:splice1,:],axis=0,ddof=1)
    features[np.isnan(features)]=0.0
    features=features[10:-10,:]
    raw_data=load_data[:,:,:,10:-10]
    trailing=raw_data.shape[3]-features.shape[0]
    if trailing>0:
        raw_data=raw_data[:,:,:,:-trailing]
else:
    begin_delay=534-raw_features.shape[0]
    load_data=nii.get_fdata()[:,:,:,8:-8]
    raw_data=load_data[:,:,:,begin_delay:]
    features=raw_features[10:-10,:]
    raw_data=raw_data[:,:,:,10:-10]
    trailing=features.shape[0]-raw_data.shape[3]
    if trailing>0:
        features=features[:-trailing]
assert raw_data.shape[3]==features.shape[0]

num_parcels=1000
data=np.zeros((num_parcels,raw_data.shape[-1]))
for p in range(num_parcels):
    data[p,:]=raw_data[np.where(parcellation==p+1)].mean(axis=0) 

np.save(data_dir+sub+"_parcelwise_data.npy",data)




