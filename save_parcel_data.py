import numpy as np 
import nibabel as nib 
import sys 
from scipy.stats import zscore 

d='/jukebox/griffiths/bert-brains/'

dataset=sys.argv[1]
sub=sys.argv[2]


data_dir=d+dataset+'_data/'  
parcellation_nii=nib.load(data_dir+"Schaefer1000_3mm.nii.gz")
affine=parcellation_nii.affine
parcellation=parcellation_nii.get_fdata().astype('int') 

phoneme_counts=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/'+dataset+"/"+dataset+"_phoneme_counts.npy").reshape((-1,1))
silent=((phoneme_counts[:,0]==0).astype('int')).reshape((-1,1))

layer_dir=d+'code/bert-brains/data/'+dataset+'/bert-base-uncased/raw_embeddings/'+dataset+'_bert-base-uncased_layer_5_activations.npy'
load_features=np.load(layer_dir,allow_pickle=True)
raw_features=[]
raw_silent_features=[]
for i in range(load_features.shape[0]):
       if load_features[i] is not None and len(load_features[i])>0:
              raw_features.append(load_features[i])
              raw_silent_features.append(silent[i])
raw_features=np.vstack(raw_features) 
raw_silent_features=np.vstack(raw_silent_features)



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
    raw_silent_features=raw_silent_features[:splice1]
    features=features[10:-10,:]
    raw_silent_features=raw_silent_features[10:-10]
    raw_data=load_data[:,:,:,10:-10]
    trailing=raw_data.shape[3]-features.shape[0]
    if trailing>0:
        raw_data=raw_data[:,:,:,:-trailing]
else:
    begin_delay=534-raw_features.shape[0]
    load_data=nii.get_fdata()[:,:,:,8:-8]
    raw_data=load_data[:,:,:,begin_delay:]
    features=raw_features[10:-10,:]
    raw_silent_features=raw_silent_features[10:-10,:]
    raw_data=raw_data[:,:,:,10:-10]
    trailing=features.shape[0]-raw_data.shape[3]
    if trailing>0:
        features=features[:-trailing]
        raw_silent_features=raw_silent_features[:-trailing] 
assert raw_data.shape[3]==features.shape[0]
assert raw_silent_features.shape[0]==features.shape[0]

"""
num_parcels=1000
data=np.zeros((num_parcels,raw_data.shape[-1]))
for p in range(num_parcels):
    data[p,:]=raw_data[np.where(parcellation==p+1)].mean(axis=0) 
np.save(data_dir+sub+"_parcelwise_data.npy",data)
"""

np.save(data_dir+"silent_vector.npy",raw_silent_features) 



