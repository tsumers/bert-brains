from brainiak import image, io
import nibabel as nib
import numpy as np
import brainiak.funcalign.srm

d='/jukebox/griffiths/bert-brains/'

num_features=200 

subs=['sub-075', 'sub-131', 'sub-190', 'sub-201', 'sub-235', 'sub-244',
       'sub-249', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258',
       'sub-259', 'sub-260', 'sub-261', 'sub-262', 'sub-263', 'sub-264',
       'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270',
       'sub-271']

data_dir=d+'21styear_data/'
mask_name=data_dir+"whole_brain_mask.nii.gz"
mask=nib.load(mask_name).get_fdata().astype('bool')

fnames=[data_dir+sub+".nii.gz" for sub in subs]
print("Loading and Masking Data")
train_data=[nib.load(fname).get_fdata()[mask][:,972:] for fname in fnames]
srm=brainiak.funcalign.srm.SRM(n_iter=20,features=num_features)
print("Training SRM")
srm.fit(train_data)
print("Saving Weights")
weights=np.asarray(srm.w_)
np.save(data_dir+"srm_weights.npy",weights)



