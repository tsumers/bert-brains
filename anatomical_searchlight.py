# Run a whole brain searchlight

# Import libraries
import nibabel as nib
import numpy as np
from mpi4py import MPI
from brainiak.searchlight.searchlight import Searchlight
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
import sys
from scipy.stats import zscore 


# What subject are you running

sub = sys.argv[1]
layer_dir = sys.argv[2]
results_dir = sys.argv[3]
data_dir = sys.argv[4]


output_name = (results_dir+'/%s_whole_brain_anatomical_SL.nii.gz' % (sub))
# output_mask_name=(d+'results/'+results_dir_name+'/%s_whole_brain_anatomical_SL_mask.nii.gz' % (sub))

# Get information
print("Loading data")
nii = nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here


big_mask = nib.load(data_dir+"whole_brain_mask.nii.gz").get_fdata()
# big_mask=np.zeros(big_mask.shape)
# big_mask[40,30,40]=1

load_features=np.load(layer_dir,allow_pickle=True)
raw_features=[]
for i in range(load_features.shape[0]):
    if load_features[i] is not None and len(load_features[i])>0:
        raw_features.append(load_features[i])
raw_features=np.vstack(raw_features)

if 'black' in data_dir:
    load_data=nii.get_fdata()[:,:,:,8:-8]
    begin_delay=534-raw_features.shape[0]
    raw_data=load_data[:,:,:,begin_delay:]
    features=raw_features[10:-10,:]
    raw_data=raw_data[:,:,:,10:-10]
    trailing=features.shape[0]-raw_data.shape[3]
    features=features[:-trailing]
    assert raw_data.shape[3]==features.shape[0]
elif 'slumlordreach' in data_dir:
    begin_delay=3+(1192-raw_features.shape[0])
    splice1=619-begin_delay
    splice2=644-begin_delay 
    load_data=nii.get_fdata()[:,:,:,begin_delay:1205]  
    load_data=np.concatenate([zscore(load_data[:,:,:,:splice1],axis=3,ddof=1),zscore(load_data[:,:,:,splice2:],axis=3,ddof=1)],axis=3)
    load_data[np.isnan(load_data)]=0.0
    features=np.concatenate([zscore(raw_features[:splice1,:],axis=0,ddof=1),zscore(raw_features[splice2:,:],axis=0,ddof=1)],axis=0) 
    features[np.isnan(features)]=0.0
    features=features[10:-10,:]
    raw_data=load_data[:,:,:,10:-10]
    trailing=raw_data.shape[3]-features.shape[0]
    raw_data=raw_data[:,:,:,:-trailing] 
    print(raw_data.shape)

    raw_data=raw_data[:,:,:,601:]
    features=features[601:,:]
    assert raw_data.shape[3]==features.shape[0]

data=raw_data 
raw_rsm=np.corrcoef(features) 
raw_rsm[np.isnan(raw_rsm)]=0.0
bcvar=raw_rsm[np.triu(np.ones(raw_rsm.shape),k=3).astype('bool')]  



sl_rad = 3
max_blk_edge = 5
pool_size = 1 


# Pull out the MPI information
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print("Preparing searchlight")
# Create the searchlight object
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)

# Distribute the information to the searchlights (preparing it to run)
sl.distribute([data], big_mask)



# Broadcast variables
sl.broadcast(bcvar)

# Kernel function for searchlight RSA
def rsa(data,mask,myrad,bcvar): 
    if np.sum(mask)<2:
        return 0.0

    data4D=data[0]
    mask=mask.astype('bool')
    bolddata_sl=data4D[mask,:].T

    
    human=np.corrcoef(bolddata_sl[:,:]) 
    vec=human[np.triu(np.ones(human.shape),k=3).astype('bool')]
    vec[np.isnan(vec)]=0
    # print(bolddata_sl.shape,bcvar.shape)
    return pearsonr(vec,bcvar)[0]   



print("Running Searchlight") 
sl_result = sl.run_searchlight(rsa, pool_size=pool_size)
# print("End SearchLight")

# Only save the data if this is the first core
if rank == 0:  

    # Convert the output into what can be used
    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0  # If there are nans we want this

    # Save the volume
    sl_nii = nib.Nifti1Image(sl_result, affine_mat)
    nib.save(sl_nii, output_name)  # Save
    # np.save(output_mask_name,big_mask) 
    print('Finished searchlight')  

