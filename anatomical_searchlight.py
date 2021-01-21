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


# What subject are you running

sub = sys.argv[1]
layer_dir = sys.argv[2]
results_dir = sys.argv[3]
data_dir = sys.argv[4]
begin_trim = int(sys.argv[5])
end_trim = int(sys.argv[6])
kernel = sys.argv[7]


output_name = (results_dir+'/%s_whole_brain_anatomical_SL.nii.gz' % (sub))
# output_mask_name=(d+'results/'+results_dir_name+'/%s_whole_brain_anatomical_SL_mask.nii.gz' % (sub))

# Get information
print("Loading data")
nii = nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here


# Preset the variables

data = nii.get_fdata()[:, :, :, begin_trim:end_trim]
if kernel == 'rsa':
    data = data[:, :, :, 960:]
print("Data",data.shape)
big_mask = nib.load(data_dir+"whole_brain_mask.nii.gz").get_fdata()
# big_mask=np.zeros(big_mask.shape)
# big_mask[40,30,40]=1
""" Decoding
if 'syntactic_complexity' in layer_dir:
    bcvar=np.load(layer_dir)[2:,:]
else:
    bcvar=np.load(layer_dir)[:,:]
"""


if kernel == 'rsa':
    if 'syntactic_complexity' in layer_dir or 'syntactic_distance' in layer_dir:
        if 'gpt' in layer_dir:
            raw_features = np.load(layer_dir)[3:, :]
        else:
            raw_features=np.load(layer_dir)[2:,:]
    else:
        raw_features = np.load(layer_dir)[:, :]
    print("Feats",raw_features[960:,:].shape)
    raw_rsm=np.corrcoef(raw_features[960:,:])
    raw_rsm[np.isnan(raw_rsm)]=0.0
    bcvar=raw_rsm[np.triu(np.ones(raw_rsm.shape),k=3).astype('bool')]  
else:
    # Randomly subsample dominant class to balance classes
    labels=np.load(layer_dir)
    dominant=int(np.sum(labels==1)>np.sum(labels==0))
    mask=np.zeros(labels.shape)
    mask[np.where(labels!=dominant)]=1
    mask[np.random.choice(np.where(labels==dominant)[0],replace=False,size=np.sum(labels!=dominant))]=1
    bcvar=np.vstack([mask,labels])  




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

# Kernel function for searchlight decoding
def decode(data,mask,myrad,bcvar):
    if np.sum(mask)<2:
        return 0.0
    
    data4D=data[0]
    mask=mask.astype('bool') 
    bolddata_sl=data4D[mask,:].T

    X_raw=bolddata_sl
    idx_mask=bcvar[0].astype('bool')
    y_raw=bcvar[1]

    X=X_raw[idx_mask,:]
    y=y_raw[idx_mask]

    
    model=SVC()
    skf=KFold(n_splits=3)
    scores=[]
    for train_index,test_index in skf.split(X,y):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train,y_train)
        scores.append(accuracy_score(y_test,model.predict(X_test)))
    return np.mean(scores)  



print("Running Searchlight") 
if kernel=='rsa':
    sl_result = sl.run_searchlight(rsa, pool_size=pool_size)
else:
    sl_result = sl.run_searchlight(decode, pool_size=pool_size)
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

