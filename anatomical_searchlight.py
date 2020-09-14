# Run a whole brain searchlight

# Import libraries
import nibabel as nib 
import numpy as np 
from mpi4py import MPI 
from brainiak.searchlight.searchlight import Searchlight
from scipy.stats.stats import pearsonr 
import sys 



# What subject are you running

sub=sys.argv[1]
layer_dir=sys.argv[2]
results_dir=sys.argv[3]
data_dir=sys.argv[4]


output_name = (results_dir+'/%s_whole_brain_anatomical_SL.nii.gz' % (sub))  
#output_mask_name=(d+'results/'+results_dir_name+'/%s_whole_brain_anatomical_SL_mask.nii.gz' % (sub))

# Get information
print("Loading data")
nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here


# Preset the variables

data = nii.get_data()  
big_mask=nib.load(data_dir+"whole_brain_mask.nii.gz").get_data() 
raw_rsm=np.genfromtxt(layer_dir,delimiter=',')[:495,:495] 
bcvar=raw_rsm[np.triu(np.ones(raw_rsm.shape),k=10).astype('bool')] 



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

#Kernel function for searchlight 
def rsa(data,mask,myrad,bcvar): 
    if np.sum(mask)<2:
        return 0.0

    data4D=data[0]
    mask=mask.astype('bool')
    bolddata_sl=data4D[mask,:].T

    
    human=np.corrcoef(bolddata_sl[:495,:]) 
    vec=human[np.triu(np.ones(human.shape),k=10).astype('bool')]
    return pearsonr(vec,bcvar)[0]  

print("Running Searchlight")
sl_result = sl.run_searchlight(decode, pool_size=pool_size)
#print("End SearchLight")

# Only save the data if this is the first core
if rank == 0: 

    # Convert the output into what can be used
    sl_result = sl_result.astype('double')
    sl_result[np.isnan(sl_result)] = 0  # If there are nans we want this

    # Save the volume
    sl_nii = nib.Nifti1Image(sl_result, affine_mat)
    nib.save(sl_nii, output_name)  # Save
    #np.save(output_mask_name,big_mask) 
    print('Finished searchlight') 
