import nibabel as nib
import numpy as np
from scipy.stats.stats import pearsonr
import sys

sub=sys.argv[1]
layer_dir=sys.argv[2]
layer_name=sys.argv[3]
data_dir=sys.argv[4]
begin_trim=int(sys.argv[5])
end_trim=int(sys.argv[6])
r1=int(sys.argv[7])
r2=int(sys.argv[8])

space=np.load(data_dir+"functional_spaces/"+sub+".npy")

output_name="/scratch/sreejank/"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy" 


nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here


# Preset the variables

data = nii.get_fdata()[:,:,:,begin_trim:end_trim]
data=data[:,:,:,960:]
big_mask=nib.load(data_dir+"whole_brain_mask.nii.gz").get_fdata().astype('bool')
data=data[big_mask]
#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1
raw_rsm=np.load(layer_dir)[960:,960:]
bcvar=raw_rsm[np.triu(np.ones(raw_rsm.shape),k=3).astype('bool')]

def rsa(bolddata,bcvar):
    bolddata_sl=bolddata.T 
    human=np.corrcoef(bolddata_sl[:,:])
    vec=human[np.triu(np.ones(human.shape),k=3).astype('bool')]
    vec[np.isnan(vec)]=0
    return pearsonr(vec,bcvar)[0]

def process(i):
	if space[i,0]<0:
		return np.nan 
	else:
		idx=space[i].astype('int')
		#print(data.shape,space.shape,idx)
		return rsa(data[idx,:],bcvar)

inputs=list(range(r1,r2))
results=np.asarray([process(i) for i in inputs])

np.save(output_name,results)


