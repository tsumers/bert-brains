import nibabel as nib 
import numpy as np 
from sklearn.linear_model import LinearRegression
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
sub=sys.argv[1]
layer_dir=sys.argv[2]
layer_name=sys.argv[3]
data_dir=sys.argv[4]
begin_trim=int(sys.argv[5])
end_trim=int(sys.argv[6])
r1=int(sys.argv[7])
r2=int(sys.argv[8])

output_name="/scratch/sreejank/enc_"+sub+"_"+layer_name+"_"+str(r1)+"_"+str(r2)+".npy" 


nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here

data = nii.get_fdata()[:,:,:,begin_trim:end_trim]
big_mask=nib.load(data_dir+"whole_brain_mask.nii.gz").get_fdata().astype('bool')
data=data[big_mask]
#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1
raw_features=np.load(layer_dir)[:,:]
model=LinearRegression(normalize=True)

def process(i):
	y=data[i,:].reshape((-1,1))
	X=raw_features
	#print(X.shape,y.shape,data.shape)
	skf=KFold(n_splits=3)
	r=[]
	for train_index,test_index in skf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
		model.fit(X_train,y_train)
		preds=model.predict(X_test)
		#print(preds.shape,y_test.shape)
		r.append(pearsonr(y_test[:,0],preds[:,0])[0])
	return np.mean(r)


inputs=list(range(r1,r2))
results=np.asarray([process(i) for i in inputs])

np.save(output_name,results) 

