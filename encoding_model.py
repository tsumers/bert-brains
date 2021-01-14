import nibabel as nib 
import numpy as np 
from sklearn.linear_model import LinearRegression,Ridge
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
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

#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1
if 'syntactic_complexity' in layer_dir or 'syntactic_distance' in layer_dir:
	raw_features=np.load(layer_dir)[2:,:]
else:
	raw_features=np.load(layer_dir)[:,:]
features=np.asarray([np.hstack([raw_features[tr+2],raw_features[tr+3],raw_features[tr+4],raw_features[tr+5]]) for tr in range(raw_features.shape[0]-5)])
#features=np.asarray([raw_features[tr+3] for tr in range(raw_features.shape[0]-3)])
data = nii.get_fdata()[:,:,:,begin_trim:end_trim-5]
#data = nii.get_fdata()[:,:,:,begin_trim:end_trim-3]
big_mask=nib.load(data_dir+"whole_brain_mask.nii.gz").get_fdata().astype('bool')
data=data[big_mask]


  

def process(i):
	y=data[i,:].reshape((-1,1)) 
	X=features
	#print(X.shape,y.shape,data.shape)
	test_performances=[]
	#skf=KFold(n_splits=3,shuffle=False)
	#for train_index,test_index in skf.split(X):

	#X_train,X_test=X[train_index],X[test_index]
	#y_train,y_test=y[train_index],y[test_index]
	X_train,X_test=X[:-750],X[-750:]
	y_train,y_test=y[:-750],y[-750:] 

	X_trainval,X_testval=X_train[:-300],X_train[-300:]
	y_trainval,y_testval=y_train[:-300],y_train[-300:] 

	alphas=[0.0001,0.001,0.01,1.0]

	val_performances=[]
	for alpha in alphas:
		model=Ridge(alpha=alpha,normalize=False)
		X_trainval=(X_trainval-X_trainval.mean(axis=0))/(X_trainval.std(axis=0,ddof=1)+1e-9) 
		model.fit(X_trainval,y_trainval)
		X_testval=(X_testval-X_testval.mean(axis=0))/(X_testval.std(axis=0,ddof=1)+1e-9) 
		y_hatval=model.predict(X_testval)
		val_performances.append(pearsonr(y_hatval[:,0],y_testval[:,0])[0])

	tuned=alphas[np.argmax(val_performances)]
	model=Ridge(alpha=tuned,normalize=False) 
	X_train=(X_train-X_train.mean(axis=0))/(X_train.std(axis=0,ddof=1)+1e-9) 
	model.fit(X_train,y_train)
	X_test=(X_test-X_test.mean(axis=0))/(X_test.std(axis=0,ddof=1)+1e-9)  
	y_hat=model.predict(X_test)
	test_performances.append(pearsonr(y_hat[:,0],y_test[:,0])[0])

	return np.mean(test_performances) 


inputs=list(range(r1,r2))
results=np.asarray([process(i) for i in inputs])

np.save(output_name,results)   

