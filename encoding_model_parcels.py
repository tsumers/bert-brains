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
save_dir=sys.argv[7]




nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here

#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1
if '21styear' in data_dir:
	if 'syntactic_complexity' in layer_dir or 'syntactic_distance' in layer_dir:
		raw_features=np.load(layer_dir)[2:,:]
	else:
		raw_features=np.load(layer_dir)[:,:]
	raw_data = nii.get_fdata()[:,:,:,begin_trim:end_trim-5]

elif 'slumlordreach' in data_dir:
	load_features=np.load(layer_dir,allow_pickle=True)
	raw_features=[]
	for i in range(load_features.shape[0]):
		if load_features[i] is not None and len(load_features[i])>0:
			if 'semantic_composition_max_l2' in layer_dir:
				raw_features.append(load_features[i][0])
			else:
				raw_features.append(load_features[i])
	raw_features=np.vstack(raw_features)
	load_data=nii.get_fdata()[:,:,:,:1205]
	desired_size=raw_features.shape[0]
	curr_size=load_data.shape[3]
	raw_data=nii.get_fdata()[:,:,:,curr_size-desired_size:]




features=np.asarray([np.hstack([raw_features[tr+2],raw_features[tr+3],raw_features[tr+4],raw_features[tr+5]]) for tr in range(raw_features.shape[0]-5)])
#features=np.asarray([raw_features[tr+3] for tr in range(raw_features.shape[0]-3)])
#data = nii.get_fdata()[:,:,:,begin_trim:end_trim-3]
big_mask_nii=nib.load(data_dir+"whole_brain_mask.nii.gz")
big_mask=big_mask_nii.get_fdata().astype('bool')
affine=big_mask_nii.affine
parcellation=nib.load("/jukebox/griffiths/bert-brains/Schaefer1000_3mm.nii.gz").get_fdata().astype('int')
num_parcels=1000
data=np.zeros((num_parcels,raw_data.shape[-1]))
for p in range(num_parcels):
	data[p,:]=raw_data[np.where(parcellation==p+1)].mean(axis=0) 




  

def process(i):
	#print(i)
	y=data[i,:].reshape((-1,1)) 
	X=features
	#print(X.shape,y.shape,data.shape)
	test_performances=[]
	skf=KFold(n_splits=3,shuffle=False)
	for train_index,test_index in skf.split(X):

		X_train,X_test=X[train_index],X[test_index]
		y_train,y_test=y[train_index],y[test_index]
		#X_train,X_test=X[:-750],X[-750:]
		#y_train,y_test=y[:-750],y[-750:] 

		X_trainval,X_testval=X_train[:-300],X_train[-300:]
		y_trainval,y_testval=y_train[:-300],y_train[-300:] 

		alphas=[0.00001,0.0001,0.001,0.01,1.0]

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
		#print(test_performances[-1])

	return np.mean(test_performances) 




raw_results=np.asarray([process(i) for i in range(num_parcels)])
output_name=save_dir+sub+"_parcels_encoding.nii.gz"
results_volume=np.zeros(parcellation.shape)
for i in range(num_parcels):
	results_volume[np.where(parcellation==i+1)]=raw_results[i]

result_nii=nib.Nifti1Image(results_volume,affine)  
nib.save(result_nii,output_name)
