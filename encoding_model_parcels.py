import nibabel as nib 
import numpy as np 
from sklearn.linear_model import LinearRegression,Ridge
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
from scipy.stats import zscore 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
import pandas as pd 
sub=sys.argv[1]
layer_dir=sys.argv[2]
layer_name=sys.argv[3]
data_dir=sys.argv[4]
begin_trim=int(sys.argv[5])
end_trim=int(sys.argv[6]) 
save_dir=sys.argv[7] 

zero_intercept=False 

def ffill_array(arr):
	arr2=arr.copy().astype('float')
	arr2[np.sum(arr,axis=1)==0]=np.nan 
	df=pd.DataFrame(arr2)
	df2=df.ffill()
	return df2.to_numpy()


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
	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/'
	phoneme_counts=np.load(data_prefix+"slumlordreach_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"slumlordreach_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"slumlordreach_phoneme_vectors.npy")
	silent=((phoneme_counts[:,0]==0).astype('int')).reshape((-1,1))
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts,silent])
	pca=PCA(n_components=1)
	silent=pca.fit_transform(primary_features).reshape((-1,1))

	#primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])

	load_features=np.load(layer_dir,allow_pickle=True)
	if len(load_features.shape)==1:
		load_features=np.reshape(load_features,(-1,1))
	raw_features=[] 
	raw_primary_features=[]
	raw_silent_features=[]
	for i in range(load_features.shape[0]):
		if load_features[i] is not None and len(load_features[i])>0 and load_features[i][0] is not None:
			raw_primary_features.append(primary_features[i])
			raw_silent_features.append(silent[i])
			if 'semantic_composition' in layer_dir or 'syntactic_complexity' in layer_dir:
				raw_features.append(np.asarray(load_features[i][0]))
			else:
				raw_features.append(load_features[i]) 
	raw_features=np.vstack(raw_features)
	raw_primary_features=np.vstack(raw_primary_features) 
	raw_silent_features=np.vstack(raw_silent_features)
	#raw_features[raw_silent_features[:,0]==1]=0
	
	raw_features=ffill_array(raw_features)
	
	num_primary=primary_features.shape[1]

	raw_features=np.hstack([raw_features,raw_primary_features])

	shifted=[]
	is_primary_lst=[]
	shifted_silent=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		is_primary_lst.append(arr_prim)
		shifted.append(arr)

		arr_shifted=np.zeros((raw_silent_features.shape[0]+5,raw_silent_features.shape[1]))
		arr_shifted[d:raw_silent_features.shape[0]+d,:]=raw_silent_features
		shifted_silent.append(arr_shifted) 

	features=np.hstack(shifted)
	is_primary=np.hstack(is_primary_lst)
	silent_features=np.hstack(shifted_silent)

	begin_delay=3+(1192-raw_features.shape[0])

	splice1=619-begin_delay
	splice2=644-begin_delay  

	load_data=nii.get_fdata()[:,:,:,begin_delay:1205]  

	#load_data=np.concatenate([zscore(load_data[:,:,:,:splice1],axis=3,ddof=1),zscore(load_data[:,:,:,splice2:],axis=3,ddof=1)],axis=3)
	#load_data[np.isnan(load_data)]=0.0
	#features=np.concatenate([features[:splice1,:],features[splice2:,:]],axis=0) 
	#is_primary=np.concatenate([is_primary[:splice1,:],is_primary[splice2:,:]],axis=0)   
	#print(load_data.shape)
	load_data=zscore(load_data[:,:,:,:splice1],axis=3,ddof=1)
	#print(load_data.shape)
	load_data[np.isnan(load_data)]=0.0
	#print(load_data.shape)
	features=features[:splice1,:] 
	is_primary=is_primary[:splice1,:] 
	silent_features=silent_features[:splice1,:]

	features=features[10:-10,:]
	is_primary=is_primary[10:-10,:] 
	raw_data=load_data[:,:,:,10:-10] 
	silent_features=silent_features[10:-10,:]

	#include=(np.sum(silent_features,axis=1)==0).astype('bool') 
	#features=features[include]
	#is_primary=is_primary[include]
	#raw_data=raw_data[:,:,:,include]
	print(raw_data.shape,features.shape)



	trailing=raw_data.shape[3]-features.shape[0]
	if trailing>0:
		raw_data=raw_data[:,:,:,:-trailing] 
	

	assert raw_data.shape[3]==features.shape[0]
	assert is_primary.shape==features.shape 
	row_equivalence=True 
	for row in is_primary:
		if np.sum(row!=is_primary[0])>0:
			row_equivalence=False 
	assert row_equivalence

	val_size=70

elif 'black' in data_dir:
	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/'
	phoneme_counts=np.load(data_prefix+"black_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"black_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"black_phoneme_vectors.npy")
	silent=((phoneme_counts[:,0]==0).astype('int')).reshape((-1,1))

	#embedding_layer=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_12_activations.npy')
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts,silent])
	pca=PCA(n_components=1)
	silent=pca.fit_transform(primary_features).reshape((-1,1))

	#primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts]) 

	load_features=np.load(layer_dir,allow_pickle=True) 
	if len(load_features.shape)==1:
		load_features=np.reshape(load_features,(-1,1))
	raw_features=[]
	raw_primary_features=[]
	raw_silent_features=[]
	for i in range(load_features.shape[0]):
		if load_features[i] is not None and len(load_features[i])>0 and load_features[i][0] is not None:
			raw_primary_features.append(primary_features[i])
			raw_silent_features.append(silent[i])
			if 'semantic_composition' in layer_dir or 'syntactic_complexity' in layer_dir:
				raw_features.append(np.asarray(load_features[i][0]))
			else:
				raw_features.append(load_features[i])
	
	raw_features=np.vstack(raw_features)
	raw_primary_features=np.vstack(raw_primary_features)
	raw_silent_features=np.vstack(raw_silent_features)
	#raw_features[raw_silent_features[:,0]==1]=0
	begin_delay=534-raw_features.shape[0]

	raw_features=ffill_array(raw_features) 


	raw_features=np.hstack([raw_features,raw_primary_features])
	num_primary=raw_primary_features.shape[1]

	shifted=[]
	is_primary_lst=[]
	shifted_silent=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		is_primary_lst.append(arr_prim)
		shifted.append(arr)

		arr_shifted=np.zeros((raw_silent_features.shape[0]+5,raw_silent_features.shape[1]))
		arr_shifted[d:raw_silent_features.shape[0]+d,:]=raw_silent_features
		shifted_silent.append(arr_shifted)


	features=np.hstack(shifted)
	is_primary=np.hstack(is_primary_lst)

	silent_features=np.hstack(shifted_silent)


	load_data=nii.get_fdata()[:,:,:,8:-8]
	raw_data=load_data[:,:,:,begin_delay:]

	features=features[10:-10,:]
	raw_data=raw_data[:,:,:,10:-10]
	is_primary=is_primary[10:-10,:]
	silent_features=silent_features[10:-10,:]




	trailing=features.shape[0]-raw_data.shape[3]
	features=features[:-trailing]
	silent_features=silent_features[:-trailing]
	is_primary=is_primary[:-trailing] 
	
	#include=(np.sum(silent_features,axis=1)==0).astype('bool') 
	#features=features[include]
	#is_primary=is_primary[include]
	#raw_data=raw_data[:,:,:,include] 

	
	assert raw_data.shape[3]==features.shape[0]

	assert is_primary.shape==features.shape 

	row_equivalence=True 
	for row in is_primary:
		if np.sum(row!=is_primary[0])>0:
			row_equivalence=False 
	assert row_equivalence

	val_size=70 

	raw_data=zscore(raw_data,ddof=1,axis=3)
	raw_data[np.isnan(raw_data)]=0.0  



#features=np.asarray([np.hstack([raw_features[tr+2],raw_features[tr+3],raw_features[tr+4],raw_features[tr+5]]) for tr in range(raw_features.shape[0]-5)])
#features=np.asarray([raw_features[tr+3] for tr in range(raw_features.shape[0]-3)])
#data = nii.get_fdata()[:,:,:,begin_trim:end_trim-3]
parcellation_nii=nib.load(data_dir+"Schaefer1000_3mm.nii.gz")
affine=parcellation_nii.affine
parcellation=parcellation_nii.get_fdata().astype('int') 
num_parcels=1000
data=np.zeros((num_parcels,raw_data.shape[-1]))
for p in range(num_parcels):
	data[p,:]=raw_data[np.where(parcellation==p+1)].mean(axis=0)  



  

def process(i): 
	y=data[i,:].reshape((-1,1)) 
	is_primary_features=is_primary[0,:].astype('bool')
	X=features[:,~is_primary_features] 
	#X=features[:,:] 
	#X0=features[:,is_primary_features]
	X0=silent_features
	#if i==5:
	np.save(data_dir+"nuisance_features.npy",X0)
	np.save(data_dir+"real_ling_features.npy",X)
	np.save(data_dir+"sample_parcel.npy",y)

	#print(X.shape,y.shape,data.shape)
	test_performances=[] 
	skf=KFold(n_splits=3,shuffle=False)
	encoding_model_weights=[]
	tuned_alphas=[]  


	X0=(X0-X0.mean(axis=0))/(X0.std(axis=0,ddof=1)+1e-9) 
	resid_model=LinearRegression(normalize=False)
	resid_model.fit(X0,y)
	y=y-resid_model.predict(X0).reshape((-1,1))   
	
	for train_index,test_index in skf.split(X):

		X_train,X_test=X[train_index],X[test_index]
		y_train,y_test=y[train_index],y[test_index]

		


		#X_train,X_test=X[:-750],X[-750:]
		#y_train,y_test=y[:-750],y[-750:] 

		X_trainval,X_testval=X_train[:-val_size],X_train[-val_size:]
		y_trainval,y_testval=y_train[:-val_size],y_train[-val_size:] 
		exponents=list(range(-25,26,2))
		alphas=[10**e for e in exponents]
		val_performances=[] 
		for alpha in alphas:
			model=Ridge(alpha=alpha,normalize=False)
			X_trainval=(X_trainval-X_trainval.mean(axis=0))/(X_trainval.std(axis=0,ddof=1)+1e-9) 
			model.fit(X_trainval,y_trainval)
			X_testval=(X_testval-X_trainval.mean(axis=0))/(X_trainval.std(axis=0,ddof=1)+1e-9) 
			#model.coef_[:,is_primary_features]=0.0
			if zero_intercept:
				model.intercept_=0.0 
			y_hatval=model.predict(X_testval)
			val_performances.append(pearsonr(y_hatval[:,0],y_testval[:,0])[0])
		val_performances=np.asarray(val_performances)
		val_performances[np.isnan(val_performances)]=0.0
		tuned=alphas[np.argmax(val_performances)]  
		tuned_alphas.append(tuned)
		model=Ridge(alpha=tuned,normalize=False)  
		X_train=(X_train-X_train.mean(axis=0))/(X_train.std(axis=0,ddof=1)+1e-9) 
		model.fit(X_train,y_train)
		X_test=(X_test-X_train.mean(axis=0))/(X_train.std(axis=0,ddof=1)+1e-9)  
		#model.coef_[:,is_primary_features]=0.0
		if zero_intercept: 
			model.intercept_=0.0 
		#encoding_model_weights.append(model.coef_[:,~is_primary_features])
		encoding_model_weights.append(model.coef_[:,:])

		y_hat=model.predict(X_test)
		test_performances.append(pearsonr(y_hat[:,0],y_test[:,0])[0])
	mean_weights=np.mean(np.asarray(encoding_model_weights),axis=0)
	test_performances=np.asarray(test_performances)
	print(i,np.mean(test_performances)) 
	test_performances[np.isnan(test_performances)]=0.0
	
	return [np.mean(test_performances),mean_weights,tuned_alphas]  

raw_results=[] 
weights=[]
reg_lambdas=[]
#roi_mapping=np.load('/jukebox/griffiths/bert-brains/black_data/Parcel2ROI.npy')
for i in range(1000): 
	r=process(i)
	raw_results.append(r[0])
	weights.append(r[1])
	reg_lambdas.append(r[2]) 
	#print(r[2])
raw_results=np.asarray(raw_results) 
weights=np.asarray(weights) 
reg_lambdas=np.asarray(reg_lambdas) 
np.save(save_dir+sub+"_encoding_weights_full_residual.npy",weights) 
np.save(save_dir+sub+"_parcelwise_results_full_residual.npy",raw_results)  
np.save(save_dir+sub+"_regularization_coef_full_residual.npy",reg_lambdas)             

""" 
output_name=save_dir+sub+"_parcels_encoding.nii.gz"
results_volume=np.zeros(parcellation.shape)
for i in range(1000):
	results_volume[np.where(parcellation==i+1)]=raw_results[i]

result_nii=nib.Nifti1Image(results_volume,affine)  
nib.save(result_nii,output_name)          
""" 
