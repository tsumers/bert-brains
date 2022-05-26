import nibabel as nib 
import numpy as np 
from himalaya.ridge import GroupRidgeCV
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
from scipy.stats import zscore 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from sklearn.metrics import make_scorer 
from himalaya.scoring import correlation_score,correlation_score_split
import pandas as pd 
import pickle 

sub=sys.argv[1]
data_dir=sys.argv[2]
partial=False  
if partial:
	p1=int(sys.argv[3])
	p2=int(sys.argv[4])

if 'black' in data_dir:
	layer_dir='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_all_z_representations.npy'
	save_dir='/jukebox/griffiths/bert-brains/results/black/encoding_headwise/'
else: 
	layer_dir='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_all_z_representations.npy'
	save_dir='/jukebox/griffiths/bert-brains/results/slumlordreach/encoding_headwise/'

layer_name='full_z'  


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
	primary_features=np.hstack([phoneme_vectors,phoneme_counts,word_counts,silent])

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
	
	#raw_features=ffill_array(raw_features)
	raw_features[raw_silent_features[:,0].astype('bool')]=0.0
	
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
	primary_features=np.hstack([phoneme_vectors,phoneme_counts,word_counts,silent])
	print(primary_features.shape)

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

	#raw_features=ffill_array(raw_features) 
	raw_features[raw_silent_features[:,0].astype('bool')]=0.0


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


exponents=list(range(-25,26,2))
alphas=[10**e for e in exponents]

is_primary_features=is_primary[0,:].astype('bool')
X=features[:,~is_primary_features] 
X0_full=features[:,is_primary_features]

is_silent_features=np.zeros(X0_full.shape[1])
is_silent_features[42]=1
is_silent_features[42+43]=1
is_silent_features[42+43*2]=1
is_silent_features[42+43*3]=1 

is_number_features=np.zeros(X0_full.shape[1])
is_number_features[41]=1
is_number_features[40]=1
is_number_features[41+43]=1
is_number_features[41+43*2]=1
is_number_features[41+43*3]=1 
is_number_features[40+43]=1
is_number_features[40+43*2]=1
is_number_features[40+43*3]=1 

is_phoneme_features=np.zeros(X0_full.shape[1])
for i in range(is_phoneme_features.shape[0]):
	if is_number_features[i]==0 and is_silent_features[i]==0:
		is_phoneme_features[i]=1 


is_silent_features=is_silent_features.astype('bool')
is_number_features=is_number_features.astype('bool')
is_phoneme_features=is_phoneme_features.astype('bool')





X0=X0_full[:,is_silent_features]
X1=X0_full[:,is_number_features]
X2=X0_full[:,is_phoneme_features]


Y=data.T 
if partial:
	Y=Y[:,p1:p2]

	

skf_outer=KFold(n_splits=3,shuffle=False)
skf_inner=KFold(n_splits=3,shuffle=False)
test_idx=0

if partial:
	performances=np.zeros((Y.shape[1],3,144))
	alphas=np.zeros((Y.shape[1],3,4)) 

else:
	performances=np.zeros((1000,3,144))
	alphas=np.zeros((1000,3,4))


for train_index,test_index in skf_outer.split(X):
	X_train,X_test=X[train_index],X[test_index]
	X0_train,X0_test=X0[train_index],X0[test_index]
	X1_train,X1_test=X1[train_index],X1[test_index]
	X2_train,X2_test=X2[train_index],X2[test_index] 
	Y_train,Y_test=Y[train_index],Y[test_index]

	X_train=(X_train-X_train.mean(axis=0))/(X_train.std(axis=0,ddof=1)+1e-9)
	X_test=(X_test-X_train.mean(axis=0))/(X_train.std(axis=0,ddof=1)+1e-9)  

	#X0_train=(X0_train-X0_train.mean(axis=0))/(X0_train.std(axis=0,ddof=1)+1e-9)
	#X0_test=(X0_test-X0_train.mean(axis=0))/(X0_train.std(axis=0,ddof=1)+1e-9)   

	X1_train=(X1_train-X1_train.mean(axis=0))/(X1_train.std(axis=0,ddof=1)+1e-9)
	X1_test=(X1_test-X1_train.mean(axis=0))/(X1_train.std(axis=0,ddof=1)+1e-9)

	#X2_train=(X2_train-X2_train.mean(axis=0))/(X2_train.std(axis=0,ddof=1)+1e-9)
	#X2_test=(X2_test-X2_train.mean(axis=0))/(X2_train.std(axis=0,ddof=1)+1e-9)    

	
	

	model=GroupRidgeCV(groups="input",cv=skf_inner,fit_intercept=False,solver_params=dict(score_func=correlation_score,progress_bar=True,n_iter=100))
	model.fit([X_train,X0_train,X1_train,X2_train],Y_train) 


	alphas[:,test_idx,:]=np.exp(model.deltas_).T

	original_coef=model.coef_.copy()
	
	assert original_coef.shape[0]==9259*4
	for head in range(144):
		hnum_coef=original_coef.copy() 
		for zero_head in range(144):
			if zero_head!=head:
				for d in range(4):
					start=64*zero_head 
					offset=9259*d 
					hnum_coef[offset+start:offset+start+64,:]=0.0 
		model.coef_=hnum_coef 
		Y_pred=model.predict([X_test,X0_test,X1_test,X2_test],split=True)[0]
		
		for p in range(performances.shape[0]):  
			performances[p,test_idx,head]=pearsonr(Y_pred[:,p],Y_test[:,p])[0] 
	test_idx+=1  
	
performances[np.isnan(performances)]=0.0
alphas[np.isnan(alphas)]=0.0
raw_results=performances.mean(axis=1)
alphas=alphas.mean(axis=1)
if partial: 
	np.save(save_dir+sub+"_parcelwise_results_banded_ridge_partial_"+str(p1)+"_"+str(p2)+".npy",raw_results) 
	np.save(save_dir+sub+"_parcelwise_results_banded_ridge_alphas_partial_"+str(p1)+"_"+str(p2)+".npy",alphas) 
else:
	np.save(save_dir+sub+"_parcelwise_results_banded_ridge.npy",raw_results) 
	np.save(save_dir+sub+"_parcelwise_results_banded_ridge_alphas.npy",alphas) 

