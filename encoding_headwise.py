import nibabel as nib 
import numpy as np 
from sklearn.linear_model import LinearRegression,Ridge
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
from scipy.stats import zscore 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

sub=sys.argv[1] 
data_dir=sys.argv[2]
save_dir=sys.argv[3] 
lower_p=int(sys.argv[4]) 
upper_p=int(sys.argv[5]) 

raw_features=[] 
for lnum in range(12): 
	if 'black' in data_dir:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_'+str(lnum)+"_z_representations.npy"
	else:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_layer_'+str(lnum)+"_z_representations.npy" 
	raw_features.append(np.load(fname)) 
raw_features=np.hstack(raw_features)   


nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here

#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1

if 'slumlordreach' in data_dir:
	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/'
	phoneme_counts=np.load(data_prefix+"slumlordreach_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"slumlordreach_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"slumlordreach_phoneme_vectors.npy")
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])

	raw_primary_features=primary_features
	num_primary=primary_features.shape[1]

	raw_features=np.hstack([raw_features,raw_primary_features])

	shifted=[]
	is_primary_lst=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		is_primary_lst.append(arr_prim)
		shifted.append(arr)
	features=np.hstack(shifted)
	is_primary=np.hstack(is_primary_lst)

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

	features=features[10:-10,:]
	is_primary=is_primary[10:-10,:] 
	raw_data=load_data[:,:,:,10:-10]



	trailing=raw_data.shape[3]-features.shape[0]
	if trailing>0:
		raw_data=raw_data[:,:,:,:-trailing] 
	print(raw_data.shape,features.shape)

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
	#embedding_layer=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_12_activations.npy')
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])

	
	raw_primary_features=primary_features
	begin_delay=534-raw_features.shape[0]
	raw_features=np.hstack([raw_features,raw_primary_features])
	num_primary=raw_primary_features.shape[1]

	shifted=[]
	is_primary_lst=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		is_primary_lst.append(arr_prim)
		shifted.append(arr)
	features=np.hstack(shifted)
	is_primary=np.hstack(is_primary_lst)

	load_data=nii.get_fdata()[:,:,:,8:-8]
	raw_data=load_data[:,:,:,begin_delay:]

	features=features[10:-10,:]
	raw_data=raw_data[:,:,:,10:-10]
	is_primary=is_primary[10:-10,:]



	trailing=features.shape[0]-raw_data.shape[3]
	features=features[:-trailing]
	is_primary=is_primary[:-trailing] 
	
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
	parcel_number=i
	y=data[i,:].reshape((-1,1)) 
	X=features
	is_primary_features=is_primary[0,:].astype('bool')

	#print(X.shape,y.shape,data.shape)
	test_performances=np.zeros((144,3))
	original_test_performance=np.zeros((3,))
	#test_performances=np.zeros((3,))
	skf=KFold(n_splits=3,shuffle=False)
	tuned_alphas=[]
	test_idx=0  
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
			model.coef_[:,is_primary_features]=0.0
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
		model.coef_[:,is_primary_features]=0.0

		original_coef=model.coef_.copy()

		y_hat=model.predict(X_test)
		original_test_performance[test_idx]=pearsonr(y_hat[:,0],y_test[:,0])[0]
		print("SHAPE",X_train.shape,original_coef.shape)
		for head in range(144):
			hnum_coef=original_coef.copy() 
			for zero_head in range(144):
				if zero_head!=head:
					for d in range(4): 
						start=64*zero_head 
						offset=9258*d 
						hnum_coef[:,offset+start:offset+start+64]=0.0   
			model.coef_=hnum_coef 
			y_hat=model.predict(X_test)
			head_test_performance=pearsonr(y_hat[:,0],y_test[:,0])[0]
			test_performances[head,test_idx]=head_test_performance
			print(head,test_idx,test_performances[head,test_idx])  
		

		test_idx+=1 

	test_performances[np.isnan(test_performances)]=0.0  
	original_test_performance[np.isnan(original_test_performance)]=0.0
	#return np.mean(test_performances,axis=1)
	#print(parcel_number,np.mean(test_performances,axis=1))
	print(parcel_number,np.mean(original_test_performance))
	return np.mean(test_performances,axis=1),np.mean(original_test_performance)  
	#return np.mean(original_test_performance) 



raw_results_headwise=[] 
raw_results_total=[]
#roi_mapping=np.load('/jukebox/griffiths/bert-brains/black_data/Parcel2ROI.npy')
for i in range(lower_p,upper_p):
	headwise,total=process(i)
	raw_results_headwise.append(headwise) 
	#total=process(i)
	raw_results_total.append(total)

raw_results_headwise=np.asarray(raw_results_headwise)
raw_results_total=np.asarray(raw_results_total)   


np.save(save_dir+sub+"_partial_headwise_result_"+str(lower_p)+"_"+str(upper_p)+".npy",raw_results_headwise) 

#np.save(save_dir+sub+"_partial_total_result_"+str(lower_p)+"_"+str(upper_p)+".npy",raw_results_total)  










