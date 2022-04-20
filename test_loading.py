import nibabel as nib 
import numpy as np 
from sklearn.linear_model import LinearRegression,Ridge
import sys
from sklearn.model_selection import KFold 
from scipy.stats import pearsonr
from scipy.stats import zscore 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

d='/jukebox/griffiths/bert-brains/'

sub='sub-145'
data_dir=d+"slumlordreach_data/"
save_dir=d+"results/slumlordreach/encoding_residuals/" 
lower_p=0
upper_p=100 
which=3


raw_features=[] 
for lnum in range(12): 
	if 'black' in data_dir:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_'+str(lnum)+"_z_representations.npy"
	else:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_layer_'+str(lnum)+"_z_representations.npy" 
	raw_features.append(np.load(fname)) 
raw_features_init=np.hstack(raw_features)


nii=nib.load(data_dir+sub+".nii.gz")
affine_mat = nii.affine  # What is the data transformation used here

#big_mask=np.zeros(big_mask.shape)
#big_mask[40,30,40]=1


if 'slumlordreach' in data_dir:
	load_features2=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/syntactic_analyses/slumlordreach_bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat.npy',allow_pickle=True)
	layer_dir='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/syntactic_analyses/slumlordreach_bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat.npy'


	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/'
	phoneme_counts=np.load(data_prefix+"slumlordreach_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"slumlordreach_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"slumlordreach_phoneme_vectors.npy")
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])

	raw_primary_features=[]
	num_primary=primary_features.shape[1]

	load_features2=np.load(layer_dir,allow_pickle=True)
	if len(load_features2.shape)==1:
		load_features2=np.reshape(load_features2,(-1,1))
	raw_features=[]
	raw_features2=[]
	for i in range(load_features2.shape[0]):
		if load_features2[i] is not None and len(load_features2[i])>0 and load_features2[i][0] is not None :
			raw_primary_features.append(primary_features[i])
			raw_features2.append(load_features2[i][0])
			raw_features.append(raw_features_init[i])

	raw_features=np.vstack(raw_features)
	raw_features2=np.vstack(raw_features2)


	raw_features=np.hstack([raw_features,raw_primary_features])
	raw_features2=np.hstack([raw_features2,raw_primary_features])

	assert raw_features.shape[0]==raw_features2.shape[0]


	shifted=[]
	shifted2=[]
	is_primary_lst=[]
	is_primary_lst2=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr2=np.zeros((raw_features2.shape[0]+5,raw_features2.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim2=np.zeros(arr2.shape)
		arr_prim[:,-num_primary:]=1
		arr_prim2[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		arr2[d:raw_features2.shape[0]+d,:]=raw_features2
		is_primary_lst.append(arr_prim)
		is_primary_lst2.append(arr_prim2)
		shifted.append(arr)
		shifted2.append(arr2)

	features=np.hstack(shifted)
	features2=np.hstack(shifted2)
	is_primary=np.hstack(is_primary_lst)
	is_primary2=np.hstack(is_primary_lst2)

	assert raw_features.shape[0]==raw_features2.shape[0]
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
	features2=features2[:splice1,:] 
	is_primary=is_primary[:splice1,:] 
	is_primary2=is_primary2[:splice1,:]

	features=features[10:-10,:]
	features2=features2[10:-10,:]
	is_primary=is_primary[10:-10,:] 
	is_primary2=is_primary2[10:-10,:] 
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
	load_features2=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/syntactic_analyses/black_bert-base-uncased_syntactic_complexity_L-inf_T-128_D-concat.npy',allow_pickle=True)

	
	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/'
	phoneme_counts=np.load(data_prefix+"black_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"black_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"black_phoneme_vectors.npy")
	#embedding_layer=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_12_activations.npy')
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])

	
	raw_primary_features=[]
	num_primary=primary_features.shape[1]

	if len(load_features2.shape)==1:
		load_features2=np.reshape(load_features2,(-1,1))
	raw_features=[]
	raw_features2=[]
	for i in range(load_features2.shape[0]):
		if load_features2[i] is not None and len(load_features2[i])>0 and load_features2[i][0] is not None :
			raw_primary_features.append(primary_features[i])
			raw_features2.append(load_features2[i][0])
			raw_features.append(raw_features_init[i])
	raw_features=np.vstack(raw_features)
	raw_features2=np.vstack(raw_features2)

	raw_features=np.hstack([raw_features,raw_primary_features])
	raw_features2=np.hstack([raw_features2,raw_primary_features])

	assert raw_features.shape[0]==raw_features2.shape[0]

	begin_delay=534-raw_features.shape[0]
	

	shifted=[]
	shifted2=[]
	is_primary_lst=[]
	is_primary_lst2=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0]+5,raw_features.shape[1]))
		arr2=np.zeros((raw_features2.shape[0]+5,raw_features2.shape[1]))
		arr_prim=np.zeros(arr.shape)
		arr_prim2=np.zeros(arr2.shape)
		arr_prim[:,-num_primary:]=1
		arr_prim2[:,-num_primary:]=1
		arr[d:raw_features.shape[0]+d,:]=raw_features
		arr2[d:raw_features2.shape[0]+d,:]=raw_features2
		is_primary_lst.append(arr_prim)
		is_primary_lst2.append(arr_prim2)
		shifted.append(arr)
		shifted2.append(arr2)

	features=np.hstack(shifted)
	features2=np.hstack(shifted2)
    
	is_primary=np.hstack(is_primary_lst)
	is_primary2=np.hstack(is_primary_lst2)
	assert features.shape[0]==features2.shape[0]
	assert is_primary.shape[0]==is_primary.shape[0]

	print(features.shape,features2.shape)
	load_data=nii.get_fdata()[:,:,:,8:-8]
	raw_data=load_data[:,:,:,begin_delay:]

	features=features[10:-10,:]
	features2=features2[10:-10,:]
	raw_data=raw_data[:,:,:,10:-10]
	is_primary=is_primary[10:-10,:]
	is_primary2=is_primary2[10:-10,:]

	print(features.shape,features2.shape)

	trailing=features.shape[0]-raw_data.shape[3]
	features=features[:-trailing]
	features2=features2[:-trailing]
	is_primary=is_primary[:-trailing] 
	is_primary2=is_primary2[:-trailing] 
	
	print(features.shape,features2.shape)
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

assert features.shape[0]==features2.shape[0]
assert is_primary.shape[0]==is_primary2.shape[0]