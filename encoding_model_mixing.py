from scipy.optimize import minimize
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

raw_features=[]
for lnum in range(13):
	if 'black' in data_dir:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_'+str(lnum)+"_activations.npy"
	else:
		fname='/jukebox/griffiths/bert-brains/code/bert-brains/data/slumlordreach/bert-base-uncased/raw_embeddings/slumlordreach_bert-base-uncased_layer_'+str(lnum)+"_activations.npy"
	raw_features.append(np.load(fname))
raw_features=np.asarray(raw_features)  







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


	#load_features=np.load(layer_dir,allow_pickle=True)
	raw_primary_features=primary_features
	num_primary=primary_features.shape[1]

	

	shifted=[]
	shifted_primary=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0],raw_features.shape[1]+5,raw_features.shape[2]))
		arr_prim=np.zeros((raw_features.shape[1]+5,raw_primary_features.shape[1]))
		arr_prim[d:raw_features.shape[1]+d,:]=raw_primary_features
		arr[:,d:raw_features.shape[1]+d,:]=raw_features
		shifted_primary.append(arr_prim)
		shifted.append(arr)
	shifted=np.asarray(shifted)
	features=np.asarray([np.hstack(shifted[:,i]) for i in range(shifted.shape[1])])
	features_primary=np.hstack(shifted_primary)

	begin_delay=3+(1192-raw_features.shape[1])

	splice1=619-begin_delay
	splice2=644-begin_delay  

	load_data=nii.get_fdata()[:,:,:,begin_delay:1205]  

	load_data=np.concatenate([zscore(load_data[:,:,:,:splice1],axis=3,ddof=1),zscore(load_data[:,:,:,splice2:],axis=3,ddof=1)],axis=3)
	load_data[np.isnan(load_data)]=0.0
	features=np.concatenate([features[:,:splice1,:],features[:,splice2:,:]],axis=1) 
	features_primary=np.concatenate([features_primary[:splice1,:],features_primary[splice2:,:]],axis=0)   

	features=features[:,10:-10,:]
	features_primary=features_primary[10:-10,:] 
	raw_data=load_data[:,:,:,10:-10]



	trailing=raw_data.shape[3]-features.shape[1]
	raw_data=raw_data[:,:,:,:-trailing] 
	#print(raw_data.shape,features.shape)

	#print(features.shape,features_primary.shape,raw_data.shape)
	assert raw_data.shape[3]==features.shape[1]
	assert features_primary.shape[0]==features.shape[1] 
	



	val_size=300

elif 'black' in data_dir:
	data_prefix='/jukebox/griffiths/bert-brains/code/bert-brains/data/black/'
	phoneme_counts=np.load(data_prefix+"black_phoneme_counts.npy").reshape((-1,1))
	word_counts=np.load(data_prefix+"black_word_counts.npy").reshape((-1,1))
	phoneme_vectors=np.load(data_prefix+"black_phoneme_vectors.npy")
	embedding_layer=np.load('/jukebox/griffiths/bert-brains/code/bert-brains/data/black/bert-base-uncased/raw_embeddings/black_bert-base-uncased_layer_12_activations.npy')
	primary_features=np.hstack([phoneme_counts,phoneme_vectors,word_counts])


	#load_features=np.load(layer_dir,allow_pickle=True)
	
	raw_primary_features=primary_features
	begin_delay=534-raw_features.shape[1]



	num_primary=raw_primary_features.shape[1]

	shifted=[]
	shifted_primary=[]
	delays=[2,3,4,5]
	for d in delays:
		arr=np.zeros((raw_features.shape[0],raw_features.shape[1]+5,raw_features.shape[2]))
		arr_prim=np.zeros((raw_features.shape[1]+5,raw_primary_features.shape[1]))
		arr_prim[d:raw_features.shape[1]+d,:]=raw_primary_features
		arr[:,d:raw_features.shape[1]+d,:]=raw_features
		shifted_primary.append(arr_prim)
		shifted.append(arr)
	shifted=np.asarray(shifted)
	features=np.asarray([np.hstack(shifted[:,i]) for i in range(shifted.shape[1])])
	features_primary=np.hstack(shifted_primary)

	load_data=nii.get_fdata()[:,:,:,8:-8]
	raw_data=load_data[:,:,:,begin_delay:]

	features=features[:,10:-10,:]
	raw_data=raw_data[:,:,:,10:-10]
	features_primary=features_primary[10:-10,:]



	trailing=features.shape[1]-raw_data.shape[3]
	features=features[:,:-trailing,:]
	features_primary=features_primary[:-trailing] 
	

	print(features.shape,features_primary.shape,raw_data.shape)
	assert raw_data.shape[3]==features.shape[1] 
	assert features_primary.shape[0]==features.shape[1] 
	



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
	def loss(params):
		mixing_weights=params[:13]
		alpha=params[13]
		model=Ridge(alpha=alpha,normalize=False)
		mixed_features=np.zeros(features.shape[1:])
		for i in range(features.shape[0]):
			mixed_features+=mixing_weights[i]*features[i]
		X=np.hstack([mixed_features,features_primary])
		X=X-X.mean(axis=0)/(X.std(axis=0,ddof=1)+1e-9)
		model.fit(X,y)
		model.coef_[:,-features_primary.shape[1]:]=0.0 
		preds=model.predict(X)
		return np.sum((preds-y)**2.0)
	
	cons=({'type':'eq','fun':lambda x: np.sum(x[:13])-1})
	bounds=[(0.0,1.0) for _ in range(13)]
	bounds.append((None,None))
	x0=np.ones((14,))*(1.0/13)
	x0[-1]=1.0
	res=minimize(loss,x0,bounds=bounds,constraints=cons)
	
	optimized_mixing_weights=res.x[:13]
	center_of_mass=np.sum(optimized_mixing_weights*np.arange(13))
	return center_of_mass
	


raw_results=[]
weights=[]
for i in range(num_parcels):
	r=process(i)
	raw_results.append(r)
raw_results=np.asarray(raw_results)
np.save(save_dir+sub+"_parcelwise_results.npy",raw_results)  
output_name=save_dir+sub+"_parcels_encoding.nii.gz"
results_volume=np.zeros(parcellation.shape)
for i in range(num_parcels):
	results_volume[np.where(parcellation==i+1)]=raw_results[i]

result_nii=nib.Nifti1Image(results_volume,affine)  
nib.save(result_nii,output_name)        
